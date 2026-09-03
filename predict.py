"""
Interactive inference -- score one headline/image pair with the trained artefact.

Every other stage in this pipeline evaluates the artefact in aggregate, over a
held-out test split. This script does the complementary thing a viva or a demo
actually needs: point it at one image and one headline and get a verdict, using
the same Consistency-Gated Fusion model and the same frozen encoders as the
rest of the study.

    python -m src.predict --image path/to/photo.jpg \\
                          --headline "Scientists discover water on the sun"

The first call trains and caches a checkpoint (a few minutes on CPU); every
call after that loads it and answers in seconds. By default it builds the
report's proposed system (CGF + LLM + adversarial) if the LLM features have
been exported, and the plain CGF model otherwise; override with --variant.

Caveat, stated plainly rather than hidden: five of the model's nine metadata
features (post score, comment count, upvote ratio, whether the author field
was present, whether it was a self-post) describe *platform behaviour around*
a post, which does not exist for an image and headline typed in from outside
Reddit. Left unspecified, they default to the training corpus's median values
-- --score, --comments, --upvote-ratio and --has-author let you supply the
real figures for an actual Reddit post you are checking. The other four
metadata features (title length in characters and words, hour, day of week)
are computed from the headline and the current time, same as at training.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                    # noqa: E402
from src.build_dataset import META_FEATURES                    # noqa: E402
from src.train import (build_inputs, build_model, load_data,   # noqa: E402
                       train_one, variant_spec)

CKPT_DIR = C.RESULTS / "checkpoints"
N_DISTRACTORS = 128
DISTRACTOR_SEED = 42

LLM_VARIANTS = {"cgf_llm", "cgf_llm_adv"}
CGF_VARIANTS = ["cgf", "cgf_no_gate", "cgf_no_cons", "cgf_no_inter",
                "cgf_no_meta", "cgf_adv", "cgf_llm", "cgf_llm_adv"]


# --------------------------------------------------------------------------- #
# Single-example encoders. These mirror src/features.py exactly (same models,
# same pooling, same normalisation) but encode one item instead of a batch, so
# a demo does not need the whole corpus re-encoded to answer one question.
# --------------------------------------------------------------------------- #
def encode_clip_one(image_path: str, headline: str, device: str):
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    from src.features import _tensor

    if not Path(image_path).exists():
        raise SystemExit(f"[predict] no image found at '{image_path}' -- "
                         f"check the path and try again")
    model = CLIPModel.from_pretrained(C.CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(C.CLIP_MODEL)
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise SystemExit(f"[predict] could not open '{image_path}' as an "
                         f"image: {e}")

    ii = proc(images=[img], return_tensors="pt").to(device)
    ti = proc(text=[headline], return_tensors="pt", padding=True,
              truncation=True, max_length=77).to(device)
    with torch.no_grad():
        ie = torch.nn.functional.normalize(
            _tensor(model.get_image_features(**ii)), dim=-1)
        te = torch.nn.functional.normalize(
            _tensor(model.get_text_features(**ti)), dim=-1)
    del model
    return ie[0].float().cpu().numpy(), te[0].float().cpu().numpy()


def encode_text_one(headline: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    mdl = AutoModel.from_pretrained(C.TEXT_MODEL).to(device).eval()
    enc = tok([headline], return_tensors="pt", padding=True,
              truncation=True, max_length=C.MAX_TEXT_TOKENS).to(device)
    with torch.no_grad():
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        emb = ((h * m).sum(1) / m.sum(1))[0]
    del mdl
    return emb.float().cpu().numpy()


def clip_retrieval_rank(image_emb: np.ndarray, text_emb: np.ndarray,
                        distractor_text_embs: np.ndarray) -> float:
    """The in-batch retrieval rank definition from ``features.clip_features``,
    applied to one new item against a fixed reference set of distractor
    captions instead of the rest of its training batch: the fraction of
    distractors whose embedding the image prefers over the true headline's.
    0 means the image matches this headline best of everything it was
    compared against; 1 means every distractor beat it."""
    clip_sim = float(image_emb @ text_emb)
    dsims = distractor_text_embs @ image_emb
    if len(dsims) == 0:
        return 0.0
    return float((dsims > clip_sim).sum()) / len(dsims)


def encode_llm_one(headline: str, device: str):
    from src.features import LLM_PROMPT
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.LLM_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        C.LLM_MODEL, dtype=torch.float32).to(device).eval()
    id_fake = tok(" fake", add_special_tokens=False)["input_ids"][0]
    id_real = tok(" genuine", add_special_tokens=False)["input_ids"][0]

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": LLM_PROMPT.format(title=headline)}],
        tokenize=False, add_generation_prompt=True)
    enc = tok([prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        body = mdl.get_decoder() if hasattr(mdl, "get_decoder") else mdl.model
        h = body(**enc).last_hidden_state[:, -1, :]
        logits = mdl.lm_head(h)
        two = torch.stack([logits[:, id_real], logits[:, id_fake]], dim=1)
        p_fake = torch.softmax(two, dim=1)[0, 1].item()
    del mdl
    return h[0].float().cpu().numpy(), p_fake


# --------------------------------------------------------------------------- #
def _default_variant() -> str:
    return "cgf_llm_adv" if (C.PROCESSED / "feat_llm.npz").exists() else "cgf"


def train_and_cache(variant: str, device: str) -> Path:
    """Train the requested CGF variant on protocol A and cache everything
    a later call needs to score a new example the same way."""
    print(f"[predict] no cached checkpoint for '{variant}'; training one now "
          f"(protocol=random, this happens once)")
    df, feats = load_data("corpus.parquet", "feat")
    y = 1 - df["2_way_label"].to_numpy()          # positive class = fake

    idxs = {s: np.where(df["split_random"].to_numpy() == s)[0]
            for s in ("train", "val", "test")}
    X = build_inputs(feats, df, idxs["train"])
    domains = pd.factorize(df["subreddit"])[0]

    spec = variant_spec(variant, X, df)
    if spec is None:
        raise SystemExit(
            f"'{variant}' needs LLM features (data/processed/feat_llm.npz). "
            f"Run `python -m src.features --stages llm` first, or choose a "
            f"variant from {CGF_VARIANTS[:6]}.")

    r = train_one(X, y, idxs, spec, C.SEEDS[0], device, domains=domains,
                  return_model=True)
    print(f"[predict] trained {variant}: val macro-F1={r['val_f1']:.3f}, "
          f"{r['epochs']} epochs, {r['train_seconds']:.0f}s")

    tr = idxs["train"]
    meta_mu = feats["meta"][tr].mean(0)
    meta_sd = feats["meta"][tr].std(0) + 1e-6
    cons_raw = np.stack([feats["clip_sim"], feats["clip_rank"]], axis=1)
    cons_mu = cons_raw[tr].mean(0)
    cons_sd = cons_raw[tr].std(0) + 1e-6
    meta_median = np.median(feats["meta"][tr], axis=0)

    rng = np.random.default_rng(DISTRACTOR_SEED)
    d_idx = rng.choice(tr, size=min(N_DISTRACTORS, len(tr)), replace=False)
    distractor_text = feats["clip_text"][d_idx]     # already L2-normalised

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{variant}.pt"
    torch.save({
        "variant": variant,
        "kwargs": spec["kwargs"],
        "state_dict": r["model"].state_dict(),
        "meta_mu": meta_mu, "meta_sd": meta_sd,
        "cons_mu": cons_mu, "cons_sd": cons_sd,
        "meta_median": meta_median,
        "distractor_text": distractor_text,
        "val_f1": r["val_f1"],
    }, ckpt_path)
    print(f"[predict] cached checkpoint -> {ckpt_path}")
    return ckpt_path


def load_checkpoint(variant: str, device: str) -> dict:
    path = CKPT_DIR / f"{variant}.pt"
    if not path.exists():
        path = train_and_cache(variant, device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model({"kind": "cgf", "kwargs": ckpt["kwargs"]}).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ckpt["model"] = model
    return ckpt


# --------------------------------------------------------------------------- #
def score(image_path: str, headline: str, variant: str | None = None,
          device: str | None = None, score_=None, comments=None,
          upvote_ratio=None, has_author=None, is_selfpost=None) -> dict:
    device = device or C.get_device()
    variant = variant or _default_variant()
    if variant not in CGF_VARIANTS:
        raise SystemExit(f"--variant must be one of {CGF_VARIANTS}")

    ckpt = load_checkpoint(variant, device)
    model = ckpt["model"]

    image_emb, text_emb_clip = encode_clip_one(image_path, headline, device)
    clip_sim = float(image_emb @ text_emb_clip)
    clip_rank = clip_retrieval_rank(image_emb, text_emb_clip,
                                    ckpt["distractor_text"])

    text_emb = encode_text_one(headline, device)

    now = datetime.now(timezone.utc)
    meta_raw = ckpt["meta_median"].copy()
    order = {name: i for i, name in enumerate(META_FEATURES)}
    meta_raw[order["title_len_chars"]] = len(headline)
    meta_raw[order["title_len_words"]] = len(headline.split())
    meta_raw[order["hour"]] = now.hour
    meta_raw[order["dayofweek"]] = now.weekday()
    if score_ is not None:
        meta_raw[order["log_score"]] = np.log1p(max(0, score_))
    if comments is not None:
        meta_raw[order["log_comments"]] = np.log1p(max(0, comments))
    if upvote_ratio is not None:
        meta_raw[order["upvote_ratio"]] = upvote_ratio
    if has_author is not None:
        meta_raw[order["has_author"]] = float(has_author)
    if is_selfpost is not None:
        meta_raw[order["is_selfpost"]] = float(is_selfpost)

    meta_z = (meta_raw - ckpt["meta_mu"]) / ckpt["meta_sd"]
    cons_z = (np.array([clip_sim, clip_rank]) - ckpt["cons_mu"]) / ckpt["cons_sd"]

    batch = {
        "text": torch.as_tensor(text_emb, dtype=torch.float32,
                                device=device).unsqueeze(0),
        "image": torch.as_tensor(image_emb, dtype=torch.float32,
                                 device=device).unsqueeze(0),
        "meta": torch.as_tensor(meta_z, dtype=torch.float32,
                                device=device).unsqueeze(0),
        "cons": torch.as_tensor(cons_z, dtype=torch.float32,
                                device=device).unsqueeze(0),
    }
    if variant in LLM_VARIANTS:
        llm_hidden, llm_p_fake = encode_llm_one(headline, device)
        batch["llm"] = torch.as_tensor(
            llm_hidden, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        llm_p_fake = None

    with torch.no_grad():
        logits = model(batch)
        p_fake = torch.softmax(logits, 1)[0, 1].item()
        gate = (model.last_gate.mean().item()
                if getattr(model, "last_gate", None) is not None else None)

    return {"variant": variant, "p_fake": p_fake, "label": p_fake >= 0.5,
            "clip_sim": clip_sim, "clip_rank": clip_rank, "gate": gate,
            "llm_p_fake": llm_p_fake, "checkpoint_val_f1": ckpt["val_f1"]}


def explain(r: dict) -> str:
    lines = []
    verdict = "LIKELY FAKE" if r["label"] else "LIKELY GENUINE"
    lines.append(f"{verdict}  --  P(fake) = {r['p_fake']:.1%}  "
                 f"(model: {r['variant']}, its held-out val macro-F1 was "
                 f"{r['checkpoint_val_f1']:.3f})")
    lines.append(f"image/headline CLIP agreement: cosine={r['clip_sim']:.3f}, "
                 f"retrieval rank={r['clip_rank']:.2f} "
                 f"(0 = image matches this headline best of {N_DISTRACTORS} "
                 f"references seen, 1 = worst)")
    if r["gate"] is not None:
        g = r["gate"]
        note = ("the model leaned on the image" if g > 0.6 else
                "the model discounted the image" if g < 0.4 else
                "the model gave the image partial weight")
        lines.append(f"learned visual gate activation: {g:.3f} -- {note}")
    if r["llm_p_fake"] is not None:
        lines.append(f"LLM zero-shot judge on the headline alone: "
                     f"P(fake)={r['llm_p_fake']:.1%}")
    lines.append("")
    lines.append("This score reflects the artefact's definition of "
                 "'fake': a post whose headline claim is not corroborated "
                 "by its image, as Fakeddit's annotators judged it -- not a "
                 "check for AI-generated or digitally manipulated pixels.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="path to the image")
    ap.add_argument("--headline", required=True, help="the claimed headline")
    ap.add_argument("--variant", default=None, choices=CGF_VARIANTS,
                    help="which CGF variant to use (default: the report's "
                         "proposed system if LLM features exist, else 'cgf')")
    ap.add_argument("--device", default=None)
    ap.add_argument("--score", type=float, default=None,
                    help="real post score, if you know it")
    ap.add_argument("--comments", type=float, default=None,
                    help="real comment count, if you know it")
    ap.add_argument("--upvote-ratio", type=float, default=None)
    ap.add_argument("--has-author", type=int, choices=[0, 1], default=None)
    ap.add_argument("--is-selfpost", type=int, choices=[0, 1], default=None)
    args = ap.parse_args()

    r = score(args.image, args.headline, variant=args.variant,
              device=args.device, score_=args.score, comments=args.comments,
              upvote_ratio=args.upvote_ratio, has_author=args.has_author,
              is_selfpost=args.is_selfpost)
    print()
    print(explain(r))


if __name__ == "__main__":
    main()
