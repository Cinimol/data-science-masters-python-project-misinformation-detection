"""
Stage 2 -- representation extraction.

Produces one cached ``.npz`` bundle of frozen representations per corpus:

    clip_image   (n, 512)  CLIP ViT-B/32 visual projection
    clip_text    (n, 512)  CLIP text projection
    clip_sim     (n,)      cosine(clip_image, clip_text)  -- the explicit
                           image/caption consistency signal
    clip_rank    (n,)      normalised retrieval rank of the true caption among
                           in-batch distractors: a harder, batch-calibrated
                           consistency measure that is robust to the fact that
                           raw CLIP cosines are not comparable across topics
    text_emb     (n, 768)  DistilRoBERTa mean-pooled sentence representation
    llm_hidden   (n, 896)  final-layer hidden state of an instruction-tuned LLM
                           at the answer position (LLM-as-encoder)
    llm_p_fake   (n,)      P(fake) from the same forward pass, obtained by
                           comparing the logits of the two answer tokens
                           (LLM-as-judge, zero-shot -- no generation needed)
    meta         (n, 9)    engineered platform metadata

Freezing the encoders is a deliberate design choice: it makes the whole study
reproducible on a laptop CPU in well under an hour, lets us run three seeds
across three evaluation protocols and eight model variants, and isolates the
contribution of the *fusion* mechanism from that of encoder capacity.
``src/finetune_text.py`` provides the end-to-end fine-tuned counterpart so the
cost of that choice can be quantified rather than assumed.

Usage
-----
    python -m src.features --corpus corpus.parquet --out features.npz
    python -m src.features --skip-llm          # ~4x faster, drops LLM streams
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                      # noqa: E402
from src.build_dataset import META_FEATURES      # noqa: E402

LOG = print


# --------------------------------------------------------------------------- #
def _batches(n: int, bs: int):
    for i in range(0, n, bs):
        yield i, min(n, i + bs)


def _tensor(x):
    """Normalise the return type of ``CLIPModel.get_*_features``.

    transformers <5 returns a plain tensor; transformers >=5 wraps it in a
    ``BaseModelOutputWithPooling``. Supporting both keeps the artefact runnable
    across common library versions.
    """
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("pooler_output", "last_hidden_state", "image_embeds",
                 "text_embeds"):
        v = getattr(x, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(x, (tuple, list)):
        return x[0]
    raise TypeError(f"cannot extract tensor from {type(x)}")


@torch.no_grad()
def clip_features(df: pd.DataFrame, device: str, bs: int = 64):
    """Encode images and captions with a frozen CLIP model."""
    from transformers import CLIPModel, CLIPProcessor

    LOG(f"[clip] loading {C.CLIP_MODEL} on {device}")
    model = CLIPModel.from_pretrained(C.CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(C.CLIP_MODEL)

    n = len(df)
    img_out = np.zeros((n, model.config.projection_dim), dtype=np.float32)
    txt_out = np.zeros((n, model.config.projection_dim), dtype=np.float32)
    ranks = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    for a, b in _batches(n, bs):
        sub = df.iloc[a:b]
        imgs = [Image.open(C.IMAGES / f).convert("RGB") for f in sub["image_file"]]
        texts = sub["clean_title"].tolist()

        ii = proc(images=imgs, return_tensors="pt").to(device)
        ti = proc(text=texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=77).to(device)

        ie = _tensor(model.get_image_features(**ii))
        te = _tensor(model.get_text_features(**ti))
        ie = torch.nn.functional.normalize(ie, dim=-1)
        te = torch.nn.functional.normalize(te, dim=-1)

        # In-batch retrieval rank of the *true* caption for each image. A
        # genuine image/caption pair should rank its own caption highly; an
        # out-of-context or manipulated pair should not.
        sim = ie @ te.T                                    # (bs, bs)
        diag = sim.diag().unsqueeze(1)
        r = (sim > diag).sum(dim=1).float() / max(1, sim.shape[1] - 1)

        img_out[a:b] = ie.float().cpu().numpy()
        txt_out[a:b] = te.float().cpu().numpy()
        ranks[a:b] = r.float().cpu().numpy()

        if a and a % (bs * 20) == 0:
            LOG(f"[clip]   {a:,}/{n:,}  ({a / (time.time() - t0):.1f}/s)")

    sims = (img_out * txt_out).sum(1)
    LOG(f"[clip] done in {time.time() - t0:.0f}s  mean cos={sims.mean():.3f}")
    del model
    return img_out, txt_out, sims.astype(np.float32), ranks


@torch.no_grad()
def text_features(df: pd.DataFrame, device: str, bs: int = 128):
    """Mean-pooled sentence embeddings from a frozen DistilRoBERTa."""
    from transformers import AutoModel, AutoTokenizer

    LOG(f"[text] loading {C.TEXT_MODEL} on {device}")
    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    mdl = AutoModel.from_pretrained(C.TEXT_MODEL).to(device).eval()

    n = len(df)
    out = np.zeros((n, mdl.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    for a, b in _batches(n, bs):
        enc = tok(df["clean_title"].iloc[a:b].tolist(), return_tensors="pt",
                  padding=True, truncation=True,
                  max_length=C.MAX_TEXT_TOKENS).to(device)
        h = mdl(**enc).last_hidden_state                    # (bs, L, H)
        m = enc["attention_mask"].unsqueeze(-1).float()
        out[a:b] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
        if a and a % (bs * 20) == 0:
            LOG(f"[text]   {a:,}/{n:,}  ({a / (time.time() - t0):.1f}/s)")
    LOG(f"[text] done in {time.time() - t0:.0f}s")
    del mdl
    return out


LLM_PROMPT = (
    "Fact-check this social media headline.\n"
    'Headline: "{title}"\n'
    "Is it genuine or fake? Answer with one word."
)


@torch.no_grad()
def llm_features(df: pd.DataFrame, device: str, bs: int = 16):
    """Zero-shot LLM judgement *and* LLM hidden state, in one forward pass.

    Rather than sampling free text (slow, and not directly comparable across
    items) we read the logits at the answer position and renormalise over the
    two answer tokens. This is a single forward pass per item, which makes an
    LLM baseline affordable on CPU, and it yields a calibrated probability
    instead of a hard label. The final hidden state at the same position is
    kept as a dense feature: it summarises what the LLM inferred about the
    claim and is used in the ``+LLM`` fusion variants.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOG(f"[llm] loading {C.LLM_MODEL} on {device}")
    tok = AutoTokenizer.from_pretrained(C.LLM_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        C.LLM_MODEL, dtype=torch.float32).to(device).eval()

    id_fake = tok(" fake", add_special_tokens=False)["input_ids"][0]
    id_real = tok(" genuine", add_special_tokens=False)["input_ids"][0]

    n = len(df)
    hid = np.zeros((n, mdl.config.hidden_size), dtype=np.float32)
    pf = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    for a, b in _batches(n, bs):
        prompts = [
            tok.apply_chat_template(
                [{"role": "user",
                  "content": LLM_PROMPT.format(title=t)}],
                tokenize=False, add_generation_prompt=True)
            for t in df["clean_title"].iloc[a:b]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=160).to(device)
        # Run the transformer body only, then project a *single* position
        # through the output head. Materialising logits for every position of
        # a 150k-token vocabulary is by far the dominant cost otherwise; this
        # reduces LLM feature extraction from hours to minutes on CPU.
        body = mdl.get_decoder() if hasattr(mdl, "get_decoder") else mdl.model
        h = body(input_ids=enc["input_ids"],
                 attention_mask=enc["attention_mask"]).last_hidden_state
        h_last = h[:, -1, :]                                 # answer position
        logits = mdl.lm_head(h_last)
        two = torch.stack([logits[:, id_real], logits[:, id_fake]], dim=1)
        pf[a:b] = torch.softmax(two, dim=1)[:, 1].float().cpu().numpy()
        hid[a:b] = h_last.float().cpu().numpy()
        if a and a % (bs * 20) == 0:
            el = time.time() - t0
            LOG(f"[llm]   {a:,}/{n:,}  ({a / el:.1f}/s, "
                f"eta {(n - a) / max(1e-9, a / el) / 60:.0f} min)")
    LOG(f"[llm] done in {time.time() - t0:.0f}s  mean P(fake)={pf.mean():.3f}")
    del mdl
    return hid, pf


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus.parquet")
    ap.add_argument("--prefix", default="feat",
                    help="output files are <prefix>_<stage>.npz")
    ap.add_argument("--stages", default="base,llm",
                    help="comma-separated: base (CLIP+text+meta), llm")
    ap.add_argument("--device", default=None)
    ap.add_argument("--llm-batch", type=int, default=16)
    args = ap.parse_args()

    device = args.device or C.get_device()
    torch.manual_seed(C.RANDOM_SEED)
    df = pd.read_parquet(C.PROCESSED / args.corpus)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    LOG(f"[main] {len(df):,} rows, device={device}, stages={stages}")

    if "base" in stages:
        ci, ct, cs, cr = clip_features(df, device)
        te = text_features(df, device)
        meta = df[META_FEATURES].to_numpy(dtype=np.float32)
        p = C.PROCESSED / f"{args.prefix}_base.npz"
        np.savez_compressed(p, clip_image=ci, clip_text=ct, clip_sim=cs,
                            clip_rank=cr, text_emb=te, meta=meta,
                            ids=df["id"].to_numpy())
        LOG(f"[done] wrote {p}")

    if "llm" in stages:
        lh, lp = llm_features(df, device, bs=args.llm_batch)
        p = C.PROCESSED / f"{args.prefix}_llm.npz"
        np.savez_compressed(p, llm_hidden=lh, llm_p_fake=lp,
                            ids=df["id"].to_numpy())
        LOG(f"[done] wrote {p}")


if __name__ == "__main__":
    main()


@torch.no_grad()
def clip_text_only(df, device: str, bs: int = 128):
    """Encode captions with the frozen CLIP text tower only.

    Used by the perturbation study, where headlines change but images do not,
    so re-running the (much more expensive) vision tower would be wasted work.
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(C.CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(C.CLIP_MODEL)
    n = len(df)
    out = np.zeros((n, model.config.projection_dim), dtype=np.float32)
    for a, b in _batches(n, bs):
        ti = proc(text=df["clean_title"].iloc[a:b].tolist(), return_tensors="pt",
                  padding=True, truncation=True, max_length=77).to(device)
        te = torch.nn.functional.normalize(
            _tensor(model.get_text_features(**ti)), dim=-1)
        out[a:b] = te.float().cpu().numpy()
    del model
    return out
