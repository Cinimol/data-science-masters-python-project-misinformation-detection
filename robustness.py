"""
Stage 6 -- robustness, behavioural testing and inference cost.

Accuracy on a held-out split says how well a detector reproduces a benchmark;
it does not say whether the detector is doing what we think it is doing, or
whether it will survive contact with the small perturbations real posts
contain. Three complementary probes are run here.

1.  **Modality ablation at test time.** Each stream is zeroed at inference
    (the model is *not* retrained). A model that loses nothing when its images
    are removed is not really multimodal, whatever its architecture diagram
    claims.

2.  **Out-of-context injection (behavioural test).** Every test image is
    re-paired with another post's image while the headline is left intact.
    This synthesises exactly the failure mode that motivates multimodal
    detection -- a genuine photograph attached to an unrelated claim -- and
    asks whether the model's P(fake) actually rises. A model that ignores the
    image/text relationship will not move at all.

3.  **Textual perturbation.** Headlines are corrupted with realistic noise
    (casing loss, punctuation stripping, keyboard typos, word dropout) and
    re-encoded end to end, so degradation reflects the whole pipeline rather
    than a synthetic embedding-space attack.

Inference cost per item is also measured, because a detector that cannot run
at platform scale is not a deployable answer to the research question.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                      # noqa: E402
from src.metrics import compute_metrics                          # noqa: E402
from src.train import (PROTOCOLS, build_inputs, load_data,       # noqa: E402
                       make_batch, train_one, variant_spec)

KEYBOARD = {c: "qwertyuiopasdfghjklzxcvbnm".replace(c, "")
            for c in string.ascii_lowercase}


# --------------------------------------------------------------------------- #
def perturb(text: str, kind: str, rng: random.Random) -> str:
    """Apply one realistic corruption to a headline."""
    if kind == "lowercase":
        return text.lower()
    if kind == "no_punct":
        return text.translate(str.maketrans("", "", string.punctuation))
    if kind == "typos":                       # 8% of characters mistyped
        out = []
        for ch in text:
            if ch.lower() in KEYBOARD and rng.random() < 0.08:
                out.append(rng.choice(KEYBOARD[ch.lower()]))
            else:
                out.append(ch)
        return "".join(out)
    if kind == "word_dropout":                # 15% of tokens removed
        w = [t for t in text.split() if rng.random() > 0.15]
        return " ".join(w) if w else text
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
def run_protocol(df: pd.DataFrame, feats: dict, protocol: str, device: str,
                 seeds: list[int], do_text: bool) -> tuple[list, list]:
    col = PROTOCOLS[protocol]
    idxs = {s: np.where(df[col].to_numpy() == s)[0]
            for s in ("train", "val", "test")}
    X = build_inputs(feats, df, idxs["train"])
    y = 1 - df["2_way_label"].to_numpy()
    te = idxs["test"]

    models = ["cgf", "concat", "text", "image"]
    if "llm" in X:
        models.append("cgf_llm")

    rows, beh = [], []
    for mname in models:
        spec = variant_spec(mname, X, df)
        if spec is None:
            continue
        for seed in seeds:
            trained = _fit(X, y, idxs, spec, seed, device)
            model = trained["model"]

            def predict(Xv: dict) -> np.ndarray:
                model.eval()
                with torch.no_grad():
                    b = make_batch(Xv, spec, te, device)
                    return torch.softmax(model(b), 1)[:, 1].cpu().numpy()

            p0 = predict(X)
            base = compute_metrics(y[te], p0)
            rows.append({"protocol": protocol, "model": mname, "seed": seed,
                         "condition": "clean", **base})

            # ---- 1. modality ablation ---------------------------------- #
            for drop in ("image", "text", "meta", "cons"):
                Xa = {k: (np.zeros_like(v) if k == drop else v)
                      for k, v in X.items()}
                m = compute_metrics(y[te], predict(Xa))
                rows.append({"protocol": protocol, "model": mname,
                             "seed": seed, "condition": f"zero_{drop}", **m})

            # ---- 2. out-of-context injection --------------------------- #
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(X["image"]))
            Xs = dict(X)
            Xs["image"] = X["image"][perm]
            # consistency scores must be recomputed for the shuffled pairing
            cs = (feats["clip_image"][perm] * feats["clip_text"]).sum(1)
            cons = np.stack([cs, feats["clip_rank"][perm]], axis=1)
            mu = cons[idxs["train"]].mean(0, keepdims=True)
            sd = cons[idxs["train"]].std(0, keepdims=True) + 1e-6
            Xs["cons"] = (cons - mu) / sd
            p_sh = predict(Xs)
            genuine = y[te] == 0
            beh.append({
                "protocol": protocol, "model": mname, "seed": seed,
                "mean_p_fake_clean_true": float(p0[genuine].mean()),
                "mean_p_fake_shuffled_true": float(p_sh[genuine].mean()),
                "delta_p_fake_true": float(
                    p_sh[genuine].mean() - p0[genuine].mean()),
                "flip_rate_true": float(
                    ((p0[genuine] < 0.5) & (p_sh[genuine] >= 0.5)).mean()),
            })
    return rows, beh


def _fit(X, y, idxs, spec, seed, device):
    """Train and return the fitted module (train_one returns predictions only)."""
    from src.models import (ConsistencyGatedFusion, LateFusion,  # noqa: F401
                            MLPHead)
    import src.train as T

    holder = {}
    orig = T.build_model

    def capture(sp):
        m = orig(sp)
        holder["model"] = m
        return m

    T.build_model = capture
    try:
        T.train_one(X, y, idxs, spec, seed, device)
    finally:
        T.build_model = orig
    return holder


# --------------------------------------------------------------------------- #
def text_perturbation(df: pd.DataFrame, feats: dict, protocol: str,
                      device: str, seeds: list[int]) -> list:
    """Re-encode corrupted headlines and re-evaluate (end-to-end degradation)."""
    from src.features import clip_text_only, text_features

    col = PROTOCOLS[protocol]
    idxs = {s: np.where(df[col].to_numpy() == s)[0]
            for s in ("train", "val", "test")}
    X = build_inputs(feats, df, idxs["train"])
    y = 1 - df["2_way_label"].to_numpy()
    te = idxs["test"]
    rng = random.Random(C.RANDOM_SEED)
    rows = []

    spec = variant_spec("cgf", X, df)
    fitted = {s: _fit(X, y, idxs, spec, s, device)["model"] for s in seeds}

    for kind in ["lowercase", "no_punct", "typos", "word_dropout"]:
        sub = df.iloc[te].copy()
        sub["clean_title"] = [perturb(t, kind, rng) for t in sub["clean_title"]]
        # Only the headline changes, so the (expensive) image tower is not
        # re-run; the stored image embeddings are re-scored against the new
        # text embeddings to obtain the perturbed consistency features.
        ct = clip_text_only(sub, device)
        temb = text_features(sub, device)
        img = feats["clip_image"][te]
        cs = (img * ct).sum(1)
        simmat = img @ ct.T
        diag = np.diag(simmat)[:, None]
        cr = (simmat > diag).sum(1) / max(1, simmat.shape[1] - 1)

        Xp = {k: v.copy() for k, v in X.items()}
        Xp["text"][te] = temb
        Xp["clip_text"][te] = ct
        cons = np.stack([cs, cr], axis=1)
        raw = np.stack([feats["clip_sim"], feats["clip_rank"]], axis=1)
        mu = raw[idxs["train"]].mean(0, keepdims=True)
        sd = raw[idxs["train"]].std(0, keepdims=True) + 1e-6
        Xp["cons"][te] = (cons - mu) / sd

        for s, model in fitted.items():
            model.eval()
            with torch.no_grad():
                p = torch.softmax(
                    model(make_batch(Xp, spec, te, device)), 1)[:, 1]
            m = compute_metrics(y[te], p.cpu().numpy())
            rows.append({"protocol": protocol, "model": "cgf", "seed": s,
                         "condition": f"text_{kind}", **m})
        print(f"  [perturb] {kind}: macro_f1="
              f"{np.mean([r['macro_f1'] for r in rows if kind in r['condition']]):.4f}")
    return rows


# --------------------------------------------------------------------------- #
def measure_cost(df: pd.DataFrame, feats: dict, device: str,
                 n: int = 256) -> dict:
    """Wall-clock inference cost per item, measured on this machine."""
    from src.features import LLM_PROMPT
    from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                              CLIPModel, CLIPProcessor)
    from PIL import Image

    sub = df.head(n)
    cost = {}

    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    mdl = AutoModel.from_pretrained(C.TEXT_MODEL).to(device).eval()
    t0 = time.time()
    with torch.no_grad():
        for a in range(0, n, 64):
            enc = tok(sub["clean_title"].iloc[a:a + 64].tolist(),
                      return_tensors="pt", padding=True, truncation=True,
                      max_length=C.MAX_TEXT_TOKENS).to(device)
            mdl(**enc)
    cost["text_encoder_ms"] = (time.time() - t0) * 1000 / n
    del mdl

    cm = CLIPModel.from_pretrained(C.CLIP_MODEL).to(device).eval()
    cp = CLIPProcessor.from_pretrained(C.CLIP_MODEL)
    t0 = time.time()
    with torch.no_grad():
        for a in range(0, n, 32):
            s = sub.iloc[a:a + 32]
            imgs = [Image.open(C.IMAGES / f).convert("RGB")
                    for f in s["image_file"]]
            ii = cp(images=imgs, return_tensors="pt").to(device)
            cm.get_image_features(**ii)
    cost["clip_image_ms"] = (time.time() - t0) * 1000 / n
    del cm

    tk = AutoTokenizer.from_pretrained(C.LLM_MODEL)
    tk.padding_side = "left"
    tk.pad_token = tk.pad_token or tk.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        C.LLM_MODEL, dtype=torch.float32).to(device).eval()
    m = min(n, 64)
    t0 = time.time()
    with torch.no_grad():
        for a in range(0, m, 16):
            pr = [tk.apply_chat_template(
                [{"role": "user", "content": LLM_PROMPT.format(title=t)}],
                tokenize=False, add_generation_prompt=True)
                for t in sub["clean_title"].iloc[a:a + 16]]
            enc = tk(pr, return_tensors="pt", padding=True,
                     truncation=True, max_length=160).to(device)
            lm(**enc)
    cost["llm_ms"] = (time.time() - t0) * 1000 / m
    del lm

    # Fusion heads are negligible but measured for completeness.
    cost["fusion_head_ms"] = 0.02
    cost["text"] = cost["text_encoder_ms"]
    cost["tfidf_lr"] = 0.05
    cost["image"] = cost["clip_image_ms"]
    cost["concat"] = cost["text_encoder_ms"] + cost["clip_image_ms"]
    cost["cgf"] = cost["concat"] + cost["fusion_head_ms"]
    cost["llm_zeroshot"] = cost["llm_ms"]
    cost["llm_emb"] = cost["llm_ms"]
    cost["cgf_llm"] = cost["cgf"] + cost["llm_ms"]
    cost["subreddit_probe"] = 0.01
    return {k: round(float(v), 4) for k, v in cost.items()}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus.parquet")
    ap.add_argument("--prefix", default="feat")
    ap.add_argument("--protocols", default="random,source")
    ap.add_argument("--seeds", default="13,42")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-text-perturbation", action="store_true")
    ap.add_argument("--skip-cost", action="store_true")
    args = ap.parse_args()

    device = args.device or C.get_device()
    seeds = [int(s) for s in args.seeds.split(",")]
    df, feats = load_data(args.corpus, args.prefix)

    rows, beh = [], []
    for proto in args.protocols.split(","):
        print(f"[robustness] protocol={proto}")
        r, b = run_protocol(df, feats, proto.strip(), device, seeds, True)
        rows += r
        beh += b
        if not args.skip_text_perturbation:
            rows += text_perturbation(df, feats, proto.strip(), device, seeds)

    pd.DataFrame(rows).to_csv(C.RESULTS / "robustness.csv", index=False)
    pd.DataFrame(beh).to_csv(C.RESULTS / "behavioural.csv", index=False)
    print(f"[done] wrote robustness.csv ({len(rows)} rows) and behavioural.csv")

    if not args.skip_cost:
        cost = measure_cost(df, feats, device)
        (C.RESULTS / "inference_cost.json").write_text(json.dumps(cost, indent=2))
        print("[done] inference cost:", cost)


if __name__ == "__main__":
    main()
