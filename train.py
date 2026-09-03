"""
Stage 4 -- experiment runner.

Trains every model variant under every evaluation protocol for every seed and
writes one tidy row per (protocol, model, seed) to ``results/runs.csv``, plus
the raw test-set probabilities to ``results/preds/`` so that significance
tests, calibration curves and error analysis can be recomputed without
retraining.

    python -m src.train                      # everything
    python -m src.train --protocols random   # one protocol
    python -m src.train --models cgf,text    # subset of models

Every run is fully seeded (Python, NumPy and torch), the feature scaler is fit
on training rows only, and model selection uses the validation split alone --
the test split is touched exactly once per run, at the end.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                    # noqa: E402
from src.metrics import compute_metrics                        # noqa: E402
from src.models import (ConsistencyGatedFusion, LateFusion,    # noqa: E402
                        MLPHead, count_parameters)

PROTOCOLS = {"random": "split_random",
             "source": "split_source",
             "temporal": "split_temporal"}


# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_data(corpus: str, prefix: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(C.PROCESSED / corpus)
    base = np.load(C.PROCESSED / f"{prefix}_base.npz", allow_pickle=True)
    feats = {k: base[k] for k in base.files}

    llm_path = C.PROCESSED / f"{prefix}_llm.npz"
    if llm_path.exists():
        llm = np.load(llm_path, allow_pickle=True)
        assert (llm["ids"] == base["ids"]).all(), "feature files are misaligned"
        feats["llm_hidden"] = llm["llm_hidden"]
        feats["llm_p_fake"] = llm["llm_p_fake"]
    else:
        print("[warn] no LLM features found; LLM variants will be skipped")
    assert (df["id"].to_numpy() == base["ids"]).all(), "corpus/feature mismatch"
    return df, feats


# --------------------------------------------------------------------------- #
# Feature assembly. Every entry returns the dict of tensors a model consumes.
# --------------------------------------------------------------------------- #
def _z(train_idx, x):
    """Standardise using training-split statistics only (no leakage)."""
    mu = x[train_idx].mean(0, keepdims=True)
    sd = x[train_idx].std(0, keepdims=True) + 1e-6
    return (x - mu) / sd


def build_inputs(feats: dict, df: pd.DataFrame, train_idx: np.ndarray) -> dict:
    text = feats["text_emb"]
    image = feats["clip_image"]
    clip_txt = feats["clip_text"]
    meta = _z(train_idx, feats["meta"])
    cons = np.stack([feats["clip_sim"], feats["clip_rank"]], axis=1)
    cons = _z(train_idx, cons)
    out = {
        "text": text, "image": image, "clip_text": clip_txt,
        "meta": meta, "cons": cons,
    }
    if "llm_hidden" in feats:
        out["llm"] = feats["llm_hidden"]
        out["llm_p_fake"] = feats["llm_p_fake"]
    return out


def variant_spec(name: str, X: dict, df: pd.DataFrame) -> dict | None:
    """Map a model name to (constructor kwargs, list of input keys)."""
    d_text, d_img = X["text"].shape[1], X["image"].shape[1]
    d_meta = X["meta"].shape[1]
    d_llm = X["llm"].shape[1] if "llm" in X else 0
    has_llm = d_llm > 0

    cgf = dict(d_text=d_text, d_img=d_img, d_meta=d_meta, hidden=C.HIDDEN,
               dropout=C.DROPOUT)
    n_dom = int(df["subreddit"].nunique())

    table = {
        # ---- single-stream and naive-fusion baselines --------------------- #
        "text":       ("mlp", {"d_in": d_text}, ["text"]),
        "image":      ("mlp", {"d_in": d_img}, ["image"]),
        "meta":       ("mlp", {"d_in": d_meta}, ["meta"]),
        "cons":       ("mlp", {"d_in": X["cons"].shape[1]}, ["cons"]),
        "concat":     ("mlp", {"d_in": d_text + d_img}, ["text", "image"]),
        "concat_meta": ("mlp", {"d_in": d_text + d_img + d_meta},
                        ["text", "image", "meta"]),
        "late":       ("late", {"d_text": d_text, "d_img": d_img}, None),
        # ---- proposed model and its ablations ----------------------------- #
        "cgf":        ("cgf", {**cgf, "use_llm": False}, None),
        "cgf_no_gate": ("cgf", {**cgf, "use_gate": False, "use_llm": False},
                        None),
        "cgf_no_cons": ("cgf", {**cgf, "use_consistency": False,
                                "use_llm": False}, None),
        "cgf_no_inter": ("cgf", {**cgf, "use_interaction": False,
                                 "use_llm": False}, None),
        "cgf_no_meta": ("cgf", {**cgf, "use_meta": False, "use_llm": False},
                        None),
        # ---- domain-adversarial variants (the remedy proposed in Ch. 6) --- #
        "concat_adv": ("mlp", {"d_in": d_text + d_img, "n_domains": n_dom},
                       ["text", "image"]),
        "cgf_adv":    ("cgf", {**cgf, "use_llm": False, "n_domains": n_dom},
                       None),
    }
    if has_llm:
        table["llm_emb"] = ("mlp", {"d_in": d_llm}, ["llm"])
        table["cgf_llm"] = ("cgf", {**cgf, "d_llm": d_llm, "use_llm": True},
                            None)
        table["cgf_llm_adv"] = ("cgf", {**cgf, "d_llm": d_llm, "use_llm": True,
                                        "n_domains": n_dom}, None)
    if name not in table:
        return None
    kind, kwargs, keys = table[name]
    return {"kind": kind, "kwargs": kwargs, "keys": keys}


def make_batch(X: dict, spec: dict, idx: np.ndarray, device: str) -> dict:
    t = lambda a: torch.as_tensor(a[idx], dtype=torch.float32, device=device)
    if spec["keys"] is not None:
        return {"x": torch.cat([t(X[k]) for k in spec["keys"]], dim=1)}
    b = {"text": t(X["text"]), "image": t(X["image"]),
         "meta": t(X["meta"]), "cons": t(X["cons"])}
    if "llm" in X:
        b["llm"] = t(X["llm"])
    return b


def build_model(spec: dict) -> nn.Module:
    if spec["kind"] == "mlp":
        return MLPHead(hidden=C.HIDDEN, dropout=C.DROPOUT, **spec["kwargs"])
    if spec["kind"] == "late":
        return LateFusion(hidden=C.HIDDEN, dropout=C.DROPOUT, **spec["kwargs"])
    return ConsistencyGatedFusion(**spec["kwargs"])


# --------------------------------------------------------------------------- #
def train_one(X: dict, y: np.ndarray, idxs: dict, spec: dict, seed: int,
              device: str, epochs: int = C.EPOCHS,
              domains: np.ndarray | None = None,
              return_model: bool = False) -> dict:
    """Train with early stopping on validation macro-F1; return test outputs.

    If the model carries a ``DomainAdversary`` and ``domains`` is supplied, an
    auxiliary source-community classifier is trained through a gradient
    reversal layer. Its weight follows the standard DANN ramp
    ``lambda = 2/(1+exp(-10p)) - 1`` over training progress ``p``, which avoids
    destabilising the representation while it is still random.
    """
    set_seed(seed)
    model = build_model(spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR,
                            weight_decay=C.WEIGHT_DECAY)
    use_adv = getattr(model, "adversary", None) is not None and domains is not None
    dt = (torch.as_tensor(domains, dtype=torch.long, device=device)
          if use_adv else None)
    adv_lossf = nn.CrossEntropyLoss()

    # class weights guard against the mild imbalance in the source-disjoint
    # protocol, where held-out subreddits are not equally sized.
    tr = idxs["train"]
    cw = torch.as_tensor(
        len(tr) / (2 * np.bincount(y[tr], minlength=2).clip(min=1)),
        dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=cw)

    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    best, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for a in range(0, len(perm), C.BATCH_SIZE):
            bidx = perm[a:a + C.BATCH_SIZE]
            opt.zero_grad()
            b = make_batch(X, spec, bidx, device)
            if use_adv:
                h = model.encode(b)
                out = model.out(h)
                p = (ep + a / max(1, len(perm))) / max(1, epochs)
                lam = C.ADV_LAMBDA * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
                loss = (lossf(out, yt[bidx])
                        + adv_lossf(model.adversary(h, lam), dt[bidx]))
            else:
                out = model(b)
                loss = lossf(out, yt[bidx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = torch.softmax(
                model(make_batch(X, spec, idxs["val"], device)), 1)[:, 1]
        f1 = compute_metrics(y[idxs["val"]], pv.cpu().numpy())["macro_f1"]
        if f1 > best + 1e-4:
            best, bad = f1, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(make_batch(X, spec, idxs["test"], device))
        pt = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        gate = None
        if getattr(model, "last_gate", None) is not None:
            gate = model.last_gate.mean(dim=1).cpu().numpy()
    out = {"p_test": pt, "val_f1": best, "epochs": ep + 1,
           "params": count_parameters(model), "gate": gate,
           "train_seconds": time.time() - t0}
    if return_model:
        out["model"] = model
    return out


# --------------------------------------------------------------------------- #
def sklearn_baselines(df: pd.DataFrame, idxs: dict, y: np.ndarray,
                      X: dict) -> dict:
    """Non-neural reference points, including the shortcut probe."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    out = {}
    tr, te = idxs["train"], idxs["test"]

    # (a) classical TF-IDF + logistic regression on the headline
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000,
                          sublinear_tf=True)
    Xtr = vec.fit_transform(df["clean_title"].iloc[tr])
    Xte = vec.transform(df["clean_title"].iloc[te])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
    out["tfidf_lr"] = lr.predict_proba(Xte)[:, 1]

    # (b) SHORTCUT PROBE: subreddit identity alone, no content whatsoever.
    #     Near-perfect accuracy here is evidence that the benchmark's labels
    #     are a deterministic function of the source community, which is the
    #     central methodological finding of Chapter 4.
    enc = OneHotEncoder(handle_unknown="ignore")
    Str = enc.fit_transform(df[["subreddit"]].iloc[tr])
    Ste = enc.transform(df[["subreddit"]].iloc[te])
    lr2 = LogisticRegression(max_iter=1000).fit(Str, y[tr])
    out["subreddit_probe"] = lr2.predict_proba(Ste)[:, 1]

    # (c) zero-shot LLM judge -- no training data used at all
    if "llm_p_fake" in X:
        out["llm_zeroshot"] = X["llm_p_fake"][te]

    # (d) majority class
    maj = float(np.bincount(y[tr], minlength=2).argmax())
    out["majority"] = np.full(len(te), maj, dtype=np.float32)
    return out


# --------------------------------------------------------------------------- #
NEURAL = ["text", "image", "meta", "cons", "concat", "concat_meta", "late",
          "cgf", "cgf_no_gate", "cgf_no_cons", "cgf_no_inter", "cgf_no_meta",
          "concat_adv", "cgf_adv", "llm_emb", "cgf_llm", "cgf_llm_adv"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus.parquet")
    ap.add_argument("--prefix", default="feat")
    ap.add_argument("--protocols", default="random,source,temporal")
    ap.add_argument("--models", default=",".join(NEURAL))
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs.csv")
    args = ap.parse_args()

    device = args.device or C.get_device()
    df, feats = load_data(args.corpus, args.prefix)
    y = df["2_way_label"].to_numpy()
    # Fakeddit encodes 1 = true. We invert so that the positive class is
    # 'fake', which is the class a moderation system must not miss.
    y = 1 - y

    preds_dir = C.RESULTS / "preds"
    preds_dir.mkdir(exist_ok=True)
    rows = []

    for pname in args.protocols.split(","):
        col = PROTOCOLS[pname.strip()]
        idxs = {s: np.where(df[col].to_numpy() == s)[0]
                for s in ("train", "val", "test")}
        print(f"\n=== protocol={pname}  "
              f"train={len(idxs['train'])} val={len(idxs['val'])} "
              f"test={len(idxs['test'])} ===")
        X = build_inputs(feats, df, idxs["train"])
        domains = pd.factorize(df["subreddit"])[0]

        # deterministic, training-free baselines (seed-independent)
        for mname, p in sklearn_baselines(df, idxs, y, X).items():
            m = compute_metrics(y[idxs["test"]], p)
            rows.append({"protocol": pname, "model": mname, "seed": 0,
                         "params": 0, "train_seconds": 0.0, **m})
            np.save(preds_dir / f"{pname}__{mname}__0.npy", p)
            print(f"  {mname:16s} acc={m['accuracy']:.4f} "
                  f"F1={m['macro_f1']:.4f} auroc={m['auroc']:.4f}")

        for mname in args.models.split(","):
            mname = mname.strip()
            spec = variant_spec(mname, X, df)
            if spec is None:
                continue
            for seed in map(int, args.seeds.split(",")):
                r = train_one(X, y, idxs, spec, seed, device,
                              domains=domains)
                m = compute_metrics(y[idxs["test"]], r["p_test"])
                rows.append({"protocol": pname, "model": mname, "seed": seed,
                             "params": r["params"], "epochs": r["epochs"],
                             "val_f1": r["val_f1"],
                             "train_seconds": r["train_seconds"], **m})
                np.save(preds_dir / f"{pname}__{mname}__{seed}.npy", r["p_test"])
                if r["gate"] is not None:
                    np.save(preds_dir / f"{pname}__{mname}__{seed}__gate.npy",
                            r["gate"])
            sub = [x for x in rows if x["model"] == mname
                   and x["protocol"] == pname]
            print(f"  {mname:16s} acc={np.mean([s['accuracy'] for s in sub]):.4f}"
                  f" F1={np.mean([s['macro_f1'] for s in sub]):.4f}"
                  f" ({r['params']:,} params, {r['train_seconds']:.1f}s)")

        np.save(preds_dir / f"{pname}__y_test.npy", y[idxs["test"]])
        df.iloc[idxs["test"]].to_parquet(preds_dir / f"{pname}__test_meta.parquet")

    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / args.out, index=False)
    print(f"\n[done] wrote {C.RESULTS / args.out}  ({len(out)} runs)")
    (C.RESULTS / "run_config.json").write_text(json.dumps(
        {"corpus": args.corpus, "prefix": args.prefix, "device": device,
         "seeds": args.seeds, "hidden": C.HIDDEN, "lr": C.LR,
         "batch_size": C.BATCH_SIZE, "epochs": C.EPOCHS,
         "patience": C.PATIENCE, "dropout": C.DROPOUT}, indent=2))


if __name__ == "__main__":
    main()
