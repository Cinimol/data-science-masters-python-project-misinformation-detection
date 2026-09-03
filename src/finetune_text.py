"""
Stage 7 -- end-to-end fine-tuned text baseline.

The main study freezes all encoders, which is what makes 100+ training runs
affordable on a laptop. That choice has to be *justified*, not assumed, so this
script fine-tunes DistilRoBERTa end to end on the same splits and reports the
gap. If a fully fine-tuned unimodal transformer cannot close the distance to
the frozen multimodal model under the source-disjoint protocol, the conclusion
of Chapter 6 -- that the bottleneck is the evaluation protocol and not encoder
capacity -- is much better supported.

Runtime: roughly 35-50 minutes per protocol on a 2-core CPU, about 8 minutes on
Apple-silicon MPS. It is deliberately kept out of the default pipeline.

    python -m src.finetune_text --protocol source --epochs 2
"""
from __future__ import annotations

import argparse
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
from src.train import PROTOCOLS, set_seed                      # noqa: E402


def main() -> None:
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus.parquet")
    ap.add_argument("--protocol", default="source")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-train", type=int, default=None)
    args = ap.parse_args()

    device = args.device or C.get_device()
    set_seed(args.seed)
    df = pd.read_parquet(C.PROCESSED / args.corpus)
    col = PROTOCOLS[args.protocol]
    y = 1 - df["2_way_label"].to_numpy()

    idx = {s: np.where(df[col].to_numpy() == s)[0]
           for s in ("train", "val", "test")}
    if args.max_train:
        rng = np.random.default_rng(args.seed)
        idx["train"] = rng.choice(idx["train"],
                                  min(args.max_train, len(idx["train"])),
                                  replace=False)

    tok = AutoTokenizer.from_pretrained(C.TEXT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        C.TEXT_MODEL, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * int(np.ceil(len(idx["train"]) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    lossf = nn.CrossEntropyLoss()

    texts = df["clean_title"].to_numpy()

    def encode(ix):
        return tok(list(texts[ix]), return_tensors="pt", padding=True,
                   truncation=True, max_length=C.MAX_TEXT_TOKENS).to(device)

    @torch.no_grad()
    def predict(ix):
        model.eval()
        out = []
        for a in range(0, len(ix), 64):
            b = ix[a:a + 64]
            out.append(torch.softmax(model(**encode(b)).logits, 1)[:, 1]
                       .float().cpu().numpy())
        return np.concatenate(out)

    print(f"[ft] protocol={args.protocol} device={device} "
          f"train={len(idx['train'])} steps={total}")
    best, best_state = -1.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(idx["train"])
        for i, a in enumerate(range(0, len(perm), args.batch)):
            b = perm[a:a + args.batch]
            opt.zero_grad()
            loss = lossf(model(**encode(b)).logits,
                         torch.as_tensor(y[b], device=device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if i % 50 == 0:
                el = time.time() - t0
                done = ep * len(perm) / args.batch + i + 1
                print(f"[ft] ep{ep} step {i} loss={loss.item():.4f} "
                      f"({done * args.batch / el:.1f} samples/s, "
                      f"eta {(total - done) * el / done / 60:.0f} min)",
                      flush=True)
        f1 = compute_metrics(y[idx["val"]], predict(idx["val"]))["macro_f1"]
        print(f"[ft] epoch {ep}: val macro-F1={f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    p = predict(idx["test"])
    m = compute_metrics(y[idx["test"]], p)
    print(f"[ft] TEST {args.protocol}: " +
          " ".join(f"{k}={v:.4f}" for k, v in m.items()
                   if isinstance(v, float)))

    name = "text_finetuned"
    np.save(C.RESULTS / "preds" / f"{args.protocol}__{name}__{args.seed}.npy", p)
    row = {"protocol": args.protocol, "model": name, "seed": args.seed,
           "params": sum(q.numel() for q in model.parameters()),
           "train_seconds": time.time() - t0, "val_f1": best, **m}
    f = C.RESULTS / "runs_finetuned.csv"
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
    print(f"[done] appended to {f}")


if __name__ == "__main__":
    main()
