"""
Stage 12 -- produce the reviewer-facing triage queue.

Chapter 2 states that the system ranks content for human review rather than
deciding, that the score attached to each item must be trustworthy, and that a
reviewer must be able to interrogate the reason for a flag. Those three needs
describe an output, not a metric, so this stage produces that output: a ranked
queue in which each row carries the calibrated probability, the empirical
precision of the decile it falls in, and the two per-item signals a reviewer
can check against the post itself.

The queue is built from the cached seed-averaged predictions of a trained
system, so it costs no training and cannot disagree with the numbers reported
in Chapter 6.

    python -m src.triage                          # top 100, proposed system
    python -m src.triage --top-k 500 --model cgf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                    # noqa: E402

PRED = C.RESULTS / "preds"


def seed_averaged(protocol: str, model: str) -> np.ndarray:
    """Mean test probability across the seeds that produced a prediction file."""
    runs = pd.read_csv(C.RESULTS / "runs.csv")
    seeds = runs[(runs.protocol == protocol)
                 & (runs.model == model)]["seed"].unique()
    ps = [np.load(PRED / f"{protocol}__{model}__{int(s)}.npy") for s in seeds
          if (PRED / f"{protocol}__{model}__{int(s)}.npy").exists()]
    if not ps:
        raise SystemExit(f"no cached predictions for {model} on {protocol}; "
                         f"run 'python -m src.train' first")
    return np.mean(ps, axis=0)


def gate_activation(protocol: str, model: str, n: int) -> np.ndarray | None:
    """Mean gate activation per item, where the model exposes one."""
    runs = pd.read_csv(C.RESULTS / "runs.csv")
    seeds = runs[(runs.protocol == protocol)
                 & (runs.model == model)]["seed"].unique()
    gs = [np.load(PRED / f"{protocol}__{model}__{int(s)}__gate.npy")
          for s in seeds
          if (PRED / f"{protocol}__{model}__{int(s)}__gate.npy").exists()]
    return np.mean(gs, axis=0) if gs else None


def decile_precision(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Observed proportion of false posts within each decile of the score.

    This is what makes the queue usable rather than merely ordered: a reviewer
    reads "of items scored in this band, this proportion were in fact false",
    which is an empirical frequency rather than a model output.
    """
    edges = np.quantile(p, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    band = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    rate = np.array([y[band == b].mean() if (band == b).any() else np.nan
                     for b in range(10)])
    return rate[band]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="source",
                    choices=["random", "source", "temporal"])
    ap.add_argument("--model", default="cgf_llm_adv")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out", default="triage_queue.csv")
    a = ap.parse_args()

    meta = pd.read_parquet(PRED / f"{a.protocol}__test_meta.parquet")
    y = np.load(PRED / f"{a.protocol}__y_test.npy")
    p = seed_averaged(a.protocol, a.model)
    gate = gate_activation(a.protocol, a.model, len(p))

    base = np.load(C.PROCESSED / "feat_base.npz", allow_pickle=True)
    lookup = {v: i for i, v in enumerate(base["ids"])}
    sim = base["clip_sim"][[lookup[i] for i in meta["id"]]]

    q = pd.DataFrame({
        "post_id": meta["id"].to_numpy(),
        "headline": meta["clean_title"].to_numpy()
        if "clean_title" in meta.columns else meta["title"].to_numpy(),
        "p_false": p,
        "band_precision": decile_precision(y, p),
        "clip_consistency": sim,
        "source_community": meta["subreddit"].to_numpy(),
        "actual_label": np.where(y == 1, "false", "true"),
    })
    if gate is not None:
        q["visual_gate"] = gate
    q = q.sort_values("p_false", ascending=False).head(a.top_k)
    q.insert(0, "rank", np.arange(1, len(q) + 1))

    path = C.RESULTS / a.out
    q.to_csv(path, index=False)
    hit = (q["actual_label"] == "false").mean()
    print(f"[triage] {a.model} on the {a.protocol} protocol")
    print(f"[triage] top {len(q)} of {len(p)} items; precision {hit:.3f} "
          f"against a prevalence of {y.mean():.3f} "
          f"(lift {hit / y.mean():.2f}x)")
    print(f"[triage] wrote {path}")
    with pd.option_context("display.width", 200,
                           "display.max_colwidth", 46):
        print(q.head(8)[["rank", "p_false", "band_precision",
                         "clip_consistency", "headline"]].round(3)
              .to_string(index=False))


if __name__ == "__main__":
    main()
