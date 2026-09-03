"""
Stage 9 -- export every quantity the report figures plot into plain CSV.

MATLAB cannot read the pipeline's native caches (``.npy``, ``.npz`` and
``.parquet``), so this stage flattens exactly the arrays and tables consumed by
``src/analyse.py`` into ``matlab/data/*.csv``. The MATLAB scripts in
``matlab/`` then redraw the report figures from these files alone, which keeps
the two implementations reading identical numbers rather than re-deriving them
independently.

    python -m src.export_matlab
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                    # noqa: E402
from src.analyse import PRETTY, ORDER, mean_pred               # noqa: E402

PRED = C.RESULTS / "preds"
OUT = Path(__file__).resolve().parents[1] / "matlab" / "data"
PROTOCOLS = ["random", "source", "temporal"]
ABLATION = ["cgf", "cgf_no_gate", "cgf_no_cons", "cgf_no_inter", "cgf_no_meta"]
CAL_MODELS = ["cgf", "text", "llm_zeroshot"]
CATEGORY_NAMES = {0: "true", 2: "misleading content",
                  3: "imposter content", 5: "false connection"}
COST_MODELS = ["subreddit_probe", "tfidf_lr", "text", "image", "concat",
               "cgf", "llm_emb", "llm_zeroshot", "cgf_llm"]


def write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"[export] {name:38s} {len(df):6d} rows")


# --------------------------------------------------------------------------- #
def export_fig1(corpus: pd.DataFrame) -> None:
    """Subreddit composition: posts per community, split by binary label."""
    t = pd.crosstab(corpus["subreddit"], corpus["2_way_label"])
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index]
    write(pd.DataFrame({"subreddit": t.index,
                        "n_true": t.get(1, 0).to_numpy(),
                        "n_fake": t.get(0, 0).to_numpy()}),
          "fig1_subreddit_counts.csv")


def export_fig2(ci: pd.DataFrame) -> None:
    """Seed-averaged macro-F1 with bootstrap bounds for every model."""
    rows = []
    for rank, model in enumerate(ORDER):
        for proto in PROTOCOLS:
            s = ci[(ci.protocol == proto) & (ci.model == model)]
            if s.empty:
                continue
            r = s.iloc[0]
            rows.append({"model": model, "label": PRETTY.get(model, model),
                         "order": rank, "protocol": proto,
                         "macro_f1": r.macro_f1, "lo": r.macro_f1_lo,
                         "hi": r.macro_f1_hi})
    write(pd.DataFrame(rows), "fig2_main_results.csv")


def export_fig3(corpus: pd.DataFrame, base: np.lib.npyio.NpzFile) -> None:
    """CLIP image/headline agreement, per item and averaged per community."""
    lookup = {v: i for i, v in enumerate(base["ids"])}
    pos = [lookup[i] for i in corpus["id"] if i in lookup]
    keep = corpus[corpus["id"].isin(lookup)].reset_index(drop=True)
    sim = base["clip_sim"][pos]
    write(pd.DataFrame({"clip_sim": sim,
                        "is_fake": 1 - keep["2_way_label"].to_numpy()}),
          "fig3_item_consistency.csv")

    g = (pd.DataFrame({"subreddit": keep["subreddit"], "clip_sim": sim,
                       "label": keep["2_way_label"]})
         .groupby("subreddit")
         .agg(mean_sim=("clip_sim", "mean"), label=("label", "first"))
         .sort_values("mean_sim").reset_index())
    g["is_fake"] = 1 - g["label"]
    write(g[["subreddit", "mean_sim", "is_fake"]],
          "fig3_subreddit_consistency.csv")


def export_fig4(runs: pd.DataFrame, protocol: str) -> None:
    """Reliability-diagram points: bin-mean prediction against observed rate."""
    y = np.load(PRED / f"{protocol}__y_test.npy")
    bins = np.linspace(0, 1, 11)
    rows = []
    for model in CAL_MODELS:
        if runs[(runs.protocol == protocol)
                & (runs.model == model)].empty:
            continue
        p = mean_pred(runs, protocol, model)
        idx = np.digitize(p, bins[1:-1], right=True)
        for b in range(10):
            k = idx == b
            if k.sum() > 20:
                rows.append({"model": model, "label": PRETTY.get(model, model),
                             "bin": b, "mean_pred": float(p[k].mean()),
                             "observed": float(y[k].mean()),
                             "n": int(k.sum())})
    write(pd.DataFrame(rows), f"fig4_calibration_{protocol}.csv")


def export_fig5(ci: pd.DataFrame) -> None:
    """Ablation deltas, each component removed from the full CGF model."""
    rows = []
    for proto in PROTOCOLS:
        sub = ci[ci.protocol == proto].set_index("model")
        if "cgf" not in sub.index:
            continue
        base = sub.loc["cgf", "macro_f1"]
        for pos, model in enumerate(ABLATION):
            if model in sub.index:
                rows.append({"variant": model, "order": pos,
                             "protocol": proto,
                             "delta_macro_f1": sub.loc[model, "macro_f1"]
                             - base})
    write(pd.DataFrame(rows), "fig5_ablation.csv")


def export_fig6(runs: pd.DataFrame, protocol: str, model: str) -> None:
    """Per-category accuracy of one detector on the held-out test set."""
    meta = pd.read_parquet(PRED / f"{protocol}__test_meta.parquet")
    y = np.load(PRED / f"{protocol}__y_test.npy")
    p = mean_pred(runs, protocol, model)
    ok = ((p >= 0.5).astype(int) == y)
    rows = (pd.DataFrame({"category": meta["6_way_label"].map(CATEGORY_NAMES)
                          .to_numpy(), "correct": ok})
            .groupby("category")["correct"].agg(accuracy="mean", n="size")
            .sort_values("accuracy").reset_index())
    rows["model"] = PRETTY.get(model, model)
    write(rows, f"fig6_error_by_category_{protocol}.csv")


def export_fig7(runs: pd.DataFrame, protocol: str,
                base: np.lib.npyio.NpzFile) -> float | None:
    """Learned visual-gate activation against CLIP consistency, per item."""
    seeds = runs[(runs.protocol == protocol)
                 & (runs.model == "cgf")]["seed"].unique()
    paths = [PRED / f"{protocol}__cgf__{int(s)}__gate.npy" for s in seeds]
    gs = [np.load(p) for p in paths if p.exists()]
    if not gs:
        return None
    g = np.mean(gs, axis=0)
    meta = pd.read_parquet(PRED / f"{protocol}__test_meta.parquet")
    y = np.load(PRED / f"{protocol}__y_test.npy")
    lookup = {v: i for i, v in enumerate(base["ids"])}
    sim = base["clip_sim"][[lookup[i] for i in meta["id"]]]
    write(pd.DataFrame({"clip_sim": sim, "gate": g, "is_fake": y}),
          f"fig7_gate_{protocol}.csv")
    return float(np.corrcoef(g, sim)[0, 1])


def export_fig8(ci: pd.DataFrame) -> None:
    """Measured inference cost against source-disjoint macro-F1."""
    path = C.RESULTS / "inference_cost.json"
    if not path.exists():
        return
    cost = json.loads(path.read_text())
    sub = ci[ci.protocol == "source"].set_index("model")
    rows = [{"model": m, "label": PRETTY.get(m, m), "cost_ms": cost[m],
             "macro_f1": sub.loc[m, "macro_f1"]}
            for m in COST_MODELS if m in cost and m in sub.index]
    write(pd.DataFrame(rows), "fig8_cost.csv")


# --------------------------------------------------------------------------- #
def main() -> None:
    corpus = pd.read_parquet(C.PROCESSED / "corpus.parquet")
    runs = pd.read_csv(C.RESULTS / "runs.csv")
    ci = pd.read_csv(C.RESULTS / "confidence_intervals.csv")
    base = np.load(C.PROCESSED / "feat_base.npz", allow_pickle=True)

    export_fig1(corpus)
    export_fig2(ci)
    export_fig3(corpus, base)
    export_fig5(ci)
    export_fig8(ci)

    gate_r = {}
    for proto in PROTOCOLS:
        if not (PRED / f"{proto}__y_test.npy").exists():
            continue
        export_fig4(runs, proto)
        export_fig6(runs, proto, "cgf")
        r = export_fig7(runs, proto, base)
        if r is not None:
            gate_r[proto] = r

    (OUT / "gate_correlation.json").write_text(json.dumps(gate_r, indent=2))
    print(f"[export] gate_correlation.json               {len(gate_r)} entries")
    print(f"[export] all files written to {OUT}")


if __name__ == "__main__":
    main()
