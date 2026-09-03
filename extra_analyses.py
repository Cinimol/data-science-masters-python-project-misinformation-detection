"""
Stage 10 -- analyses that close the gaps identified in the internal audit.

Six analyses are implemented here rather than in ``src/analyse.py`` because
each answers a specific challenge to the study's conclusions and each is
reported in its own section or appendix of the report:

    1. loco       leave-one-community-out rotation, replacing the single
                  held-out partition of Protocol B with a full rotation and
                  giving a proper uncertainty on the generalisation estimate.
    2. probe      a permutation null and bootstrap interval for the
                  content-free shortcut probe, so the study's headline number
                  carries the same uncertainty quantification it demands of
                  everything else.
    3. community  per-community performance under Protocol B, discharging the
                  commitment to disaggregated reporting made on ethical grounds.
    4. ranking    precision at k on the source-disjoint test set, the metric
                  the stated triage use case actually requires.
    5. prior      decomposition of the protocol gap into the part attributable
                  to class-prior shift and the part attributable to source
                  shift, which bounds the study's central claim honestly.
    6. holm       Holm-Bonferroni correction across the family of pairwise
                  significance tests.

    python -m src.extra_analyses                    # all six
    python -m src.extra_analyses --only prior,holm  # a subset

Results are written to ``results/`` as CSV and JSON and are consumed by the
report and by ``src/export_matlab.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                    # noqa: E402
from src.metrics import compute_metrics, bootstrap_ci          # noqa: E402
from src.train import (PROTOCOLS, build_inputs, load_data,     # noqa: E402
                       set_seed, train_one, variant_spec)

PRED = C.RESULTS / "preds"
MIN_FOLD = 300          # communities smaller than this are not test folds
LOCO_MODELS = ["text", "concat_meta", "cgf", "cgf_adv"]
LOCO_SEED = 42


# --------------------------------------------------------------------------- #
def _device() -> str:
    import os
    return "cpu" if os.environ.get("FORCE_CPU") else C.get_device()


def _mean_pred(runs: pd.DataFrame, protocol: str, model: str) -> np.ndarray:
    """Seed-averaged test probabilities, as reported throughout Chapter 6."""
    seeds = runs[(runs.protocol == protocol)
                 & (runs.model == model)]["seed"].unique()
    ps = [np.load(PRED / f"{protocol}__{model}__{int(s)}.npy")
          for s in seeds
          if (PRED / f"{protocol}__{model}__{int(s)}.npy").exists()]
    return np.mean(ps, axis=0)


# --------------------------------------------------------------------------- #
def analysis_loco(df: pd.DataFrame, feats: dict) -> pd.DataFrame:
    """Leave-one-community-out rotation over the source communities.

    Protocol B holds out one fixed partition of five communities, so its
    absolute numbers carry a selection uncertainty the bootstrap intervals do
    not capture. This rotation removes that objection: each sufficiently large
    community serves as the test set in turn, the next community in the
    rotation serves as validation, and the remainder train. Every fold is
    therefore fully source-disjoint, and the spread across folds is a direct
    estimate of how much the reported figure depends on which communities were
    held out.
    """
    device = _device()
    y = (1 - df["2_way_label"].to_numpy()).astype(np.int64)
    communities = sorted(df["subreddit"].unique())
    sizes = df["subreddit"].value_counts()
    folds = [c for c in communities if sizes[c] >= MIN_FOLD]
    print(f"[loco] {len(folds)} test folds of {len(communities)} communities "
          f"(minimum fold size {MIN_FOLD})")

    rows = []
    for i, test_c in enumerate(folds):
        val_c = folds[(i + 1) % len(folds)]
        test_idx = np.where(df["subreddit"].to_numpy() == test_c)[0]
        val_idx = np.where(df["subreddit"].to_numpy() == val_c)[0]
        train_idx = np.where(~df["subreddit"].isin([test_c, val_c]))[0]
        idxs = {"train": train_idx, "val": val_idx, "test": test_idx}

        X = build_inputs(feats, df, train_idx)
        domains = pd.factorize(df["subreddit"])[0]
        for model in LOCO_MODELS:
            spec = variant_spec(model, X, df)
            if spec is None:
                continue
            set_seed(LOCO_SEED)
            out = train_one(X, y, idxs, spec, LOCO_SEED, device,
                            domains=domains)
            # A single-class fold has no meaningful macro-F1 or AUROC, so
            # accuracy is the only figure reported for it.
            m = compute_metrics(y[test_idx], out["p_test"])
            rows.append({"held_out": test_c, "n_test": len(test_idx),
                         "label": "fake" if y[test_idx][0] == 1 else "true",
                         "model": model, "accuracy": m["accuracy"],
                         "macro_f1": m["macro_f1"],
                         "mean_p_fake": float(out["p_test"].mean())})
            print(f"[loco] {test_c:<22s} {model:<12s} "
                  f"acc={m['accuracy']:.3f} n={len(test_idx)}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def analysis_probe(df: pd.DataFrame, n_perm: int = 1000) -> dict:
    """Uncertainty and a null distribution for the content-free probe.

    The probe's perfect score is the study's headline result, and a headline
    result reported without uncertainty is exactly the practice this report
    criticises. Two quantities are computed: a percentile bootstrap interval
    over test items, and a permutation null in which the community-to-label
    mapping is destroyed by shuffling labels within the training split. If the
    probe were exploiting anything other than the deterministic mapping, the
    null would not collapse to chance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    y = (1 - df["2_way_label"].to_numpy()).astype(np.int64)
    out: dict = {}
    for proto, col in PROTOCOLS.items():
        tr = np.where(df[col].to_numpy() == "train")[0]
        te = np.where(df[col].to_numpy() == "test")[0]
        enc = OneHotEncoder(handle_unknown="ignore")
        S = enc.fit_transform(df[["subreddit"]])

        clf = LogisticRegression(max_iter=1000).fit(S[tr], y[tr])
        p = clf.predict_proba(S[te])[:, 1]
        m = compute_metrics(y[te], p)
        lo, hi = bootstrap_ci(y[te], p, "macro_f1", n=2000)

        rng = np.random.default_rng(C.RANDOM_SEED)
        null = []
        for _ in range(n_perm):
            yp = rng.permutation(y[tr])
            c = LogisticRegression(max_iter=200).fit(S[tr], yp)
            null.append(compute_metrics(
                y[te], c.predict_proba(S[te])[:, 1])["macro_f1"])
        null = np.asarray(null)
        out[proto] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                      "ci_lo": lo, "ci_hi": hi,
                      "null_mean": float(null.mean()),
                      "null_sd": float(null.std()),
                      "null_max": float(null.max()),
                      "p_value": float((null >= m["macro_f1"]).mean()
                                       + 1 / n_perm),
                      "n_permutations": n_perm}
        print(f"[probe] {proto:<9s} macro-F1={m['macro_f1']:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  null={null.mean():.3f}"
              f"+/-{null.std():.3f}")
    return out


# --------------------------------------------------------------------------- #
def analysis_community(runs: pd.DataFrame, model: str = "cgf_llm_adv"
                       ) -> pd.DataFrame:
    """Per-held-out-community performance under the source-disjoint protocol.

    Section 5.4 commits to disaggregated reporting on the ground that a single
    aggregate conceals the disparities that matter when a system is deployed
    across communities with differing norms. This discharges that commitment
    from cached predictions, without retraining.
    """
    meta = pd.read_parquet(PRED / "source__test_meta.parquet")
    y = np.load(PRED / "source__y_test.npy")
    p = _mean_pred(runs, "source", model)
    rows = []
    for c in sorted(meta["subreddit"].unique()):
        k = (meta["subreddit"] == c).to_numpy()
        yhat = (p[k] >= 0.5).astype(int)
        rows.append({"community": c, "n": int(k.sum()),
                     "label": "fake" if y[k][0] == 1 else "true",
                     "accuracy": float((yhat == y[k]).mean()),
                     "mean_p_fake": float(p[k].mean())})
    out = pd.DataFrame(rows).sort_values("accuracy")
    print(f"[community] spread {out.accuracy.max() - out.accuracy.min():.3f} "
          f"accuracy points across {len(out)} held-out communities")
    return out


# --------------------------------------------------------------------------- #
def analysis_ranking(runs: pd.DataFrame,
                     models: list[str] | None = None) -> pd.DataFrame:
    """Precision at k on the source-disjoint test set.

    User need N1 states that the system ranks a review queue rather than
    deciding, which makes the precision of the top of that queue the operative
    metric. Reporting only macro-F1 would set a requirement and then measure
    something else.
    """
    models = models or ["subreddit_probe", "tfidf_lr", "text", "image",
                        "concat_meta", "cgf", "cgf_adv", "cgf_llm_adv",
                        "llm_zeroshot"]
    y = np.load(PRED / "source__y_test.npy")
    base = float(y.mean())
    rows = []
    for m in models:
        if runs[(runs.protocol == "source") & (runs.model == m)].empty:
            continue
        p = _mean_pred(runs, "source", m)
        order = np.argsort(-p)
        row = {"model": m, "prevalence": base}
        for k in (50, 100, 500, 1000):
            row[f"precision_at_{k}"] = float(y[order[:k]].mean())
        row["lift_at_100"] = row["precision_at_100"] / base
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("precision_at_100", ascending=False)
    print(f"[ranking] prevalence {base:.3f}; best P@100 "
          f"{out.precision_at_100.max():.3f}")
    return out


# --------------------------------------------------------------------------- #
def analysis_prior(df: pd.DataFrame, runs: pd.DataFrame,
                   n_boot: int = 200) -> dict:
    """Separate class-prior shift from source shift in the protocol gap.

    Holding out whole communities removes source leakage but also changes the
    class balance of the test set, so part of the measured drop is prior shift
    rather than leakage. The source-disjoint test set is resampled without
    replacement to match the random split's class prior, and the models are
    re-scored on that subsample. The residual drop is the part of the gap that
    prior shift does not explain.
    """
    y_rand = np.load(PRED / "random__y_test.npy")
    y_src = np.load(PRED / "source__y_test.npy")
    target = float(y_rand.mean())
    pos = np.where(y_src == 1)[0]
    neg = np.where(y_src == 0)[0]
    # Largest subsample with the target prior, taken without replacement.
    n_pos = min(len(pos), int(len(neg) * target / (1 - target)))
    n_neg = int(round(n_pos * (1 - target) / target))

    models = sorted(set(runs[runs.protocol == "source"]["model"])
                    & set(runs[runs.protocol == "random"]["model"]))
    models = [m for m in models if m not in ("majority", "subreddit_probe")]
    rng = np.random.default_rng(C.RANDOM_SEED)

    rows = []
    for m in models:
        p_r = _mean_pred(runs, "random", m)
        p_s = _mean_pred(runs, "source", m)
        f_rand = compute_metrics(y_rand, p_r)["macro_f1"]
        f_src = compute_metrics(y_src, p_s)["macro_f1"]
        vals = []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                                  rng.choice(neg, n_neg, replace=False)])
            vals.append(compute_metrics(y_src[idx], p_s[idx])["macro_f1"])
        f_matched = float(np.mean(vals))
        rows.append({"model": m, "random": f_rand, "source": f_src,
                     "source_prior_matched": f_matched,
                     "total_gap": f_rand - f_src,
                     "gap_after_prior_match": f_rand - f_matched})

    t = pd.DataFrame(rows)
    total = float(t.total_gap.mean())
    residual = float(t.gap_after_prior_match.mean())
    summary = {"random_prior_fake": target,
               "source_prior_fake": float(y_src.mean()),
               "matched_subsample_n": int(n_pos + n_neg),
               "mean_total_gap": total,
               "mean_gap_after_prior_match": residual,
               "prior_shift_share": (total - residual) / total if total else 0.0,
               "n_models": len(t)}
    print(f"[prior] total gap {total:.3f}; after prior matching {residual:.3f} "
          f"({summary['prior_shift_share']*100:.1f}% attributable to prior)")
    t.to_csv(C.RESULTS / "prior_decomposition.csv", index=False)
    return summary


# --------------------------------------------------------------------------- #
def analysis_holm() -> pd.DataFrame:
    """Holm-Bonferroni correction across the family of pairwise tests.

    Sixty pairwise comparisons are reported, so uncorrected p-values overstate
    the evidence. Holm's step-down procedure is used rather than Bonferroni
    because it controls the same family-wise error rate with strictly greater
    power, and corrections are applied within protocol, which is the family a
    reader actually compares across.
    """
    sig = pd.read_csv(C.RESULTS / "significance.csv")
    out = []
    for proto, g in sig.groupby("protocol"):
        g = g.sort_values("p_value").reset_index(drop=True)
        n = len(g)
        adj, running = [], 0.0
        for i, p in enumerate(g["p_value"]):
            running = max(running, min(1.0, (n - i) * p))
            adj.append(running)
        g["p_holm"] = adj
        g["significant_holm"] = g["p_holm"] < 0.05
        out.append(g)
    t = pd.concat(out, ignore_index=True)
    for proto, g in t.groupby("protocol"):
        print(f"[holm] {proto:<9s} {int(g.significant_holm.sum())}/{len(g)} "
              f"comparisons survive correction at alpha=0.05")
    return t


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="",
                    help="comma-separated subset of "
                         "loco,probe,community,ranking,prior,holm")
    ap.add_argument("--corpus", default="corpus.parquet")
    ap.add_argument("--prefix", default="feat")
    ap.add_argument("--permutations", type=int, default=1000)
    a = ap.parse_args()
    wanted = ([s.strip() for s in a.only.split(",") if s.strip()]
              or ["probe", "community", "ranking", "prior", "holm", "loco"])

    runs = pd.read_csv(C.RESULTS / "runs.csv")
    df, feats = load_data(a.corpus, a.prefix)
    summary: dict = {}

    if "probe" in wanted:
        summary["probe"] = analysis_probe(df, a.permutations)
    if "community" in wanted:
        analysis_community(runs).to_csv(
            C.RESULTS / "per_community.csv", index=False)
    if "ranking" in wanted:
        analysis_ranking(runs).to_csv(
            C.RESULTS / "ranking.csv", index=False)
    if "prior" in wanted:
        summary["prior"] = analysis_prior(df, runs)
    if "holm" in wanted:
        analysis_holm().to_csv(
            C.RESULTS / "significance_holm.csv", index=False)
    if "loco" in wanted:
        t = analysis_loco(df, feats)
        t.to_csv(C.RESULTS / "loco.csv", index=False)
        agg = (t.groupby("model")["accuracy"]
               .agg(["mean", "std", "min", "max", "count"]))
        summary["loco"] = json.loads(agg.to_json(orient="index"))
        print(agg.round(3).to_string())

    path = C.RESULTS / "extra_analyses.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(summary)
    path.write_text(json.dumps(existing, indent=2))
    print(f"[extra] wrote {path}")


if __name__ == "__main__":
    main()
