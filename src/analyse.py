"""
Stage 5 -- aggregation, significance testing and figure generation.

Reads ``results/runs.csv`` and the cached test-set probabilities, then writes:

    results/summary.csv        mean +/- sd over seeds for every metric
    results/significance.csv   McNemar and paired-bootstrap comparisons
    results/report_numbers.json  every figure quoted in the written report
    figures/*.png              all report figures

Keeping this separate from training means the analysis can be re-run (and
audited) in seconds without touching the models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                    # noqa: E402
from src.metrics import (bootstrap_ci, compute_metrics,        # noqa: E402
                         mcnemar_test, paired_bootstrap_delta)

PRED = C.RESULTS / "preds"
plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "figure.autolayout": True,
})
PALETTE = {"random": "#4C72B0", "source": "#DD8452", "temporal": "#55A868"}

PRETTY = {
    "majority": "Majority class",
    "subreddit_probe": "Subreddit probe (no content)",
    "tfidf_lr": "TF-IDF + LR (text)",
    "llm_zeroshot": "LLM zero-shot judge",
    "llm_emb": "LLM embedding + MLP",
    "meta": "Metadata only",
    "cons": "Consistency scores only",
    "image": "Image only (CLIP)",
    "text": "Text only (DistilRoBERTa)",
    "late": "Late fusion",
    "concat": "Early fusion (concat)",
    "concat_meta": "Early fusion + metadata",
    "cgf_no_gate": "CGF ablation: no gate",
    "cgf_no_cons": "CGF ablation: no consistency",
    "cgf_no_inter": "CGF ablation: no interaction",
    "cgf_no_meta": "CGF ablation: no metadata",
    "concat_adv": "Early fusion + adversarial",
    "cgf": "CGF (proposed)",
    "cgf_adv": "CGF + adversarial",
    "cgf_llm": "CGF + LLM",
    "cgf_llm_adv": "CGF + LLM + adversarial (proposed)",
    "text_finetuned": "DistilRoBERTa fine-tuned end-to-end",
}
ORDER = ["majority", "subreddit_probe", "llm_zeroshot", "meta", "cons",
         "image", "tfidf_lr", "text", "llm_emb", "late", "concat",
         "concat_adv", "concat_meta", "cgf_no_gate", "cgf_no_cons",
         "cgf_no_inter", "cgf_no_meta", "cgf", "cgf_adv", "cgf_llm",
         "cgf_llm_adv"]
METRICS = ["accuracy", "macro_f1", "precision_fake", "recall_fake", "auroc",
           "auprc", "ece", "brier"]


# --------------------------------------------------------------------------- #
def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["protocol", "model"])
    out = g[METRICS].agg(["mean", "std"]).round(4)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    out["params"] = g["params"].max()
    out["train_seconds"] = g["train_seconds"].mean().round(2)
    return out.reset_index()


def load_pred(protocol: str, model: str, seed: int) -> np.ndarray | None:
    p = PRED / f"{protocol}__{model}__{seed}.npy"
    return np.load(p) if p.exists() else None


def mean_pred(runs: pd.DataFrame, protocol: str, model: str) -> np.ndarray:
    """Seed-averaged probability (an implicit ensemble; also lower variance)."""
    seeds = runs[(runs.protocol == protocol)
                 & (runs.model == model)]["seed"].unique()
    ps = [load_pred(protocol, model, int(s)) for s in seeds]
    ps = [p for p in ps if p is not None]
    return np.mean(ps, axis=0)


# --------------------------------------------------------------------------- #
def significance(runs: pd.DataFrame,
                 reference: str = "cgf_llm_adv") -> pd.DataFrame:
    rows = []
    for proto in runs.protocol.unique():
        y = np.load(PRED / f"{proto}__y_test.npy")
        models = [m for m in runs[runs.protocol == proto].model.unique()]
        ref = reference if reference in models else "cgf"
        pa = mean_pred(runs, proto, ref)
        for m in models:
            if m == ref:
                continue
            pb = mean_pred(runs, proto, m)
            mc = mcnemar_test(y, pa, pb)
            pb_ci = paired_bootstrap_delta(y, pa, pb, "macro_f1",
                                           n=C.N_BOOTSTRAP)
            rows.append({"protocol": proto, "reference": ref, "model": m,
                         **mc, **pb_ci})
    return pd.DataFrame(rows)


def confidence_intervals(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for proto in runs.protocol.unique():
        y = np.load(PRED / f"{proto}__y_test.npy")
        for m in runs[runs.protocol == proto].model.unique():
            p = mean_pred(runs, proto, m)
            met = compute_metrics(y, p)
            lo, hi = bootstrap_ci(y, p, "macro_f1", n=C.N_BOOTSTRAP)
            alo, ahi = bootstrap_ci(y, p, "accuracy", n=C.N_BOOTSTRAP)
            rows.append({"protocol": proto, "model": m, **met,
                         "macro_f1_lo": lo, "macro_f1_hi": hi,
                         "accuracy_lo": alo, "accuracy_hi": ahi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def fig_corpus(df: pd.DataFrame) -> None:
    """Figure 1 -- the source/label confound that motivates Protocol B."""
    t = pd.crosstab(df["subreddit"], df["2_way_label"])
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(t.index, t.get(1, 0), color="#4C72B0", label="labelled true")
    ax.barh(t.index, t.get(0, 0), left=t.get(1, 0), color="#C44E52",
            label="labelled fake")
    ax.invert_yaxis()
    ax.set_xlabel("posts in corpus")
    ax.set_title("Every Fakeddit subreddit maps to exactly one label")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(C.FIGURES / "fig1_corpus_confound.png")
    plt.close(fig)


def fig_main(ci: pd.DataFrame) -> None:
    """Figure 2 -- macro-F1 for every system under all three protocols."""
    models = [m for m in ORDER if m in set(ci.model)]
    protos = [p for p in ["random", "source", "temporal"]
              if p in set(ci.protocol)]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    h = 0.8 / len(protos)
    ypos = np.arange(len(models))
    for i, proto in enumerate(protos):
        sub = ci[ci.protocol == proto].set_index("model")
        v = [sub.loc[m, "macro_f1"] if m in sub.index else np.nan
             for m in models]
        lo = [sub.loc[m, "macro_f1_lo"] if m in sub.index else np.nan
              for m in models]
        hi = [sub.loc[m, "macro_f1_hi"] if m in sub.index else np.nan
              for m in models]
        err = np.abs(np.vstack([np.array(v) - lo, np.array(hi) - np.array(v)]))
        ax.barh(ypos + i * h, v, height=h, color=PALETTE[proto],
                label=f"{proto} split", xerr=err,
                error_kw={"lw": 0.7, "ecolor": "0.3"})
    ax.set_yticks(ypos + h * (len(protos) - 1) / 2)
    ax.set_yticklabels([PRETTY.get(m, m) for m in models])
    ax.invert_yaxis()
    ax.axvline(0.5, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("macro-F1 (95% bootstrap CI)")
    ax.set_xlim(0.3, 1.0)
    ax.set_title("Detection performance collapses once sources are held out")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(C.FIGURES / "fig2_main_results.png")
    plt.close(fig)


def fig_consistency(df: pd.DataFrame, feats: dict) -> None:
    """Figure 3 -- is CLIP image/caption agreement informative on its own?"""
    y = 1 - df["2_way_label"].to_numpy()
    sim = feats["clip_sim"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for lab, name, col in [(0, "true", "#4C72B0"), (1, "fake", "#C44E52")]:
        axes[0].hist(sim[y == lab], bins=60, density=True, alpha=0.55,
                     color=col, label=name)
    axes[0].set_xlabel("CLIP cosine(image, headline)")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    axes[0].set_title("Consistency by veracity class")

    order = (df.groupby("subreddit").apply(
        lambda g: sim[g.index].mean(), include_groups=False)
        .sort_values())
    cols = ["#C44E52" if df[df.subreddit == s]["2_way_label"].iloc[0] == 0
            else "#4C72B0" for s in order.index]
    axes[1].barh(order.index, order.values, color=cols)
    axes[1].set_xlabel("mean CLIP cosine")
    axes[1].set_title("Mean consistency by community")
    axes[1].tick_params(labelsize=6)
    fig.savefig(C.FIGURES / "fig3_consistency.png")
    plt.close(fig)


def fig_calibration(runs: pd.DataFrame, protocol: str,
                    models: list[str]) -> None:
    """Figure 4 -- reliability diagram on the source-disjoint protocol."""
    y = np.load(PRED / f"{protocol}__y_test.npy")
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for m in models:
        p = mean_pred(runs, protocol, m)
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins[1:-1], right=True)
        xs, ys = [], []
        for b in range(10):
            k = idx == b
            if k.sum() > 20:
                xs.append(p[k].mean())
                ys.append(y[k].mean())
        ax.plot(xs, ys, "o-", ms=3, lw=1.2, label=PRETTY.get(m, m))
    ax.set_xlabel("predicted P(fake)")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration ({protocol} split)")
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(C.FIGURES / f"fig4_calibration_{protocol}.png")
    plt.close(fig)


def fig_ablation(ci: pd.DataFrame) -> None:
    """Figure 5 -- component ablation on the source-disjoint protocol."""
    abl = ["cgf", "cgf_no_gate", "cgf_no_cons", "cgf_no_inter", "cgf_no_meta"]
    protos = [p for p in ["random", "source", "temporal"]
              if p in set(ci.protocol)]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    w = 0.8 / len(protos)
    x = np.arange(len(abl))
    for i, proto in enumerate(protos):
        sub = ci[ci.protocol == proto].set_index("model")
        base = sub.loc["cgf", "macro_f1"]
        v = [sub.loc[m, "macro_f1"] - base if m in sub.index else np.nan
             for m in abl]
        ax.bar(x + i * w, v, width=w, color=PALETTE[proto], label=proto)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(["full CGF", "-gate", "-consistency", "-interaction",
                        "-metadata"], fontsize=8)
    ax.set_ylabel("Δ macro-F1 vs full CGF")
    ax.set_title("Component ablation")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(C.FIGURES / "fig5_ablation.png")
    plt.close(fig)


def fig_error_by_category(runs: pd.DataFrame, protocol: str,
                          model: str) -> None:
    """Figure 6 -- which kinds of misinformation survive the detector?"""
    meta = pd.read_parquet(PRED / f"{protocol}__test_meta.parquet")
    y = np.load(PRED / f"{protocol}__y_test.npy")
    p = mean_pred(runs, protocol, model)
    ok = ((p >= 0.5).astype(int) == y)
    names = {0: "true", 2: "misleading content", 3: "imposter content",
             5: "false connection"}
    rows = (pd.DataFrame({"cat": meta["6_way_label"].map(names).values,
                          "correct": ok})
            .groupby("cat")["correct"].agg(["mean", "size"])
            .sort_values("mean"))
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.barh(rows.index, rows["mean"], color="#4C72B0")
    for i, (m, n) in enumerate(zip(rows["mean"], rows["size"])):
        ax.text(m + 0.01, i, f"n={n}", va="center", fontsize=7)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("per-category accuracy")
    ax.set_title(f"{PRETTY.get(model, model)}, {protocol} split")
    fig.savefig(C.FIGURES / f"fig6_error_by_category_{protocol}.png")
    plt.close(fig)


def fig_gate(runs: pd.DataFrame, protocol: str) -> dict:
    """Figure 7 -- does the learned gate track cross-modal consistency?"""
    seeds = runs[(runs.protocol == protocol)
                 & (runs.model == "cgf")]["seed"].unique()
    gs = [np.load(PRED / f"{protocol}__cgf__{int(s)}__gate.npy")
          for s in seeds
          if (PRED / f"{protocol}__cgf__{int(s)}__gate.npy").exists()]
    if not gs:
        return {}
    g = np.mean(gs, axis=0)
    meta = pd.read_parquet(PRED / f"{protocol}__test_meta.parquet")
    y = np.load(PRED / f"{protocol}__y_test.npy")
    base = np.load(C.PROCESSED / "feat_base.npz", allow_pickle=True)
    lookup = {v: i for i, v in enumerate(base["ids"])}
    pos = [lookup[i] for i in meta["id"]]
    sim = base["clip_sim"][pos]
    r = float(np.corrcoef(g, sim)[0, 1])

    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    for lab, name, col in [(0, "true", "#4C72B0"), (1, "fake", "#C44E52")]:
        ax.scatter(sim[y == lab], g[y == lab], s=3, alpha=0.25, color=col,
                   label=name, linewidths=0)
    ax.set_xlabel("CLIP cosine(image, headline)")
    ax.set_ylabel("mean visual gate activation")
    ax.set_title(f"Gate vs consistency (r = {r:.2f})")
    ax.legend(frameon=False, markerscale=3)
    fig.savefig(C.FIGURES / f"fig7_gate_{protocol}.png")
    plt.close(fig)
    return {"gate_consistency_pearson_r": r,
            "gate_mean_true": float(g[y == 0].mean()),
            "gate_mean_fake": float(g[y == 1].mean())}


def fig_cost(runs: pd.DataFrame, ci: pd.DataFrame,
             cost_json: Path | None) -> None:
    """Figure -- macro-F1 against inference cost (log scale).

    A detector that cannot run at platform scale is not a deployable answer to
    the research question, so cost belongs on the same axes as accuracy.
    """
    if cost_json is None or not cost_json.exists():
        return
    cost = json.loads(cost_json.read_text())
    sub = ci[ci.protocol == "source"].set_index("model")
    show = ["subreddit_probe", "tfidf_lr", "text", "image", "concat", "cgf",
            "llm_emb", "llm_zeroshot", "cgf_llm"]
    offsets = {"text": (-16, -14), "image": (-52, 9), "concat": (-40, -15),
               "cgf": (7, -4), "tfidf_lr": (6, 5),
               "subreddit_probe": (7, 3), "llm_emb": (-46, -14),
               "llm_zeroshot": (7, -3), "cgf_llm": (-18, 12)}
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for m in show:
        if m not in sub.index or m not in cost:
            continue
        x, y = cost[m], sub.loc[m, "macro_f1"]
        ax.scatter(x, y, s=34, color="#4C72B0", zorder=3)
        ax.annotate(PRETTY.get(m, m), (x, y), fontsize=6.5,
                    xytext=offsets.get(m, (5, 4)),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlim(3e-3, 5e3)
    ax.set_ylim(0.25, 0.75)
    ax.set_xlabel("inference cost (ms per item, 2-core CPU)")
    ax.set_ylabel("macro-F1 (source-disjoint)")
    ax.set_title("Accuracy against inference cost")
    fig.savefig(C.FIGURES / "fig8_cost.png")
    plt.close(fig)



# --------------------------------------------------------------------------- #
def _rank_agreement(ci: pd.DataFrame) -> dict:
    """How much does the conventional protocol misrank systems?

    Spearman correlation between the macro-F1 ordering of the *learned*
    systems under each pair of protocols. A value well below 1 means the
    conventional random split does not merely inflate scores, it reorders
    which method looks best -- which is what makes the confound a problem for
    the literature rather than a constant offset.
    """
    from scipy import stats

    p = ci.pivot(index="model", columns="protocol", values="macro_f1")
    skip = {"majority", "subreddit_probe", "llm_zeroshot"}
    q = p.loc[[m for m in p.index if m not in skip]]
    out = {}
    for a, b in [("random", "source"), ("random", "temporal"),
                 ("source", "temporal")]:
        if a in q.columns and b in q.columns:
            r = stats.spearmanr(q[a], q[b])
            out[f"{a}_vs_{b}"] = {"spearman_rho": round(float(r.statistic), 3),
                                  "p_value": round(float(r.pvalue), 5)}
    if {"random", "source"} <= set(q.columns):
        out["mean_macro_f1_drop_random_to_source"] = round(
            float((q["random"] - q["source"]).mean()), 4)
        best_r = q["random"].idxmax()
        order_s = list(q["source"].sort_values(ascending=False).index)
        out["best_under_random"] = best_r
        out["its_rank_under_source"] = order_s.index(best_r) + 1
        out["n_systems_ranked"] = int(len(q))
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs.csv")
    ap.add_argument("--corpus", default="corpus.parquet")
    ap.add_argument("--prefix", default="feat")
    args = ap.parse_args()

    runs = pd.read_csv(C.RESULTS / args.runs)
    df = pd.read_parquet(C.PROCESSED / args.corpus)
    feats = dict(np.load(C.PROCESSED / f"{args.prefix}_base.npz",
                         allow_pickle=True))

    summ = summarise(runs)
    summ.to_csv(C.RESULTS / "summary.csv", index=False)
    ci = confidence_intervals(runs)
    ci.to_csv(C.RESULTS / "confidence_intervals.csv", index=False)
    sig = significance(runs)
    sig.to_csv(C.RESULTS / "significance.csv", index=False)

    fig_corpus(df)
    fig_main(ci)
    fig_consistency(df, feats)
    fig_ablation(ci)
    top = [m for m in ["cgf_llm_adv", "cgf", "concat_meta", "text"]
           if m in set(runs.model)][:4]
    gate_stats = {}
    for proto in runs.protocol.unique():
        fig_calibration(runs, proto, top)
        fig_error_by_category(runs, proto, top[0])
        gate_stats[proto] = fig_gate(runs, proto)
    fig_cost(runs, ci, C.RESULTS / "inference_cost.json")

    nums = {
        "corpus": {
            "n_posts": int(len(df)),
            "n_subreddits": int(df["subreddit"].nunique()),
            "n_true": int((df["2_way_label"] == 1).sum()),
            "n_fake": int((df["2_way_label"] == 0).sum()),
            "date_min": str(pd.to_datetime(df["created_utc"].min(),
                                           unit="s").date()),
            "date_max": str(pd.to_datetime(df["created_utc"].max(),
                                           unit="s").date()),
            "median_title_words": float(df["title_len_words"].median()),
            "splits": {p: df[f"split_{p}"].value_counts().to_dict()
                       for p in ["random", "source", "temporal"]},
        },
        "gate": gate_stats,
        "protocol_rank_agreement": _rank_agreement(ci),
    }
    (C.RESULTS / "report_numbers.json").write_text(
        json.dumps(nums, indent=2, default=str))

    print(ci[ci.protocol == "source"]
          .sort_values("macro_f1", ascending=False)
          [["model", "accuracy", "macro_f1", "auroc", "ece"]]
          .round(4).to_string(index=False))
    print(f"\n[done] figures -> {C.FIGURES}")


if __name__ == "__main__":
    main()
