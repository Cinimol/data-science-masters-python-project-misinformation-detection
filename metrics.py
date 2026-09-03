"""
Evaluation metrics, uncertainty quantification and significance testing.

Point estimates alone are insufficient for this evaluation, which requires *rigorous* evaluation, so point estimates alone are
not enough. Every reported number is accompanied by either a bootstrap
confidence interval (single-run uncertainty attributable to the finite test
set) or a standard deviation across seeds (uncertainty attributable to
optimisation), and every headline comparison between two systems is backed by
a paired significance test on the same test items.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


# --------------------------------------------------------------------------- #
def expected_calibration_error(y: np.ndarray, p: np.ndarray,
                               n_bins: int = 15) -> float:
    """Equal-width binned ECE.

    A detector that is used to *flag* content for human review must report
    trustworthy probabilities, not just a good ranking; a system that is 99%
    confident and wrong 30% of the time is unsafe to deploy. ECE quantifies
    that gap and is discussed alongside accuracy throughout Chapter 6.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def compute_metrics(y: np.ndarray, p: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Full metric suite for a binary detector.

    ``p`` is P(class 1 = fake). Positive class is 'fake', so recall is the
    proportion of misinformation that is caught and precision is the
    proportion of flags that are justified -- the two quantities a platform
    moderation team actually trades off.
    """
    yhat = (p >= threshold).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, yhat, average="binary", zero_division=0)
    out = {
        "accuracy": accuracy_score(y, yhat),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "precision_fake": pr,
        "recall_fake": rc,
        "f1_fake": f1,
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
    }
    if len(np.unique(y)) > 1:
        out["auroc"] = roc_auc_score(y, p)
        out["auprc"] = average_precision_score(y, p)
    else:                                     # degenerate held-out source
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def _fast_scores(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    """Accuracy and macro-F1 from confusion counts, without sklearn.

    The bootstrap needs thousands of evaluations per system, and the general
    metric suite is far too slow at that volume. Counting the four cells of the
    confusion matrix with ``bincount`` gives identical numbers roughly three
    orders of magnitude faster, which is what makes 2,000-replicate intervals
    for every system affordable.
    """
    c = np.bincount(y * 2 + yhat, minlength=4)
    tn, fp, fn, tp = c[0], c[1], c[2], c[3]
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    f1_pos = 2 * tp / max(1, 2 * tp + fp + fn)
    f1_neg = 2 * tn / max(1, 2 * tn + fn + fp)
    return float(acc), float((f1_pos + f1_neg) / 2)


def _boot_indices(n_items: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_items, size=(n, n_items))


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str = "macro_f1",
                 n: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI over test items (resampling with replacement)."""
    if metric not in ("accuracy", "macro_f1"):
        raise ValueError("fast bootstrap supports accuracy and macro_f1")
    yhat = (p >= 0.5).astype(np.int64)
    y = y.astype(np.int64)
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        s = rng.integers(0, len(y), len(y))
        a, f = _fast_scores(y[s], yhat[s])
        vals[i] = a if metric == "accuracy" else f
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def mcnemar_test(y: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                 threshold: float = 0.5) -> dict:
    """Exact McNemar test on the paired correct/incorrect contingency table.

    Appropriate here because both systems are evaluated on identical test
    items, so the observations are paired and an unpaired test would be
    anti-conservative.
    """
    from scipy import stats

    a = ((pa >= threshold).astype(int) == y)
    b = ((pb >= threshold).astype(int) == y)
    n01 = int((a & ~b).sum())      # A right, B wrong
    n10 = int((~a & b).sum())      # A wrong, B right
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "statistic": 0.0}
    p = float(stats.binomtest(n01, n, 0.5).pvalue)
    stat = (abs(n01 - n10) - 1) ** 2 / n
    return {"n01": n01, "n10": n10, "p_value": p, "statistic": float(stat)}


def paired_bootstrap_delta(y: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                           metric: str = "macro_f1", n: int = 2000,
                           seed: int = 0) -> dict:
    """CI on the *difference* between two systems, resampling items jointly."""
    ya = (pa >= 0.5).astype(np.int64)
    yb = (pb >= 0.5).astype(np.int64)
    y = y.astype(np.int64)
    j = 0 if metric == "accuracy" else 1
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        s = rng.integers(0, len(y), len(y))
        ys = y[s]
        d[i] = _fast_scores(ys, ya[s])[j] - _fast_scores(ys, yb[s])[j]
    return {
        "delta_mean": float(d.mean()),
        "delta_lo": float(np.percentile(d, 2.5)),
        "delta_hi": float(np.percentile(d, 97.5)),
        "p_greater": float((d <= 0).mean()),
    }
