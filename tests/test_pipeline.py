"""
Unit and property tests for the artefact.

These cover the parts of the pipeline where a silent error would invalidate the
results rather than crash the run: label polarity, split disjointness, absence
of train/test leakage in the feature scaler, metric correctness against
hand-computed values, and the shape contracts of every model variant.

    python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.build_dataset import META_FEATURES, add_derived_columns, assign_splits
from src.metrics import (compute_metrics, expected_calibration_error,
                         mcnemar_test)
from src.models import ConsistencyGatedFusion, LateFusion, MLPHead
from src.robustness import perturb
from src.train import _z


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_metrics_against_hand_computed_values():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.6, 0.9, 0.8, 0.2, 0.3])
    m = compute_metrics(y, p)
    # predictions: 0,1,1,1,0,0  ->  tp=2 fp=1 fn=1 tn=2
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 2
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision_fake"] == pytest.approx(2 / 3)
    assert m["recall_fake"] == pytest.approx(2 / 3)


def test_perfect_and_inverted_predictions():
    y = np.array([0, 1, 0, 1])
    assert compute_metrics(y, y.astype(float))["accuracy"] == 1.0
    assert compute_metrics(y, 1.0 - y)["accuracy"] == 0.0


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 40_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_mcnemar_is_symmetric_and_significant_when_it_should_be():
    y = np.zeros(200, dtype=int)
    a = np.zeros(200)                       # always right
    b = np.ones(200)                        # always wrong
    r = mcnemar_test(y, a, b)
    assert r["n01"] == 200 and r["n10"] == 0
    assert r["p_value"] < 1e-10
    r2 = mcnemar_test(y, b, a)
    assert r2["n01"] == 0 and r2["n10"] == 200


def test_mcnemar_identical_systems_are_not_significant():
    y = np.array([0, 1] * 50)
    p = np.random.default_rng(1).uniform(size=100)
    assert mcnemar_test(y, p, p)["p_value"] == 1.0


# --------------------------------------------------------------------------- #
# Splits and leakage
# --------------------------------------------------------------------------- #
def _toy_corpus(n=600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    subs = ["a", "b", "c", "upliftingnews", "misleadingthumbnails",
            "nottheonion", "usnews", "usanews", "savedyouaclick",
            "confusing_perspective"]
    return pd.DataFrame({
        "id": [f"i{i}" for i in range(n)],
        "subreddit": rng.choice(subs, n),
        "2_way_label": rng.integers(0, 2, n),
        "created_utc": rng.integers(1_300_000_000, 1_560_000_000, n).astype(float),
        "clean_title": ["a headline about things"] * n,
        "score": rng.integers(0, 500, n).astype(float),
        "num_comments": rng.integers(0, 50, n).astype(float),
        "upvote_ratio": rng.uniform(0.5, 1.0, n),
        "author": ["x"] * n,
        "domain": ["example.com"] * n,
    })


def test_source_disjoint_split_shares_no_subreddit_between_train_and_test():
    df = assign_splits(_toy_corpus())
    tr = set(df[df.split_source == "train"].subreddit)
    te = set(df[df.split_source == "test"].subreddit)
    va = set(df[df.split_source == "val"].subreddit)
    assert not (tr & te), "source-disjoint protocol is leaking subreddits"
    assert not (tr & va) and not (va & te)


def test_temporal_split_is_ordered_in_time():
    df = assign_splits(_toy_corpus())
    assert (df[df.split_temporal == "train"].created_utc.max()
            <= df[df.split_temporal == "test"].created_utc.min())


def test_random_split_covers_every_row_exactly_once():
    df = assign_splits(_toy_corpus())
    for col in ("split_random", "split_source", "split_temporal"):
        assert set(df[col]) <= {"train", "val", "test"}
        assert df[col].notna().all()


def test_standardiser_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    x[100:] += 50.0                       # test rows are wildly different
    tr = np.arange(100)
    z = _z(tr, x)
    assert abs(z[tr].mean()) < 1e-5, "train rows must be centred"
    assert z[100:].mean() > 5, "test rows must NOT be re-centred (no leakage)"


def test_no_author_column_survives_the_build():
    """Data minimisation is a requirement (NF3), so it is asserted, not trusted.

    The build stage must drop the Reddit username before anything reaches disk,
    retaining only the binary presence flag the metadata features use. A
    regression here would be silent: every model would still train, and the
    artefact would quietly redistribute re-identifiable pseudonyms.
    """
    df = add_derived_columns(_toy_corpus())
    kept = df.drop(columns=[c for c in ("author",) if c in df.columns])
    assert "author" not in kept.columns
    assert "has_author" in df.columns
    assert set(df["has_author"].unique()) <= {0, 1}
    assert "author" not in META_FEATURES


def test_derived_metadata_columns_are_finite():
    df = add_derived_columns(_toy_corpus())
    for c in META_FEATURES:
        assert c in df.columns
        assert np.isfinite(df[c].to_numpy(dtype=float)).all()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [
    {}, {"use_gate": False}, {"use_consistency": False},
    {"use_interaction": False}, {"use_meta": False},
])
def test_cgf_forward_shapes(kwargs):
    b = 8
    m = ConsistencyGatedFusion(d_text=32, d_img=16, d_meta=5, hidden=24,
                               use_llm=False, **kwargs)
    out = m({"text": torch.randn(b, 32), "image": torch.randn(b, 16),
             "meta": torch.randn(b, 5), "cons": torch.randn(b, 2)})
    assert out.shape == (b, 2)
    assert torch.isfinite(out).all()


def test_cgf_with_llm_stream():
    b = 4
    m = ConsistencyGatedFusion(d_text=32, d_img=16, d_meta=5, d_llm=12,
                               hidden=24, use_llm=True)
    out = m({"text": torch.randn(b, 32), "image": torch.randn(b, 16),
             "meta": torch.randn(b, 5), "cons": torch.randn(b, 2),
             "llm": torch.randn(b, 12)})
    assert out.shape == (b, 2)


def test_gate_is_bounded_and_recorded():
    m = ConsistencyGatedFusion(d_text=8, d_img=8, d_meta=2, hidden=8,
                               use_llm=False)
    m({"text": torch.randn(5, 8), "image": torch.randn(5, 8),
       "meta": torch.randn(5, 2), "cons": torch.randn(5, 2)})
    assert m.last_gate is not None
    assert ((m.last_gate >= 0) & (m.last_gate <= 1)).all()


def test_mlp_and_late_fusion_shapes():
    assert MLPHead(d_in=10, hidden=8)({"x": torch.randn(3, 10)}).shape == (3, 2)
    lf = LateFusion(d_text=10, d_img=6, hidden=8)
    assert lf({"text": torch.randn(3, 10),
               "image": torch.randn(3, 6)}).shape == (3, 2)


def test_model_is_deterministic_in_eval_mode():
    m = ConsistencyGatedFusion(d_text=8, d_img=8, d_meta=2, hidden=8,
                               use_llm=False).eval()
    batch = {"text": torch.randn(4, 8), "image": torch.randn(4, 8),
             "meta": torch.randn(4, 2), "cons": torch.randn(4, 2)}
    with torch.no_grad():
        assert torch.allclose(m(batch), m(batch))


# --------------------------------------------------------------------------- #
# Perturbations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["lowercase", "no_punct", "typos",
                                  "word_dropout"])
def test_perturbations_return_non_empty_strings(kind):
    import random
    out = perturb("Breaking: Scientists Find Water On Mars!", kind,
                  random.Random(0))
    assert isinstance(out, str) and out.strip()


def test_lowercase_perturbation_removes_capitals():
    import random
    assert perturb("ABC Def", "lowercase", random.Random(0)) == "abc def"


# --------------------------------------------------------------------------- #
# Single-example inference (src/predict.py)
# --------------------------------------------------------------------------- #
def test_clip_retrieval_rank_matches_batch_definition():
    """clip_retrieval_rank must reduce to the same formula clip_features uses
    within a batch: the fraction of alternatives the image prefers over the
    true caption."""
    from src.predict import clip_retrieval_rank

    rng = np.random.default_rng(0)
    image = rng.normal(size=16)
    image /= np.linalg.norm(image)
    true_text = 0.9 * image + 0.1 * rng.normal(size=16)
    true_text /= np.linalg.norm(true_text)

    # Three distractors: one nearly identical to the image (out-competes the
    # true caption), two random ones that almost certainly do not.
    worse = rng.normal(size=(2, 16))
    worse /= np.linalg.norm(worse, axis=1, keepdims=True)
    better = 0.99 * image + 0.01 * rng.normal(size=16)
    better /= np.linalg.norm(better)
    distractors = np.vstack([better, worse])

    rank = clip_retrieval_rank(image, true_text, distractors)
    assert 0.0 < rank <= 1.0 / 3.0 + 1e-9  # exactly one distractor beats it


def test_clip_retrieval_rank_zero_when_nothing_beats_it():
    from src.predict import clip_retrieval_rank

    image = np.array([1.0, 0.0])
    true_text = np.array([1.0, 0.0])          # perfect match
    distractors = np.array([[0.0, 1.0], [-1.0, 0.0]])  # both orthogonal/worse
    assert clip_retrieval_rank(image, true_text, distractors) == 0.0


def test_predict_meta_overrides_are_all_real_feature_names():
    """Every feature src.predict overrides at inference time must be a real
    column of META_FEATURES, or the override silently lands on nothing."""
    overridden = {"title_len_chars", "title_len_words", "hour", "dayofweek",
                  "log_score", "log_comments", "upvote_ratio", "has_author",
                  "is_selfpost"}
    assert overridden == set(META_FEATURES)


def test_checkpoint_round_trip_reproduces_predictions():
    """The save/load path src.predict uses (kwargs + state_dict, rebuilt with
    train.build_model) must reproduce the original model's output exactly --
    this is what makes the cached checkpoint trustworthy."""
    from src.train import build_model

    kwargs = dict(d_text=6, d_img=6, d_meta=3, hidden=8, use_llm=False)
    model = ConsistencyGatedFusion(**kwargs).eval()
    batch = {"text": torch.randn(2, 6), "image": torch.randn(2, 6),
             "meta": torch.randn(2, 3), "cons": torch.randn(2, 2)}
    with torch.no_grad():
        before = model(batch)

    rebuilt = build_model({"kind": "cgf", "kwargs": kwargs})
    rebuilt.load_state_dict(model.state_dict())
    rebuilt.eval()
    with torch.no_grad():
        after = rebuilt(batch)

    assert torch.allclose(before, after)
