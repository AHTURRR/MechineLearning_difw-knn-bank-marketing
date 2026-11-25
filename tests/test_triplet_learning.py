"""Unit tests for Developer 2: Triplet Sampling, Margin Loss, & Weight Optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_weight_optimizer import FeatureWeightOptimizer, project_onto_simplex
from src.mixed_type_distance import MixedTypeDistanceMetric
from src.triplet_learning import (
    BalancedTripletSampler,
    numerical_gradient_check,
    triplet_margin_loss_and_grad,
)


def _make_separable_mixed_dataset(
    n_samples: int = 240,
    pos_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_pos = int(round(n_samples * pos_ratio))
    n_neg = n_samples - n_pos
    y = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
    perm = rng.permutation(n_samples)
    y = y[perm]

    signal_num = np.where(y == 1, rng.normal(2.5, 0.4, n_samples), rng.normal(-2.5, 0.4, n_samples))
    noise_num_1 = rng.normal(0.0, 1.0, n_samples)
    noise_num_2 = rng.uniform(-10.0, 10.0, n_samples)

    signal_cat = np.where(
        y == 1,
        rng.choice(["high", "very_high"], size=n_samples, p=[0.85, 0.15]),
        rng.choice(["low", "mid"], size=n_samples, p=[0.85, 0.15]),
    )
    noise_cat = rng.choice(["c1", "c2", "c3"], size=n_samples)

    df = pd.DataFrame(
        {
            "signal_num": signal_num,
            "noise_num_1": noise_num_1,
            "noise_num_2": noise_num_2,
            "signal_cat": signal_cat,
            "noise_cat": noise_cat,
        }
    )
    return df, y


@pytest.mark.parametrize("strategy", ["random", "semi_hard"])
def test_balanced_triplet_sampler_contract(strategy: str) -> None:
    df, y = _make_separable_mixed_dataset(n_samples=120, pos_ratio=0.25, seed=7)
    metric = MixedTypeDistanceMetric(
        numeric_cols=["signal_num", "noise_num_1", "noise_num_2"],
        categorical_cols=["signal_cat", "noise_cat"],
        auto_engineer_pdays=False,
    ).fit(df)
    enc = metric.transform(df)

    sampler = BalancedTripletSampler(
        n_triplets=200,
        positive_anchor_ratio=0.5,
        mining_strategy=strategy,
        candidate_pool_size=8,
        random_state=42,
    )
    batch = sampler.sample(enc, y, metric)

    assert batch.n_triplets == 200
    assert batch.n_features == 5
    assert batch.delta_feature_distances.shape == (200, 5)
    assert int((batch.anchor_labels == 1).sum()) == 100
    assert int((batch.anchor_labels == 0).sum()) == 100
    assert np.all(y[batch.anchor_idx] == y[batch.positive_idx])
    assert np.all(y[batch.anchor_idx] != y[batch.negative_idx])
    assert np.all(batch.anchor_idx != batch.positive_idx)
    assert float(batch.delta_feature_distances.min()) >= -1.0 - 1e-12
    assert float(batch.delta_feature_distances.max()) <= 1.0 + 1e-12


def test_simplex_projection_properties() -> None:
    rng = np.random.default_rng(123)
    for min_w in (0.0, 0.02, 0.05):
        v = rng.normal(0.0, 2.0, size=10)
        w = project_onto_simplex(v, min_weight=min_w)
        assert np.all(w >= min_w - 1e-12)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)
        # Idempotence
        w2 = project_onto_simplex(w, min_weight=min_w)
        assert np.allclose(w, w2, atol=1e-12)


def test_numerical_gradient_check() -> None:
    rng = np.random.default_rng(42)
    delta_d = rng.uniform(-0.8, 0.8, size=(150, 6))
    w_uniform = np.full(6, 1.0 / 6.0)
    w_random = project_onto_simplex(rng.uniform(0.0, 1.0, size=6))

    for w in (w_uniform, w_random):
        res = numerical_gradient_check(w, delta_d, margin=0.2, reg_lambda=0.05, eps=1e-6, tol=1e-5)
        assert res["passed"] is True
        assert res["max_abs_err"] < 1e-6


def test_optimizer_recovers_informative_features_and_decreases_loss() -> None:
    df, y = _make_separable_mixed_dataset(n_samples=240, pos_ratio=0.2, seed=42)
    metric = MixedTypeDistanceMetric(
        numeric_cols=["signal_num", "noise_num_1", "noise_num_2"],
        categorical_cols=["signal_cat", "noise_cat"],
        auto_engineer_pdays=False,
    ).fit(df)
    enc = metric.transform(df)

    opt = FeatureWeightOptimizer(
        margin=0.2,
        reg_lambda=0.02,
        learning_rate=0.5,
        max_iter=150,
        n_triplets=600,
        mining_strategy="semi_hard",
        random_state=42,
    ).fit(enc, y, metric)

    assert opt.weights_ is not None
    assert opt.initial_loss_ is not None and opt.final_loss_ is not None
    assert opt.final_loss_ < opt.initial_loss_
    assert np.all(opt.weights_ >= -1e-12)
    assert opt.weights_.sum() == pytest.approx(1.0, abs=1e-10)
    # Monotonic non-increasing loss trace
    diffs = np.diff(opt.loss_history_)
    assert np.all(diffs <= 1e-11)

    tbl = opt.get_weights_table().set_index("feature")
    min_signal_w = min(tbl.loc["signal_num", "weight"], tbl.loc["signal_cat", "weight"])
    max_noise_w = max(
        tbl.loc["noise_num_1", "weight"],
        tbl.loc["noise_num_2", "weight"],
        tbl.loc["noise_cat", "weight"],
    )
    assert min_signal_w > max_noise_w

    _, stats = opt.evaluate_seed_stability(enc, y, metric, seeds=(42, 100, 2026))
    assert stats["mean_pairwise_cosine"] > 0.90


def test_local_context_triplet_mining_validity_and_balance() -> None:
    df, y = _make_separable_mixed_dataset(n_samples=220, pos_ratio=0.2, seed=21)
    metric = MixedTypeDistanceMetric(
        numeric_cols=["signal_num", "noise_num_1", "noise_num_2"],
        categorical_cols=["signal_cat", "noise_cat"],
        auto_engineer_pdays=False,
    ).fit(df)
    enc = metric.transform(df)

    sampler = BalancedTripletSampler(
        n_triplets=400,
        positive_anchor_ratio=0.5,
        mining_strategy="local_context",
        candidate_pool_size=12,
        margin=0.3,
        random_state=42,
    )
    batch = sampler.sample(enc, y, metric)
    assert batch.n_triplets == 400
    assert float(np.mean(batch.anchor_labels == 1)) == pytest.approx(0.5)
    assert np.all(y[batch.anchor_idx] == y[batch.positive_idx])
    assert np.all(batch.anchor_idx != batch.positive_idx)
    assert np.all(y[batch.anchor_idx] != y[batch.negative_idx])

