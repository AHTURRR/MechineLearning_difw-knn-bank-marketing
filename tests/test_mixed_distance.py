"""Unit tests for Developer 1: Mixed-Type Dissimilarity Metric."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mixed_type_distance import (
    MixedTypeDistanceMetric,
    MixedTypeEncoded,
    normalize_feature_weights,
)


def _make_mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_a": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "num_b": [-5.0, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0],
            "cat_a": ["a", "b", "a", "c", "b", "a", "c", "b", "a", "c"],
            "cat_b": ["x", "x", "y", "y", "x", "y", "x", "y", "x", "y"],
        }
    )


def test_self_distance_is_zero() -> None:
    df = _make_mixed_df()
    metric = MixedTypeDistanceMetric(
        numeric_cols=["num_a", "num_b"],
        categorical_cols=["cat_a", "cat_b"],
        auto_engineer_pdays=False,
    ).fit(df)

    d_pair = metric.compute_pairwise_feature_distances(df, df)
    diag = np.diagonal(d_pair, axis1=0, axis2=1)
    assert np.allclose(diag, 0.0)

    d_weighted = metric.compute_weighted_distance_matrix(df, df)
    assert np.allclose(np.diag(d_weighted), 0.0)


def test_symmetry_and_bounds_with_outliers() -> None:
    train_df = _make_mixed_df()
    metric = MixedTypeDistanceMetric(
        numeric_cols=["num_a", "num_b"],
        categorical_cols=["cat_a", "cat_b"],
        auto_engineer_pdays=False,
    ).fit(train_df)

    query_df = pd.DataFrame(
        {
            "num_a": [-1000.0, 25.0, 9999.0],
            "num_b": [500.0, 3.0, -500.0],
            "cat_a": ["a", "b", "c"],
            "cat_b": ["x", "y", "x"],
        }
    )

    d_qt = metric.compute_pairwise_feature_distances(query_df, train_df)
    d_tq = metric.compute_pairwise_feature_distances(train_df, query_df)
    assert np.allclose(d_qt, np.transpose(d_tq, (1, 0, 2)))
    assert float(d_qt.min()) >= 0.0
    assert float(d_qt.max()) <= 1.0

    w = np.array([0.4, 0.3, 0.2, 0.1])
    dw_qt = metric.compute_weighted_distance_matrix(query_df, train_df, weights=w, batch_size=2)
    dw_tq = metric.compute_weighted_distance_matrix(train_df, query_df, weights=w, batch_size=3)
    assert np.allclose(dw_qt, dw_tq.T)
    assert float(dw_qt.min()) >= 0.0
    assert float(dw_qt.max()) <= 1.0


def test_unseen_category_yields_mismatch_one() -> None:
    train_df = _make_mixed_df()
    metric = MixedTypeDistanceMetric(
        numeric_cols=["num_a"],
        categorical_cols=["cat_a"],
        auto_engineer_pdays=False,
    ).fit(train_df)

    test_df = pd.DataFrame({"num_a": [10.0, 10.0], "cat_a": ["unseen_label", "unseen_label"]})
    enc_test = metric.transform(test_df)
    assert (enc_test.categorical[:, 0] == -1).all()

    d_tt = metric.compute_pairwise_feature_distances(enc_test, enc_test)
    # Categorical feature is index 1 after 1 numeric feature
    assert np.all(d_tt[:, :, 1] == 1.0)


def test_leak_free_pdays_feature_engineering() -> None:
    train_df = pd.DataFrame(
        {
            "age": [25, 32, 45, 51, 38, 29, 60, 41],
            "balance": [100, 500, 1200, -50, 300, 800, 2500, 0],
            "day": [5, 10, 15, 20, 25, 8, 12, 18],
            "campaign": [1, 2, 3, 1, 4, 2, 1, 5],
            "pdays": [-1, 100, 200, -1, 300, -1, 400, -1],
            "previous": [0, 1, 2, 0, 3, 0, 1, 0],
            "job": ["admin.", "technician", "services", "management", "admin.", "blue-collar", "retired", "student"],
            "marital": ["single", "married", "married", "single", "divorced", "single", "married", "single"],
            "education": ["tertiary", "secondary", "secondary", "tertiary", "primary", "secondary", "tertiary", "secondary"],
            "default": ["no"] * 8,
            "housing": ["yes", "no", "yes", "no", "yes", "no", "no", "yes"],
            "loan": ["no", "yes", "no", "no", "yes", "no", "no", "no"],
            "contact": ["cellular", "cellular", "unknown", "telephone", "cellular", "unknown", "cellular", "telephone"],
            "month": ["may", "jun", "jul", "aug", "nov", "may", "sep", "oct"],
            "poutcome": ["unknown", "failure", "success", "unknown", "other", "unknown", "success", "unknown"],
        }
    )
    # Positive pdays in train are [100, 200, 300, 400] -> median = 250.0
    metric = MixedTypeDistanceMetric().fit(train_df)
    assert metric.feature_engineer is not None
    assert metric.feature_engineer.pdays_median_pos_ == pytest.approx(250.0)

    test_df = train_df.iloc[:2].copy()
    test_df["pdays"] = [-1, 250]  # should use training M_pos=250.0 -> 0.0 and 1/(1+1)=0.5
    enc_test = metric.transform(test_df)
    assert isinstance(enc_test, MixedTypeEncoded)
    assert enc_test.n_features == 16
    pdays_recency_idx = metric.numeric_cols_.index("pdays_recency")
    assert enc_test.numeric[0, pdays_recency_idx] == pytest.approx(0.0)
    assert enc_test.numeric[1, pdays_recency_idx] == pytest.approx(0.5)


def test_batched_weighted_distance_matches_tensor_dot() -> None:
    df = _make_mixed_df()
    metric = MixedTypeDistanceMetric(
        numeric_cols=["num_a", "num_b"],
        categorical_cols=["cat_a", "cat_b"],
        auto_engineer_pdays=False,
    ).fit(df)

    raw_weights = np.array([2.0, 1.0, 3.0, 4.0])
    w_norm = normalize_feature_weights(raw_weights, n_features=4)
    assert w_norm.sum() == pytest.approx(1.0)

    d_tensor = metric.compute_pairwise_feature_distances(df.iloc[:7], df.iloc[3:])
    expected = np.tensordot(d_tensor, w_norm, axes=([2], [0]))
    batched = metric.compute_weighted_distance_matrix(
        df.iloc[:7], df.iloc[3:], weights=raw_weights, batch_size=2
    )
    assert np.allclose(batched, expected, atol=1e-12)


def test_invalid_inputs_raise_value_error() -> None:
    bad_df = pd.DataFrame({"const": [1.0, 1.0, 1.0, 1.0], "cat": ["a", "b", "a", "b"]})
    metric = MixedTypeDistanceMetric(
        numeric_cols=["const"],
        categorical_cols=["cat"],
        auto_engineer_pdays=False,
    )
    with pytest.raises(ValueError):
        metric.fit(bad_df)

    with pytest.raises(ValueError):
        normalize_feature_weights([-0.1, 1.1], n_features=2)
    with pytest.raises(ValueError):
        normalize_feature_weights([0.0, 0.0], n_features=2)


def test_openblas_categorical_acceleration_matches_exact_with_unseen() -> None:
    rng = np.random.default_rng(7)
    n_r, n_q = 90, 60  # n_q * n_r = 5400 >= 4096 triggers BLAS path
    train_df = pd.DataFrame(
        {
            "num1": rng.normal(10.0, 3.0, size=n_r),
            "cat1": rng.choice(["a", "b", "c"], size=n_r),
            "cat2": rng.choice(["x", "y"], size=n_r),
        }
    )
    query_df = pd.DataFrame(
        {
            "num1": rng.normal(10.0, 3.0, size=n_q),
            "cat1": rng.choice(["a", "b", "unseen_val"], size=n_q),
            "cat2": rng.choice(["x", "y"], size=n_q),
        }
    )
    metric = MixedTypeDistanceMetric(
        numeric_cols=["num1"],
        categorical_cols=["cat1", "cat2"],
        auto_engineer_pdays=False,
    ).fit(train_df)

    w = np.array([0.25, 0.45, 0.30])
    d_tensor = metric.compute_pairwise_feature_distances(query_df, train_df)
    expected = np.tensordot(d_tensor, w, axes=([2], [0]))
    blas_mat = metric.compute_weighted_distance_matrix(query_df, train_df, weights=w, batch_size=25)
    assert np.allclose(blas_mat, expected, atol=1e-12)

