"""Developer 1: Domain-Informed Mixed-Type Dissimilarity Metric for DIFW-KNN.

Implements the leak-free mixed-type feature encoding and distance formulation:
  - Numeric feature j:
        d_j(x, z) = min(1, |x_j - z_j| / (Q95_j - Q05_j))
    where Q05_j and Q95_j are estimated strictly from the training split.
  - Categorical / binary feature j:
        d_j(x, z) = 0 if x_j == z_j (and both values exist in training vocabulary), else 1.
  - Feature-weighted mixed distance:
        D_w(x, z) = sum_{j=1}^p w_j * d_j(x, z),  with w_j >= 0 and sum_j w_j = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class MixedTypeEncoded:
    """Container holding separated numeric and integer-encoded categorical arrays."""

    numeric: np.ndarray
    categorical: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        if self.numeric.shape[0] > 0 or self.numeric.ndim == 2:
            return int(self.numeric.shape[0])
        return int(self.categorical.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.numeric.shape[1] + self.categorical.shape[1])


def normalize_feature_weights(
    weights: np.ndarray | Sequence[float] | None,
    n_features: int,
) -> np.ndarray:
    """Validate non-negative feature weights and normalize them to sum to 1.0."""
    if n_features <= 0:
        raise ValueError(f"n_features harus > 0, diterima {n_features}.")
    if weights is None:
        return np.full(n_features, 1.0 / n_features, dtype=np.float64)

    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n_features,):
        raise ValueError(f"Panjang weights harus ({n_features},), diterima {w.shape}.")
    if not np.isfinite(w).all():
        raise ValueError("weights tidak boleh mengandung NaN atau Inf.")
    if np.any(w < -1e-12):
        raise ValueError("Seluruh elemen weights harus non-negatif (w_j >= 0).")

    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("Jumlah total weights harus > 0.")
    if abs(total - 1.0) <= 1e-12:
        return w
    return w / total


class MixedTypeDistanceMetric(BaseEstimator, TransformerMixin):
    """Leak-free mixed-type encoder and pairwise distance metric for DIFW-KNN (E2-E5).

    Supports both:
      1. Canonical UCI Bank Marketing data (automatically invoking DomainFeatureEngineer
         to transform raw 'pdays' -> ('previously_contacted', 'pdays_recency') when
         'pdays' is present and auto_engineer_pdays=True).
      2. General or synthetic tabular DataFrames with explicit numeric_cols / categorical_cols
         or inferred column types.
    """

    def __init__(
        self,
        numeric_cols: Sequence[str] | None = None,
        categorical_cols: Sequence[str] | None = None,
        auto_engineer_pdays: bool = True,
        q_low: float = 0.05,
        q_high: float = 0.95,
    ) -> None:
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError(f"Kuantil tidak valid: q_low={q_low}, q_high={q_high}.")
        self.numeric_cols = list(numeric_cols) if numeric_cols is not None else None
        self.categorical_cols = list(categorical_cols) if categorical_cols is not None else None
        self.auto_engineer_pdays = auto_engineer_pdays
        self.q_low = float(q_low)
        self.q_high = float(q_high)

        self.feature_engineer: Any = None
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.feature_names_: list[str] = []
        self.q05_: np.ndarray | None = None
        self.q95_: np.ndarray | None = None
        self.ranges_: np.ndarray | None = None
        self.category_maps_: dict[str, dict[str, int]] = {}

    def _resolve_columns_on_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame (engineering pdays if applicable) and determine feature columns."""
        from src.preprocessing import (
            DOMAIN_CATEGORICAL_COLS,
            DOMAIN_FEATURE_COLS,
            DOMAIN_NUMERIC_COLS,
            DomainFeatureEngineer,
        )

        if self.numeric_cols is None and self.categorical_cols is None:
            if self.auto_engineer_pdays and "pdays" in X.columns:
                self.feature_engineer = DomainFeatureEngineer()
                X_prepared = self.feature_engineer.fit_transform(X)
                self.numeric_cols_ = list(DOMAIN_NUMERIC_COLS)
                self.categorical_cols_ = list(DOMAIN_CATEGORICAL_COLS)
            elif set(DOMAIN_FEATURE_COLS).issubset(X.columns):
                self.feature_engineer = None
                X_prepared = X[DOMAIN_FEATURE_COLS].copy()
                self.numeric_cols_ = list(DOMAIN_NUMERIC_COLS)
                self.categorical_cols_ = list(DOMAIN_CATEGORICAL_COLS)
            else:
                self.feature_engineer = None
                X_prepared = X.copy()
                inferred_num: list[str] = []
                inferred_cat: list[str] = []
                for col in X_prepared.columns:
                    dtype = X_prepared[col].dtype
                    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
                        inferred_cat.append(str(col))
                    else:
                        inferred_num.append(str(col))
                self.numeric_cols_ = inferred_num
                self.categorical_cols_ = inferred_cat
        else:
            self.feature_engineer = None
            X_prepared = X.copy()
            self.numeric_cols_ = list(self.numeric_cols) if self.numeric_cols is not None else []
            self.categorical_cols_ = (
                list(self.categorical_cols) if self.categorical_cols is not None else []
            )

        self.feature_names_ = self.numeric_cols_ + self.categorical_cols_
        if not self.feature_names_:
            raise ValueError("Minimal harus ada satu fitur numerik atau kategorikal.")

        missing = set(self.feature_names_) - set(X_prepared.columns)
        if missing:
            raise ValueError(f"Kolom fitur tidak ditemukan pada DataFrame: {sorted(missing)}")
        return X_prepared

    def _prepare_dataframe_on_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted domain feature engineering (if any) and validate required columns."""
        if self.feature_engineer is not None:
            X_prepared = self.feature_engineer.transform(X)
        else:
            X_prepared = X
        missing = set(self.feature_names_) - set(X_prepared.columns)
        if missing:
            raise ValueError(f"Kolom fitur tidak ditemukan saat transform: {sorted(missing)}")
        return X_prepared

    def _ensure_encoded(self, data: MixedTypeEncoded | pd.DataFrame) -> MixedTypeEncoded:
        if isinstance(data, MixedTypeEncoded):
            return data
        if isinstance(data, pd.DataFrame):
            return self.transform(data)
        raise TypeError(
            f"Input harus bertipe MixedTypeEncoded atau pd.DataFrame, diterima {type(data)!r}."
        )

    def fit(self, X: pd.DataFrame, y: Any = None) -> MixedTypeDistanceMetric:
        """Fit training-only quantile ranges (Q05, Q95) and categorical vocabularies."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MixedTypeDistanceMetric.fit mengharapkan input pd.DataFrame.")
        if len(X) == 0:
            raise ValueError("DataFrame input tidak boleh kosong.")

        X_prepared = self._resolve_columns_on_fit(X)

        if self.numeric_cols_:
            num_arr = X_prepared[self.numeric_cols_].to_numpy(dtype=np.float64)
            if not np.isfinite(num_arr).all():
                raise ValueError("Fitur numerik mengandung nilai NaN atau Inf.")
            q05 = np.quantile(num_arr, self.q_low, axis=0)
            q95 = np.quantile(num_arr, self.q_high, axis=0)
            ranges = q95 - q05

            invalid_idx = np.where(ranges <= 0.0)[0]
            if len(invalid_idx) > 0:
                bad_cols = [
                    f"{self.numeric_cols_[i]} (Q05={q05[i]:.6f}, Q95={q95[i]:.6f})"
                    for i in invalid_idx
                ]
                raise ValueError(
                    "Rentang kuantil Q95 - Q05 <= 0 ditemukan pada fitur numerik: "
                    + ", ".join(bad_cols)
                )
        else:
            q05 = np.empty(0, dtype=np.float64)
            q95 = np.empty(0, dtype=np.float64)
            ranges = np.empty(0, dtype=np.float64)

        self.q05_ = q05
        self.q95_ = q95
        self.ranges_ = ranges

        self.category_maps_ = {}
        for col in self.categorical_cols_:
            cats = sorted(X_prepared[col].astype(str).unique().tolist())
            self.category_maps_[col] = {cat: idx for idx, cat in enumerate(cats)}

        return self

    def transform(self, X: pd.DataFrame) -> MixedTypeEncoded:
        """Encode DataFrame into separated numeric float64 and categorical int64 arrays."""
        if self.ranges_ is None:
            raise RuntimeError("MixedTypeDistanceMetric belum di-fit pada data training.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MixedTypeDistanceMetric.transform mengharapkan input pd.DataFrame.")

        X_prepared = self._prepare_dataframe_on_transform(X)
        n_rows = len(X_prepared)

        if self.numeric_cols_:
            num_arr = X_prepared[self.numeric_cols_].to_numpy(dtype=np.float64)
            if not np.isfinite(num_arr).all():
                raise ValueError("Fitur numerik saat transform mengandung NaN atau Inf.")
        else:
            num_arr = np.empty((n_rows, 0), dtype=np.float64)

        if self.categorical_cols_:
            cat_arr = np.empty((n_rows, len(self.categorical_cols_)), dtype=np.int64)
            for j, col in enumerate(self.categorical_cols_):
                mapping = self.category_maps_[col]
                cat_arr[:, j] = [mapping.get(val, -1) for val in X_prepared[col].astype(str)]
        else:
            cat_arr = np.empty((n_rows, 0), dtype=np.int64)

        return MixedTypeEncoded(
            numeric=num_arr,
            categorical=cat_arr,
            feature_names=tuple(self.feature_names_),
        )

    def get_numeric_params_table(self) -> pd.DataFrame:
        """Return DataFrame of fitted Q05, Q95, and range parameters per numeric feature."""
        if self.q05_ is None or self.q95_ is None or self.ranges_ is None:
            raise RuntimeError("MixedTypeDistanceMetric belum di-fit.")
        return pd.DataFrame(
            {
                "feature": self.numeric_cols_,
                "q05_train": np.round(self.q05_, 6),
                "q95_train": np.round(self.q95_, 6),
                "range_q95_q05": np.round(self.ranges_, 6),
            }
        )

    def compute_pairwise_feature_distances(
        self,
        a: MixedTypeEncoded | pd.DataFrame,
        b: MixedTypeEncoded | pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Compute per-feature distances d_j(x, z) in [0, 1] of shape (n_a, n_b, p)."""
        if self.ranges_ is None:
            raise RuntimeError("MixedTypeDistanceMetric belum di-fit.")
        enc_a = self._ensure_encoded(a)
        enc_b = enc_a if b is None else self._ensure_encoded(b)

        parts: list[np.ndarray] = []
        if self.numeric_cols_:
            d_num = np.minimum(
                1.0,
                np.abs(enc_a.numeric[:, None, :] - enc_b.numeric[None, :, :])
                / self.ranges_[None, None, :],
            )
            parts.append(d_num)

        if self.categorical_cols_:
            # Unseen category (-1) always yields mismatch distance 1 against any observation
            a_cat = enc_a.categorical[:, None, :]
            b_cat = enc_b.categorical[None, :, :]
            d_cat = ((a_cat != b_cat) | (a_cat < 0) | (b_cat < 0)).astype(np.float64)
            parts.append(d_cat)

        return np.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]

    def compute_aligned_feature_distances(
        self,
        a: MixedTypeEncoded | pd.DataFrame,
        b: MixedTypeEncoded | pd.DataFrame,
    ) -> np.ndarray:
        """Compute element-wise per-feature distances for aligned pairs (shape: n, p)."""
        if self.ranges_ is None:
            raise RuntimeError("MixedTypeDistanceMetric belum di-fit.")
        enc_a = self._ensure_encoded(a)
        enc_b = self._ensure_encoded(b)
        if enc_a.n_samples != enc_b.n_samples:
            raise ValueError("Aligned distance membutuhkan jumlah baris yang sama.")

        parts: list[np.ndarray] = []
        if self.numeric_cols_:
            d_num = np.minimum(
                1.0,
                np.abs(enc_a.numeric - enc_b.numeric) / self.ranges_[None, :],
            )
            parts.append(d_num)

        if self.categorical_cols_:
            d_cat = (
                (enc_a.categorical != enc_b.categorical)
                | (enc_a.categorical < 0)
                | (enc_b.categorical < 0)
            ).astype(np.float64)
            parts.append(d_cat)

        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def build_weighted_categorical_onehot(
        self,
        cat_arr: np.ndarray,
        w_cat: np.ndarray,
    ) -> np.ndarray:
        """Encode categorical matrix into sqrt(w_j)-scaled one-hot matrix (unseen < 0 -> zeros)."""
        cols: list[np.ndarray] = []
        for j, col in enumerate(self.categorical_cols_):
            wj = float(w_cat[j])
            if wj <= 0.0:
                continue
            n_c = len(self.category_maps_[col])
            eye = np.eye(n_c, dtype=np.float64) * np.sqrt(wj)
            col_vals = cat_arr[:, j]
            safe_idx = np.where(col_vals >= 0, col_vals, 0)
            oh = eye[safe_idx]
            if np.any(col_vals < 0):
                oh[col_vals < 0] = 0.0
            cols.append(oh)
        if not cols:
            return np.zeros((cat_arr.shape[0], 0), dtype=np.float64)
        return np.hstack(cols)

    def compute_weighted_distance_matrix(
        self,
        queries: MixedTypeEncoded | pd.DataFrame,
        references: MixedTypeEncoded | pd.DataFrame | None = None,
        weights: np.ndarray | Sequence[float] | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Compute weighted mixed-type distance matrix D_w(q, r) in memory-bounded batches."""
        if self.ranges_ is None:
            raise RuntimeError("MixedTypeDistanceMetric belum di-fit.")
        if batch_size <= 0:
            raise ValueError(f"batch_size harus > 0, diterima {batch_size}.")

        enc_q = self._ensure_encoded(queries)
        enc_r = enc_q if references is None else self._ensure_encoded(references)

        n_features = len(self.feature_names_)
        w = normalize_feature_weights(weights, n_features=n_features)

        n_num = len(self.numeric_cols_)
        n_cat = len(self.categorical_cols_)
        w_num = w[:n_num]
        w_cat = w[n_num:]
        w_cat_sum = float(np.sum(w_cat))

        n_q = enc_q.n_samples
        n_r = enc_r.n_samples
        dist_mat = np.zeros((n_q, n_r), dtype=np.float64)

        use_blas_cat = (n_cat > 0) and (w_cat_sum > 0.0) and (n_q * n_r >= 4096)
        r_oh = self.build_weighted_categorical_onehot(enc_r.categorical, w_cat) if use_blas_cat else None

        for start in range(0, n_q, batch_size):
            end = min(start + batch_size, n_q)
            q_num = enc_q.numeric[start:end]
            q_cat = enc_q.categorical[start:end]

            if use_blas_cat and r_oh is not None:
                q_oh = self.build_weighted_categorical_onehot(q_cat, w_cat)
                acc = np.maximum(0.0, w_cat_sum - (q_oh @ r_oh.T))
            else:
                acc = np.zeros((end - start, n_r), dtype=np.float64)
                for j in range(n_cat):
                    if w_cat[j] == 0.0:
                        continue
                    q_col = q_cat[:, j : j + 1]
                    r_col = enc_r.categorical[:, j][None, :]
                    mismatch = (q_col != r_col) | (q_col < 0) | (r_col < 0)
                    acc += w_cat[j] * mismatch

            for j in range(n_num):
                if w_num[j] == 0.0:
                    continue
                diff = np.abs(q_num[:, j : j + 1] - enc_r.numeric[:, j][None, :])
                acc += w_num[j] * np.minimum(1.0, diff / self.ranges_[j])

            dist_mat[start:end] = acc
        return dist_mat


MixedTypePreprocessor = MixedTypeDistanceMetric
