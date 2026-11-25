"""Core algorithm module for Proposed DIFW-KNN (Stage 05).

Implements:
- Exact Euclidean projection onto the probability simplex (w >= 0, sum(w) = 1).
- Balanced Triplet Sampling (50% positive anchors, 50% negative anchors) over
  Mixed-Type per-feature distances.
- Triplet Margin Loss + L2 uniform-prior regularization with analytic gradients,
  numerical gradient check, and Projected Gradient Descent optimizer.
- Batched Mixed-Type k-nearest neighbor search, inverse-distance local
  probability estimation, and Cost-Sensitive Decision Rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.cost_sensitive_decision import CostSensitiveDecisionRule
from src.feature_weight_optimizer import FeatureWeightOptimizer
from src.mixed_type_distance import (
    MixedTypeDistanceMetric,
    normalize_feature_weights,
)
from src.preprocessing import (
    DOMAIN_CATEGORICAL_COLS,
    DOMAIN_FEATURE_COLS,
    DOMAIN_NUMERIC_COLS,
    MixedTypeEncoded,
    MixedTypePreprocessor,
)


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex {w >= 0, sum(w) = 1}."""
    arr = np.asarray(v, dtype=np.float64)
    n = arr.shape[0]
    u = np.sort(arr)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=np.float64)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n, dtype=np.float64)
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    w = np.maximum(arr - theta, 0.0)
    # Normalize guard against tiny floating-point drift
    return w / float(w.sum())


@dataclass(frozen=True)
class TripletBatch:
    """Container for sampled balanced triplets and per-feature distance deltas."""

    anchor_idx: np.ndarray
    positive_idx: np.ndarray
    negative_idx: np.ndarray
    anchor_labels: np.ndarray
    delta_d: np.ndarray


def _slice_encoded(enc: MixedTypeEncoded, indices: np.ndarray) -> MixedTypeEncoded:
    """Slice a MixedTypeEncoded container by integer indices."""
    return MixedTypeEncoded(
        numeric=enc.numeric[indices],
        categorical=enc.categorical[indices],
        feature_names=enc.feature_names,
    )


def sample_balanced_triplets(
    mixed_prep: MixedTypePreprocessor,
    X_enc: MixedTypeEncoded,
    y: np.ndarray,
    n_triplets: int = 25000,
    random_state: int = 42,
) -> TripletBatch:
    """Sample class-balanced (anchor, positive, negative) triplets and compute delta_d.

    50% of anchors are drawn from class y=1 ('yes') and 50% from class y=0 ('no').
    Ensures y[a] == y[p], a != p, and y[a] != y[n] for every triplet.
    """
    y_arr = np.asarray(y, dtype=np.int64)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        raise ValueError("Minimal 2 observasi per kelas dibutuhkan untuk triplet sampling.")

    rng = np.random.default_rng(random_state)
    n_pos_anchors = n_triplets // 2
    n_neg_anchors = n_triplets - n_pos_anchors

    # Positive-anchor triplets
    a_pos = rng.choice(pos_idx, size=n_pos_anchors, replace=True)
    p_pos = rng.choice(pos_idx, size=n_pos_anchors, replace=True)
    clash = p_pos == a_pos
    while np.any(clash):
        p_pos[clash] = rng.choice(pos_idx, size=int(np.sum(clash)), replace=True)
        clash = p_pos == a_pos
    n_from_neg = rng.choice(neg_idx, size=n_pos_anchors, replace=True)

    # Negative-anchor triplets
    a_neg = rng.choice(neg_idx, size=n_neg_anchors, replace=True)
    p_neg = rng.choice(neg_idx, size=n_neg_anchors, replace=True)
    clash_neg = p_neg == a_neg
    while np.any(clash_neg):
        p_neg[clash_neg] = rng.choice(neg_idx, size=int(np.sum(clash_neg)), replace=True)
        clash_neg = p_neg == a_neg
    n_from_pos = rng.choice(pos_idx, size=n_neg_anchors, replace=True)

    anchor_idx = np.concatenate([a_pos, a_neg])
    positive_idx = np.concatenate([p_pos, p_neg])
    negative_idx = np.concatenate([n_from_neg, n_from_pos])

    perm = rng.permutation(n_triplets)
    anchor_idx = anchor_idx[perm]
    positive_idx = positive_idx[perm]
    negative_idx = negative_idx[perm]

    d_ap = mixed_prep.compute_aligned_feature_distances(
        _slice_encoded(X_enc, anchor_idx),
        _slice_encoded(X_enc, positive_idx),
    )
    d_an = mixed_prep.compute_aligned_feature_distances(
        _slice_encoded(X_enc, anchor_idx),
        _slice_encoded(X_enc, negative_idx),
    )
    delta_d = d_ap - d_an

    return TripletBatch(
        anchor_idx=anchor_idx,
        positive_idx=positive_idx,
        negative_idx=negative_idx,
        anchor_labels=y_arr[anchor_idx],
        delta_d=delta_d,
    )


class TripletFeatureWeightLearner:
    """Learn non-negative simplex feature weights via Triplet Margin Loss.

    Objective:
        L(w) = mean(max(0, m + delta_d @ w)) + reg_lambda * ||w - u||_2^2
    subject to w >= 0 and sum(w) = 1, where u_j = 1 / D.
    """

    def __init__(
        self,
        margin: float = 0.1,
        reg_lambda: float = 0.05,
        lr: float = 0.15,
        n_epochs: int = 200,
    ) -> None:
        self.margin = float(margin)
        self.reg_lambda = float(reg_lambda)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.weights_: np.ndarray | None = None
        self.loss_history_: list[dict[str, float]] = []

    def compute_loss_and_grad(
        self,
        w: np.ndarray,
        delta_d: np.ndarray,
    ) -> tuple[float, float, float, float, np.ndarray]:
        """Compute total loss, components, active triplet fraction, and gradient."""
        n_features = delta_d.shape[1]
        u = np.full(n_features, 1.0 / n_features, dtype=np.float64)

        raw_margins = self.margin + delta_d @ w
        active_mask = raw_margins > 0.0
        hinge_loss = float(np.mean(np.maximum(0.0, raw_margins)))
        diff_u = w - u
        reg_loss = float(self.reg_lambda * np.dot(diff_u, diff_u))
        total_loss = hinge_loss + reg_loss

        if np.any(active_mask):
            grad_hinge = np.sum(delta_d[active_mask], axis=0) / float(len(delta_d))
        else:
            grad_hinge = np.zeros(n_features, dtype=np.float64)
        grad_reg = 2.0 * self.reg_lambda * diff_u
        grad = grad_hinge + grad_reg
        active_frac = float(np.mean(active_mask))
        return total_loss, hinge_loss, reg_loss, active_frac, grad

    def fit(
        self,
        delta_d: np.ndarray,
        init_weights: np.ndarray | None = None,
    ) -> TripletFeatureWeightLearner:
        """Optimize feature weights over the probability simplex using Projected GD."""
        n_features = delta_d.shape[1]
        if init_weights is None:
            w = np.full(n_features, 1.0 / n_features, dtype=np.float64)
        else:
            w = project_to_simplex(init_weights)

        best_loss = float("inf")
        best_w = w.copy()
        history: list[dict[str, float]] = []

        # Log initial state at epoch 0
        init_total, init_hinge, init_reg, init_act, init_grad = self.compute_loss_and_grad(w, delta_d)
        history.append(
            {
                "epoch": 0,
                "total_loss": init_total,
                "hinge_loss": init_hinge,
                "reg_loss": init_reg,
                "active_frac": init_act,
                "grad_norm": float(np.linalg.norm(init_grad)),
            }
        )
        best_loss = init_total
        best_w = w.copy()

        for epoch in range(1, self.n_epochs + 1):
            _, _, _, _, grad = self.compute_loss_and_grad(w, delta_d)
            step = self.lr / np.sqrt(1.0 + 0.05 * epoch)
            w_cand = project_to_simplex(w - step * grad)
            total_l, hinge_l, reg_l, act_f, grad_next = self.compute_loss_and_grad(w_cand, delta_d)
            w = w_cand
            if total_l < best_loss:
                best_loss = total_l
                best_w = w.copy()
            history.append(
                {
                    "epoch": epoch,
                    "total_loss": total_l,
                    "hinge_loss": hinge_l,
                    "reg_loss": reg_l,
                    "active_frac": act_f,
                    "grad_norm": float(np.linalg.norm(grad_next)),
                }
            )

        self.weights_ = best_w
        self.loss_history_ = history
        return self

    def gradient_check(
        self,
        delta_d: np.ndarray,
        probe_weights: np.ndarray | None = None,
        eps: float = 1e-6,
    ) -> dict[str, Any]:
        """Verify analytic gradient against central finite differences."""
        n_features = delta_d.shape[1]
        if probe_weights is None:
            rng = np.random.default_rng(123)
            w0 = project_to_simplex(rng.uniform(0.3, 1.7, size=n_features))
        else:
            w0 = np.asarray(probe_weights, dtype=np.float64)

        # Ensure probe point avoids hinge kinks for smooth finite differences
        margins = np.abs(self.margin + delta_d @ w0)
        min_hinge_gap = float(np.min(margins))
        if min_hinge_gap <= 2.0 * eps:
            eps = min_hinge_gap * 0.25

        _, _, _, _, grad_analytic = self.compute_loss_and_grad(w0, delta_d)
        grad_numeric = np.zeros(n_features, dtype=np.float64)
        for j in range(n_features):
            e_j = np.zeros(n_features, dtype=np.float64)
            e_j[j] = eps
            l_plus, *_ = self.compute_loss_and_grad(w0 + e_j, delta_d)
            l_minus, *_ = self.compute_loss_and_grad(w0 - e_j, delta_d)
            grad_numeric[j] = (l_plus - l_minus) / (2.0 * eps)

        abs_err = np.abs(grad_analytic - grad_numeric)
        max_abs_err = float(np.max(abs_err))
        denom = np.maximum(1.0, np.maximum(np.abs(grad_analytic), np.abs(grad_numeric)))
        max_rel_err = float(np.max(abs_err / denom))
        return {
            "max_abs_error": max_abs_err,
            "max_rel_error": max_rel_err,
            "min_hinge_gap": min_hinge_gap,
            "passed": bool(max_abs_err < 1e-7),
        }


def query_mixed_knn_neighbors(
    mixed_prep: MixedTypePreprocessor,
    queries: MixedTypeEncoded,
    references: MixedTypeEncoded,
    weights: np.ndarray | None = None,
    max_k: int = 51,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Find top max_k nearest neighbors under weighted Mixed-Type Distance."""
    if mixed_prep.ranges_ is None:
        raise RuntimeError("MixedTypePreprocessor belum di-fit.")

    n_features = len(DOMAIN_FEATURE_COLS)
    if weights is None:
        w = np.full(n_features, 1.0 / n_features, dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
        if w.shape != (n_features,):
            raise ValueError(f"Panjang weights harus {n_features}, diterima {w.shape}.")

    n_num = len(DOMAIN_NUMERIC_COLS)
    n_cat = len(DOMAIN_CATEGORICAL_COLS)
    w_num = w[:n_num]
    w_cat = w[n_num:]

    q_num = queries.numeric.astype(np.float32)
    r_num = references.numeric.astype(np.float32)
    inv_ranges = (1.0 / mixed_prep.ranges_).astype(np.float32)

    q_cat = queries.categorical.astype(np.int16)
    r_cat = references.categorical.astype(np.int16)

    n_q = queries.n_samples
    n_r = references.n_samples
    top_distances = np.empty((n_q, max_k), dtype=np.float64)
    top_indices = np.empty((n_q, max_k), dtype=np.int64)

    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        b_len = end - start
        acc = np.zeros((b_len, n_r), dtype=np.float32)

        for j in range(n_num):
            wj = float(w_num[j])
            if wj <= 0.0:
                continue
            diff = np.abs(q_num[start:end, j : j + 1] - r_num[:, j][None, :])
            diff *= inv_ranges[j]
            np.minimum(diff, 1.0, out=diff)
            acc += wj * diff

        for j in range(n_cat):
            wj = float(w_cat[j])
            if wj <= 0.0:
                continue
            qc = q_cat[start:end, j : j + 1]
            rc = r_cat[:, j][None, :]
            mismatch = (qc != rc) | (qc < 0) | (rc < 0)
            acc += wj * mismatch

        idx_part = np.argpartition(acc, kth=max_k - 1, axis=1)[:, :max_k]
        dist_part = np.take_along_axis(acc, idx_part, axis=1)
        order = np.argsort(dist_part, axis=1)
        top_indices[start:end] = np.take_along_axis(idx_part, order, axis=1)
        top_distances[start:end] = np.take_along_axis(dist_part, order, axis=1).astype(np.float64)

    return top_distances, top_indices


def predict_local_probability(
    y_ref: np.ndarray,
    top_distances: np.ndarray,
    top_indices: np.ndarray,
    k: int,
    q: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Estimate P_hat(yes|x) using distance-decaying local neighbor weights."""
    y_nn = y_ref[top_indices[:, :k]].astype(np.float64)
    if q == 0.0:
        return y_nn.mean(axis=1)
    d_nn = top_distances[:, :k]
    alpha = np.power(d_nn + eps, -q)
    return (alpha * y_nn).sum(axis=1) / alpha.sum(axis=1)


def cost_threshold_from_ratio(cost_fn_to_fp: float) -> float:
    """Compute optimal Bayes cost threshold tau = C_FP / (C_FP + C_FN)."""
    if cost_fn_to_fp <= 0.0:
        raise ValueError("Rasio biaya C_FN / C_FP harus > 0.")
    return float(1.0 / (1.0 + cost_fn_to_fp))


def compute_expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute total and per-sample expected misclassification cost."""
    tau = float(cost_fp / (cost_fp + cost_fn)) if threshold is None else float(threshold)
    y_t = np.asarray(y_true, dtype=np.int64)
    y_pred = (np.asarray(y_prob, dtype=np.float64) >= tau).astype(np.int64)
    fp = int(np.sum((y_t == 0) & (y_pred == 1)))
    fn = int(np.sum((y_t == 1) & (y_pred == 0)))
    total_cost = fp * float(cost_fp) + fn * float(cost_fn)
    avg_cost = total_cost / float(len(y_t))
    return {
        "threshold": tau,
        "fp": fp,
        "fn": fn,
        "total_cost": total_cost,
        "avg_cost_per_sample": avg_cost,
    }


class DIFWKNNClassifier(BaseEstimator, ClassifierMixin):
    """Domain-Informed Feature-Weighted K-Nearest Neighbors classifier."""

    def __init__(
        self,
        k: int = 15,
        neighbor_weights: str = "distance",
        distance_power_q: float = 1.0,
        eps: float = 1e-6,
        learn_feature_weights: bool = True,
        use_cost_sensitive: bool = True,
        c_fp: float = 1.0,
        c_fn: float = 4.0,
        margin: float = 0.2,
        reg_lambda: float = 0.05,
        learning_rate: float = 0.5,
        max_iter: int = 200,
        min_weight: float = 0.0,
        n_triplets: int = 6000,
        positive_anchor_ratio: float = 0.5,
        mining_strategy: str = "semi_hard",
        candidate_pool_size: int = 16,
        batch_size: int = 256,
        random_state: int = 42,
        numeric_cols: Sequence[str] | None = None,
        categorical_cols: Sequence[str] | None = None,
        auto_engineer_pdays: bool = True,
        fixed_feature_weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k harus > 0, diterima {k}.")
        if neighbor_weights not in {"uniform", "distance"}:
            raise ValueError(
                f"neighbor_weights harus 'uniform' atau 'distance', diterima {neighbor_weights!r}."
            )
        if distance_power_q <= 0.0:
            raise ValueError(f"distance_power_q harus > 0, diterima {distance_power_q}.")
        if eps <= 0.0:
            raise ValueError(f"eps harus > 0, diterima {eps}.")

        self.k = int(k)
        self.neighbor_weights = neighbor_weights
        self.distance_power_q = float(distance_power_q)
        self.eps = float(eps)
        self.learn_feature_weights = bool(learn_feature_weights)
        self.use_cost_sensitive = bool(use_cost_sensitive)
        self.c_fp = float(c_fp)
        self.c_fn = float(c_fn)
        self.margin = float(margin)
        self.reg_lambda = float(reg_lambda)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.min_weight = float(min_weight)
        self.n_triplets = int(n_triplets)
        self.positive_anchor_ratio = float(positive_anchor_ratio)
        self.mining_strategy = mining_strategy
        self.candidate_pool_size = int(candidate_pool_size)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.auto_engineer_pdays = auto_engineer_pdays
        self.fixed_feature_weights = fixed_feature_weights

        self.metric_: MixedTypeDistanceMetric | None = None
        self.X_train_enc_: MixedTypeEncoded | None = None
        self.y_train_: np.ndarray | None = None
        self.classes_: np.ndarray = np.array([0, 1], dtype=np.int64)
        self.feature_weights_: np.ndarray | None = None
        self.optimizer_: FeatureWeightOptimizer | None = None
        self.decision_rule_: CostSensitiveDecisionRule | None = None

    def fit(self, X: pd.DataFrame, y: Sequence[int] | np.ndarray) -> DIFWKNNClassifier:
        """Fit mixed-type distance parameters, optional triplet weights, and decision rule."""
        y_arr = np.asarray(y, dtype=np.int64)
        if len(X) != len(y_arr):
            raise ValueError(f"Panjang X ({len(X)}) dan y ({len(y_arr)}) tidak cocok.")
        if self.k > len(X):
            raise ValueError(f"Nilai k={self.k} lebih besar dari jumlah data latih ({len(X)}).")

        metric = MixedTypeDistanceMetric(
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            auto_engineer_pdays=self.auto_engineer_pdays,
        )
        X_enc = metric.fit_transform(X)

        self.metric_ = metric
        self.X_train_enc_ = X_enc
        self.y_train_ = y_arr
        self.classes_ = np.array([0, 1], dtype=np.int64)

        n_features = X_enc.n_features
        if self.fixed_feature_weights is not None:
            self.feature_weights_ = normalize_feature_weights(
                self.fixed_feature_weights, n_features=n_features
            )
            self.optimizer_ = None
        elif self.learn_feature_weights:
            optimizer = FeatureWeightOptimizer(
                margin=self.margin,
                reg_lambda=self.reg_lambda,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                min_weight=self.min_weight,
                n_triplets=self.n_triplets,
                positive_anchor_ratio=self.positive_anchor_ratio,
                mining_strategy=self.mining_strategy,
                candidate_pool_size=self.candidate_pool_size,
                random_state=self.random_state,
            )
            optimizer.fit(X_enc, y_arr, metric)
            self.optimizer_ = optimizer
            assert optimizer.weights_ is not None
            self.feature_weights_ = optimizer.weights_.copy()
        else:
            self.optimizer_ = None
            self.feature_weights_ = np.full(n_features, 1.0 / n_features, dtype=np.float64)

        if self.use_cost_sensitive:
            self.decision_rule_ = CostSensitiveDecisionRule(c_fp=self.c_fp, c_fn=self.c_fn)
        else:
            self.decision_rule_ = CostSensitiveDecisionRule(c_fp=1.0, c_fn=1.0)
        return self

    def kneighbors(
        self,
        X: pd.DataFrame | MixedTypeEncoded,
        n_neighbors: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find top-k nearest training neighbors in memory-bounded query batches."""
        if (
            self.metric_ is None
            or self.X_train_enc_ is None
            or self.feature_weights_ is None
        ):
            raise RuntimeError("DIFWKNNClassifier belum di-fit.")

        k_eff = self.k if n_neighbors is None else int(n_neighbors)
        if k_eff <= 0 or k_eff > self.X_train_enc_.n_samples:
            raise ValueError(
                f"n_neighbors harus berada di [1, {self.X_train_enc_.n_samples}], diterima {k_eff}."
            )

        enc_q = self.metric_.transform(X) if isinstance(X, pd.DataFrame) else X
        n_q = enc_q.n_samples
        n_r = self.X_train_enc_.n_samples
        distances = np.empty((n_q, k_eff), dtype=np.float64)
        indices = np.empty((n_q, k_eff), dtype=np.int64)

        w = normalize_feature_weights(self.feature_weights_, n_features=len(self.metric_.feature_names_))
        n_num = len(self.metric_.numeric_cols_)
        n_cat = len(self.metric_.categorical_cols_)
        w_num = w[:n_num]
        w_cat = w[n_num:]
        w_cat_sum = float(np.sum(w_cat))

        use_blas = (n_cat > 0) and (w_cat_sum > 0.0) and (n_q * n_r >= 4096)
        if use_blas:
            r_oh = self.metric_.build_weighted_categorical_onehot(
                self.X_train_enc_.categorical, w_cat
            ).astype(np.float32)
            q_oh_all = self.metric_.build_weighted_categorical_onehot(
                enc_q.categorical, w_cat
            ).astype(np.float32)
            assert self.metric_.ranges_ is not None
            r_num_s = (self.X_train_enc_.numeric / self.metric_.ranges_[None, :]).astype(np.float32)
            q_num_s = (enc_q.numeric / self.metric_.ranges_[None, :]).astype(np.float32)
            w_num_32 = w_num.astype(np.float32)
            w_cat_sum_32 = np.float32(w_cat_sum)
            diff_buf = np.empty((min(self.batch_size, n_q), n_r), dtype=np.float32)
        else:
            r_oh = None
            q_oh_all = None
            r_num_s = None
            q_num_s = None
            w_num_32 = None
            w_cat_sum_32 = None
            diff_buf = None

        for start in range(0, n_q, self.batch_size):
            end = min(start + self.batch_size, n_q)
            b_len = end - start

            if use_blas and r_oh is not None and q_oh_all is not None and r_num_s is not None and q_num_s is not None and w_num_32 is not None and w_cat_sum_32 is not None and diff_buf is not None:
                buf = diff_buf[:b_len]
                dist_batch = w_cat_sum_32 - (q_oh_all[start:end] @ r_oh.T)
                np.maximum(dist_batch, 0.0, out=dist_batch)
                q_sub = q_num_s[start:end]
                for j in range(n_num):
                    wj = float(w_num_32[j])
                    if wj <= 0.0:
                        continue
                    np.subtract(q_sub[:, j : j + 1], r_num_s[:, j][None, :], out=buf)
                    np.abs(buf, out=buf)
                    np.minimum(buf, 1.0, out=buf)
                    buf *= wj
                    dist_batch += buf
            else:
                q_batch = MixedTypeEncoded(
                    numeric=enc_q.numeric[start:end],
                    categorical=enc_q.categorical[start:end],
                    feature_names=enc_q.feature_names,
                )
                dist_batch = self.metric_.compute_weighted_distance_matrix(
                    q_batch,
                    self.X_train_enc_,
                    weights=w,
                    batch_size=self.batch_size,
                )

            if k_eff == n_r:
                top_idx = np.argsort(dist_batch, axis=1, kind="stable")
                top_dist = np.take_along_axis(dist_batch, top_idx, axis=1)
            else:
                part_idx = np.argpartition(dist_batch, kth=k_eff - 1, axis=1)[:, :k_eff]
                part_dist = np.take_along_axis(dist_batch, part_idx, axis=1)
                order = np.argsort(part_dist, axis=1, kind="stable")
                top_idx = np.take_along_axis(part_idx, order, axis=1)
                top_dist = np.take_along_axis(part_dist, order, axis=1)

            distances[start:end] = top_dist.astype(np.float64)
            indices[start:end] = top_idx

        return distances, indices

    def predict_proba_from_neighbors(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        k: int | None = None,
    ) -> np.ndarray:
        """Compute P_hat(yes | x) from precomputed nearest neighbor distances and indices."""
        if self.y_train_ is None:
            raise RuntimeError("DIFWKNNClassifier belum di-fit.")
        k_eff = self.k if k is None else int(k)
        y_nn = self.y_train_[indices[:, :k_eff]].astype(np.float64)
        if self.neighbor_weights == "uniform":
            return y_nn.mean(axis=1)

        d_nn = distances[:, :k_eff].astype(np.float64)
        alpha = np.power(d_nn + self.eps, -self.distance_power_q)
        return (alpha * y_nn).sum(axis=1) / alpha.sum(axis=1)

    def predict_proba(self, X: pd.DataFrame | MixedTypeEncoded) -> np.ndarray:
        """Estimate class probabilities [P(no | x), P(yes | x)] for queries X."""
        distances, indices = self.kneighbors(X, n_neighbors=self.k)
        p_yes = self.predict_proba_from_neighbors(distances, indices, k=self.k)
        p_yes = np.clip(p_yes, 0.0, 1.0)
        return np.column_stack([1.0 - p_yes, p_yes])

    def predict(self, X: pd.DataFrame | MixedTypeEncoded) -> np.ndarray:
        """Predict binary labels {0, 1} using the configured decision rule."""
        if self.decision_rule_ is None:
            raise RuntimeError("DIFWKNNClassifier belum di-fit.")
        p_yes = self.predict_proba(X)[:, 1]
        return self.decision_rule_.predict(p_yes)

    def evaluate(
        self,
        X: pd.DataFrame | MixedTypeEncoded,
        y: Sequence[int] | np.ndarray,
    ) -> dict[str, Any]:
        """Predict probabilities on X and evaluate classification & cost metrics."""
        if self.decision_rule_ is None:
            raise RuntimeError("DIFWKNNClassifier belum di-fit.")
        p_yes = self.predict_proba(X)[:, 1]
        return self.decision_rule_.evaluate(y, p_yes)

    def get_feature_weights_table(self) -> pd.DataFrame:
        """Return feature weights table sorted by learned/assigned importance."""
        if self.feature_weights_ is None or self.metric_ is None:
            raise RuntimeError("DIFWKNNClassifier belum di-fit.")
        if self.optimizer_ is not None:
            return self.optimizer_.get_weights_table(self.metric_.feature_names_)
        u = np.full(len(self.feature_weights_), 1.0 / len(self.feature_weights_), dtype=np.float64)
        return pd.DataFrame(
            {
                "feature": list(self.metric_.feature_names_),
                "weight": self.feature_weights_,
                "uniform_weight": u,
                "relative_importance": self.feature_weights_ / u,
                "weight_pct": self.feature_weights_ * 100.0,
            }
        ).sort_values("weight", ascending=False).reset_index(drop=True)


def evaluate_ablation_suite(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray | Sequence[int],
    X_test_df: pd.DataFrame,
    y_test: np.ndarray | Sequence[int],
    k: int = 15,
    c_fp: float = 1.0,
    c_fn: float = 4.0,
    n_triplets: int = 6000,
    margin: float = 0.2,
    reg_lambda: float = 0.05,
    random_state: int = 42,
    batch_size: int = 256,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run ablation suite E2, E3, E4, E5 reusing shared distance/neighbor searches."""
    y_tr = np.asarray(y_train, dtype=np.int64)
    y_te = np.asarray(y_test, dtype=np.int64)

    t0 = time.perf_counter()
    model_e2 = DIFWKNNClassifier(
        k=k,
        learn_feature_weights=False,
        use_cost_sensitive=False,
        c_fp=c_fp,
        c_fn=c_fn,
        batch_size=batch_size,
        random_state=random_state,
    )
    model_e2.fit(X_train_df, y_tr)
    fit_e2_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    p_uniform = model_e2.predict_proba(X_test_df)[:, 1]
    pred_e2_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    model_e5 = DIFWKNNClassifier(
        k=k,
        learn_feature_weights=True,
        use_cost_sensitive=True,
        c_fp=c_fp,
        c_fn=c_fn,
        margin=margin,
        reg_lambda=reg_lambda,
        n_triplets=n_triplets,
        batch_size=batch_size,
        random_state=random_state,
    )
    model_e5.fit(X_train_df, y_tr)
    fit_e5_sec = time.perf_counter() - t2

    t3 = time.perf_counter()
    p_weighted = model_e5.predict_proba(X_test_df)[:, 1]
    pred_e5_sec = time.perf_counter() - t3

    sym_rule = CostSensitiveDecisionRule(c_fp=1.0, c_fn=1.0)
    cs_rule = CostSensitiveDecisionRule(c_fp=c_fp, c_fn=c_fn)

    def _eval_with_target_cost(prob: np.ndarray, tau: float) -> dict[str, Any]:
        return cs_rule.evaluate(y_te, prob, threshold=tau)

    rows = [
        {
            "model_id": "E2",
            "model_name": "Mixed-Type KNN (Uniform, tau=0.50)",
            "k": int(k),
            "fit_sec": round(fit_e2_sec, 4),
            "predict_sec": round(pred_e2_sec, 4),
            **_eval_with_target_cost(p_uniform, sym_rule.threshold),
        },
        {
            "model_id": "E3",
            "model_name": "Feature-Weighted KNN (Triplet, tau=0.50)",
            "k": int(k),
            "fit_sec": round(fit_e5_sec, 4),
            "predict_sec": round(pred_e5_sec, 4),
            **_eval_with_target_cost(p_weighted, sym_rule.threshold),
        },
        {
            "model_id": "E4",
            "model_name": f"Cost-Sensitive KNN (Uniform, tau={cs_rule.threshold:.2f})",
            "k": int(k),
            "fit_sec": round(fit_e2_sec, 4),
            "predict_sec": round(pred_e2_sec, 4),
            **_eval_with_target_cost(p_uniform, cs_rule.threshold),
        },
        {
            "model_id": "E5",
            "model_name": f"DIFW-KNN Complete (Triplet + Cost tau={cs_rule.threshold:.2f})",
            "k": int(k),
            "fit_sec": round(fit_e5_sec, 4),
            "predict_sec": round(pred_e5_sec, 4),
            **_eval_with_target_cost(p_weighted, cs_rule.threshold),
        },
    ]

    model_e3 = DIFWKNNClassifier(
        k=k,
        learn_feature_weights=True,
        use_cost_sensitive=False,
        c_fp=c_fp,
        c_fn=c_fn,
        margin=margin,
        reg_lambda=reg_lambda,
        n_triplets=n_triplets,
        batch_size=batch_size,
        random_state=random_state,
    )
    model_e3.metric_ = model_e5.metric_
    model_e3.X_train_enc_ = model_e5.X_train_enc_
    model_e3.y_train_ = model_e5.y_train_
    model_e3.feature_weights_ = model_e5.feature_weights_
    model_e3.optimizer_ = model_e5.optimizer_
    model_e3.decision_rule_ = sym_rule

    model_e4 = DIFWKNNClassifier(
        k=k,
        learn_feature_weights=False,
        use_cost_sensitive=True,
        c_fp=c_fp,
        c_fn=c_fn,
        batch_size=batch_size,
        random_state=random_state,
    )
    model_e4.metric_ = model_e2.metric_
    model_e4.X_train_enc_ = model_e2.X_train_enc_
    model_e4.y_train_ = model_e2.y_train_
    model_e4.feature_weights_ = model_e2.feature_weights_
    model_e4.optimizer_ = None
    model_e4.decision_rule_ = cs_rule

    results_df = pd.DataFrame(rows)
    artifacts = {
        "model_e2": model_e2,
        "model_e3": model_e3,
        "model_e4": model_e4,
        "model_e5": model_e5,
        "y_prob_uniform": p_uniform,
        "y_prob_weighted": p_weighted,
        "weights_table": model_e5.get_feature_weights_table(),
        "loss_history": model_e5.optimizer_.loss_history_ if model_e5.optimizer_ is not None else [],
        "optimizer": model_e5.optimizer_,
    }
    return results_df, artifacts
