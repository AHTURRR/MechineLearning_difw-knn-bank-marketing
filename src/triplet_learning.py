"""Developer 2: Class-Balanced Triplet Sampling & Triplet Margin Loss for DIFW-KNN.

Implements:
  - BalancedTripletSampler: constructs class-balanced triplets (a, p, n) with
    y_a = y_p and y_a != y_n, supporting random and semi-hard mining.
  - triplet_margin_loss_and_grad: computes the regularized Triplet Margin Loss
        L(w) = mean(max(0, m + delta_d^T w)) + lambda * ||w - u||_2^2
    and its exact analytical gradient w.r.t. w.
  - numerical_gradient_check: central-difference verification of the analytical gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.mixed_type_distance import (
    MixedTypeDistanceMetric,
    MixedTypeEncoded,
    normalize_feature_weights,
)


@dataclass(frozen=True)
class TripletBatch:
    """Container storing sampled triplet indices, anchor labels, and delta_d vectors."""

    anchor_idx: np.ndarray
    positive_idx: np.ndarray
    negative_idx: np.ndarray
    anchor_labels: np.ndarray
    delta_feature_distances: np.ndarray

    @property
    def n_triplets(self) -> int:
        return int(len(self.anchor_idx))

    @property
    def n_features(self) -> int:
        return int(self.delta_feature_distances.shape[1])


class BalancedTripletSampler:
    """Construct class-balanced triplets for supervised feature weight learning."""

    def __init__(
        self,
        n_triplets: int = 6000,
        positive_anchor_ratio: float = 0.5,
        mining_strategy: str = "semi_hard",
        candidate_pool_size: int = 16,
        margin: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if n_triplets <= 0:
            raise ValueError(f"n_triplets harus > 0, diterima {n_triplets}.")
        if not (0.0 < positive_anchor_ratio < 1.0):
            raise ValueError(
                f"positive_anchor_ratio harus dalam (0, 1), diterima {positive_anchor_ratio}."
            )
        if mining_strategy not in {"random", "semi_hard", "local_context"}:
            raise ValueError(
                f"mining_strategy harus 'random', 'semi_hard', atau 'local_context', diterima {mining_strategy!r}."
            )
        if candidate_pool_size <= 0:
            raise ValueError(f"candidate_pool_size harus > 0, diterima {candidate_pool_size}.")
        self.n_triplets = int(n_triplets)
        self.positive_anchor_ratio = float(positive_anchor_ratio)
        self.mining_strategy = mining_strategy
        self.candidate_pool_size = int(candidate_pool_size)
        self.margin = float(margin)
        self.random_state = int(random_state)

    @staticmethod
    def _slice_encoded(encoded: MixedTypeEncoded, idx: np.ndarray) -> MixedTypeEncoded:
        return MixedTypeEncoded(
            numeric=encoded.numeric[idx],
            categorical=encoded.categorical[idx],
            feature_names=encoded.feature_names,
        )

    def _sample_anchor_indices(
        self,
        pos_pool: np.ndarray,
        neg_pool: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_pos_anchors = int(round(self.n_triplets * self.positive_anchor_ratio))
        n_neg_anchors = self.n_triplets - n_pos_anchors

        pos_anchors = rng.choice(
            pos_pool,
            size=n_pos_anchors,
            replace=(n_pos_anchors > len(pos_pool)),
        )
        neg_anchors = rng.choice(
            neg_pool,
            size=n_neg_anchors,
            replace=(n_neg_anchors > len(neg_pool)),
        )

        anchors = np.concatenate([pos_anchors, neg_anchors])
        labels = np.concatenate(
            [
                np.ones(n_pos_anchors, dtype=np.int64),
                np.zeros(n_neg_anchors, dtype=np.int64),
            ]
        )
        perm = rng.permutation(self.n_triplets)
        return anchors[perm], labels[perm]

    @staticmethod
    def _sample_excluding_self(
        pool: np.ndarray,
        exclude: np.ndarray,
        size_per_row: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample candidate indices from pool ensuring no element equals exclude[i]."""
        n_rows = len(exclude)
        n_pool = len(pool)
        # Draw positions in [0, n_pool - 1]; shift if colliding with exclude[i]
        cand = rng.choice(pool, size=(n_rows, size_per_row), replace=True)
        collision = cand == exclude[:, None]
        if collision.any():
            # Build lookup from pool value to position for fast collision resolution
            pos_map = {int(val): idx for idx, val in enumerate(pool)}
            row_idx, col_idx = np.where(collision)
            shifts = rng.integers(1, n_pool, size=len(row_idx))
            for r, c, sh in zip(row_idx, col_idx, shifts):
                ex_pos = pos_map[int(exclude[r])]
                cand[r, c] = pool[(ex_pos + int(sh)) % n_pool]
        return cand

    def sample(
        self,
        encoded_X: MixedTypeEncoded,
        y: np.ndarray,
        metric: MixedTypeDistanceMetric,
        mining_weights: Sequence[float] | np.ndarray | None = None,
        batch_size: int = 512,
    ) -> TripletBatch:
        """Sample class-balanced triplets and precompute delta_d = d(a, p) - d(a, n)."""
        y_arr = np.asarray(y, dtype=np.int64)
        if y_arr.shape != (encoded_X.n_samples,):
            raise ValueError(
                f"Panjang y ({len(y_arr)}) tidak cocok dengan jumlah sampel ({encoded_X.n_samples})."
            )
        unique_classes = set(np.unique(y_arr).tolist())
        if unique_classes != {0, 1}:
            raise ValueError(f"Target y harus biner {{0, 1}}, ditemukan {sorted(unique_classes)}.")

        pos_pool = np.where(y_arr == 1)[0]
        neg_pool = np.where(y_arr == 0)[0]
        if len(pos_pool) < 2 or len(neg_pool) < 2:
            raise ValueError("Masing-masing kelas minimal membutuhkan 2 sampel untuk sampling triplet.")

        rng = np.random.default_rng(self.random_state)
        anchor_idx, anchor_labels = self._sample_anchor_indices(pos_pool, neg_pool, rng)

        n_features = encoded_X.n_features
        w_mine = normalize_feature_weights(mining_weights, n_features=n_features)

        positive_idx = np.empty(self.n_triplets, dtype=np.int64)
        negative_idx = np.empty(self.n_triplets, dtype=np.int64)

        is_pos_anchor = anchor_labels == 1
        n_pos = int(is_pos_anchor.sum())
        n_neg = self.n_triplets - n_pos

        if self.mining_strategy == "random":
            positive_idx[is_pos_anchor] = self._sample_excluding_self(
                pos_pool, anchor_idx[is_pos_anchor], size_per_row=1, rng=rng
            )[:, 0]
            positive_idx[~is_pos_anchor] = self._sample_excluding_self(
                neg_pool, anchor_idx[~is_pos_anchor], size_per_row=1, rng=rng
            )[:, 0]
            negative_idx[is_pos_anchor] = rng.choice(neg_pool, size=n_pos, replace=True)
            negative_idx[~is_pos_anchor] = rng.choice(pos_pool, size=n_neg, replace=True)

            delta_d = np.empty((self.n_triplets, n_features), dtype=np.float64)
            for start in range(0, self.n_triplets, batch_size):
                end = min(start + batch_size, self.n_triplets)
                enc_a = self._slice_encoded(encoded_X, anchor_idx[start:end])
                enc_p = self._slice_encoded(encoded_X, positive_idx[start:end])
                enc_n = self._slice_encoded(encoded_X, negative_idx[start:end])
                d_ap = metric.compute_aligned_feature_distances(enc_a, enc_p)
                d_an = metric.compute_aligned_feature_distances(enc_a, enc_n)
                delta_d[start:end] = d_ap - d_an

            return TripletBatch(
                anchor_idx=anchor_idx,
                positive_idx=positive_idx,
                negative_idx=negative_idx,
                anchor_labels=anchor_labels,
                delta_feature_distances=delta_d,
            )

        if self.mining_strategy == "local_context":
            from sklearn.neighbors import NearestNeighbors

            # Hybrid 50% Global Class-Balanced + 50% Local-Context Target-Neighbor Triplets:
            # Preserves multimodal intra-class clusters (e.g., campaign month & prior outcome)
            # while maintaining global inter-class margin across all 16 domain features.
            n_glob = self.n_triplets // 2
            n_loc = self.n_triplets - n_glob

            # Build normalized context embedding strictly from fitted metric parameters
            emb_parts: list[np.ndarray] = []
            if metric.numeric_cols_ and metric.q05_ is not None and metric.ranges_ is not None:
                num_clipped = np.clip(
                    (encoded_X.numeric - metric.q05_[None, :]) / metric.ranges_[None, :],
                    0.0,
                    1.0,
                ).astype(np.float32)
                emb_parts.append(num_clipped)

            if metric.categorical_cols_:
                preferred = [c for c in ("month", "poutcome") if c in metric.categorical_cols_]
                ctx_cols = preferred if preferred else [
                    c for c in metric.categorical_cols_ if len(metric.category_maps_[c]) >= 3
                ]
                if not ctx_cols:
                    ctx_cols = list(metric.categorical_cols_)
                for col in ctx_cols:
                    j_cat = metric.categorical_cols_.index(col)
                    n_c = len(metric.category_maps_[col])
                    eye = np.eye(n_c, dtype=np.float32) * 0.5
                    vals = encoded_X.categorical[:, j_cat]
                    oh = eye[np.maximum(vals, 0)]
                    if np.any(vals < 0):
                        oh[vals < 0] = 0.0
                    emb_parts.append(oh)

            ctx_emb = (
                np.hstack(emb_parts)
                if emb_parts
                else np.zeros((encoded_X.n_samples, 1), dtype=np.float32)
            )

            # Part A: Global balanced triplets (indices 0 .. n_glob - 1)
            is_pos_g = is_pos_anchor[:n_glob]
            n_pos_g = int(is_pos_g.sum())
            n_neg_g = n_glob - n_pos_g
            if n_pos_g > 0:
                positive_idx[:n_glob][is_pos_g] = self._sample_excluding_self(
                    pos_pool, anchor_idx[:n_glob][is_pos_g], size_per_row=1, rng=rng
                )[:, 0]
                negative_idx[:n_glob][is_pos_g] = rng.choice(neg_pool, size=n_pos_g, replace=True)
            if n_neg_g > 0:
                positive_idx[:n_glob][~is_pos_g] = self._sample_excluding_self(
                    neg_pool, anchor_idx[:n_glob][~is_pos_g], size_per_row=1, rng=rng
                )[:, 0]
                negative_idx[:n_glob][~is_pos_g] = rng.choice(pos_pool, size=n_neg_g, replace=True)

            # Part B: Local-context target-neighbor triplets (indices n_glob .. n_triplets - 1)
            if n_loc > 0:
                k_pos_nn = min(max(2, self.candidate_pool_size), len(pos_pool))
                k_neg_nn = min(max(2, self.candidate_pool_size), len(neg_pool))
                nn_pos = NearestNeighbors(n_neighbors=k_pos_nn, metric="manhattan", n_jobs=-1).fit(
                    ctx_emb[pos_pool]
                )
                nn_neg = NearestNeighbors(n_neighbors=k_neg_nn, metric="manhattan", n_jobs=-1).fit(
                    ctx_emb[neg_pool]
                )

                anch_loc = anchor_idx[n_glob:]
                is_pos_l = is_pos_anchor[n_glob:]
                pos_loc_out = np.empty(n_loc, dtype=np.int64)
                neg_loc_out = np.empty(n_loc, dtype=np.int64)

                n_pos_l = int(is_pos_l.sum())
                n_neg_l = n_loc - n_pos_l
                if n_pos_l > 0:
                    a_pl = anch_loc[is_pos_l]
                    _, nn_idx_p = nn_pos.kneighbors(ctx_emb[a_pl], n_neighbors=k_pos_nn)
                    cand_global = pos_pool[nn_idx_p]
                    # Pick random local neighbor rank >= 1 (excluding self)
                    ranks = rng.integers(1, k_pos_nn, size=n_pos_l)
                    chosen_p = cand_global[np.arange(n_pos_l), ranks]
                    self_hit = chosen_p == a_pl
                    if np.any(self_hit):
                        chosen_p[self_hit] = self._sample_excluding_self(
                            pos_pool, a_pl[self_hit], size_per_row=1, rng=rng
                        )[:, 0]
                    pos_loc_out[is_pos_l] = chosen_p
                    neg_loc_out[is_pos_l] = rng.choice(neg_pool, size=n_pos_l, replace=True)

                if n_neg_l > 0:
                    a_nl = anch_loc[~is_pos_l]
                    _, nn_idx_n = nn_neg.kneighbors(ctx_emb[a_nl], n_neighbors=k_neg_nn)
                    cand_global_n = neg_pool[nn_idx_n]
                    ranks_n = rng.integers(1, k_neg_nn, size=n_neg_l)
                    chosen_n_same = cand_global_n[np.arange(n_neg_l), ranks_n]
                    self_hit_n = chosen_n_same == a_nl
                    if np.any(self_hit_n):
                        chosen_n_same[self_hit_n] = self._sample_excluding_self(
                            neg_pool, a_nl[self_hit_n], size_per_row=1, rng=rng
                        )[:, 0]
                    pos_loc_out[~is_pos_l] = chosen_n_same
                    neg_loc_out[~is_pos_l] = rng.choice(pos_pool, size=n_neg_l, replace=True)

                positive_idx[n_glob:] = pos_loc_out
                negative_idx[n_glob:] = neg_loc_out

            delta_d = np.empty((self.n_triplets, n_features), dtype=np.float64)
            for start in range(0, self.n_triplets, batch_size):
                end = min(start + batch_size, self.n_triplets)
                enc_a = self._slice_encoded(encoded_X, anchor_idx[start:end])
                enc_p = self._slice_encoded(encoded_X, positive_idx[start:end])
                enc_n = self._slice_encoded(encoded_X, negative_idx[start:end])
                d_ap = metric.compute_aligned_feature_distances(enc_a, enc_p)
                d_an = metric.compute_aligned_feature_distances(enc_a, enc_n)
                delta_d[start:end] = d_ap - d_an

            return TripletBatch(
                anchor_idx=anchor_idx,
                positive_idx=positive_idx,
                negative_idx=negative_idx,
                anchor_labels=anchor_labels,
                delta_feature_distances=delta_d,
            )

        # Semi-hard mining over random candidate pools
        pool_k = self.candidate_pool_size
        pos_cands = np.empty((self.n_triplets, pool_k), dtype=np.int64)
        neg_cands = np.empty((self.n_triplets, pool_k), dtype=np.int64)

        pos_cands[is_pos_anchor] = self._sample_excluding_self(
            pos_pool, anchor_idx[is_pos_anchor], size_per_row=pool_k, rng=rng
        )
        pos_cands[~is_pos_anchor] = self._sample_excluding_self(
            neg_pool, anchor_idx[~is_pos_anchor], size_per_row=pool_k, rng=rng
        )
        neg_cands[is_pos_anchor] = rng.choice(neg_pool, size=(n_pos, pool_k), replace=True)
        neg_cands[~is_pos_anchor] = rng.choice(pos_pool, size=(n_neg, pool_k), replace=True)

        delta_d = np.empty((self.n_triplets, n_features), dtype=np.float64)
        for start in range(0, self.n_triplets, batch_size):
            end = min(start + batch_size, self.n_triplets)
            bs = end - start

            a_rep = np.repeat(anchor_idx[start:end], pool_k)
            p_flat = pos_cands[start:end].ravel()
            n_flat = neg_cands[start:end].ravel()

            enc_a = self._slice_encoded(encoded_X, a_rep)
            enc_p = self._slice_encoded(encoded_X, p_flat)
            enc_n = self._slice_encoded(encoded_X, n_flat)

            d_ap_feat = metric.compute_aligned_feature_distances(enc_a, enc_p).reshape(
                bs, pool_k, n_features
            )
            d_an_feat = metric.compute_aligned_feature_distances(enc_a, enc_n).reshape(
                bs, pool_k, n_features
            )

            d_ap = d_ap_feat @ w_mine
            d_an = d_an_feat @ w_mine

            # Choose a random or close-to-median positive from candidate pool to preserve local clusters
            p_choice = np.argsort(d_ap, axis=1)[:, min(pool_k // 4, pool_k - 1)]
            row_ar = np.arange(bs)
            chosen_ap_dist = d_ap[row_ar, p_choice][:, None]

            # Semi-hard negative: D(a, p) < D(a, n) < D(a, p) + margin (closest to D(a, p))
            semi_hard_mask = (d_an > chosen_ap_dist) & (d_an < chosen_ap_dist + self.margin)
            beyond_mask = d_an > chosen_ap_dist

            n_choice = np.empty(bs, dtype=np.int64)
            for i in range(bs):
                if semi_hard_mask[i].any():
                    masked = np.where(semi_hard_mask[i], d_an[i], np.inf)
                    n_choice[i] = int(np.argmin(masked))
                elif beyond_mask[i].any():
                    masked = np.where(beyond_mask[i], d_an[i], np.inf)
                    n_choice[i] = int(np.argmin(masked))
                else:
                    # All negatives closer than positive: pick negative closest to positive boundary
                    n_choice[i] = int(np.argmax(d_an[i]))

            positive_idx[start:end] = pos_cands[start:end][row_ar, p_choice]
            negative_idx[start:end] = neg_cands[start:end][row_ar, n_choice]
            delta_d[start:end] = d_ap_feat[row_ar, p_choice] - d_an_feat[row_ar, n_choice]

        return TripletBatch(
            anchor_idx=anchor_idx,
            positive_idx=positive_idx,
            negative_idx=negative_idx,
            anchor_labels=anchor_labels,
            delta_feature_distances=delta_d,
        )


def triplet_margin_loss_and_grad(
    weights: np.ndarray | Sequence[float],
    delta_d: np.ndarray,
    margin: float = 0.2,
    reg_lambda: float = 0.05,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Compute regularized Triplet Margin Loss and its exact analytical gradient.

    L(w) = mean(max(0, m + delta_d @ w)) + reg_lambda * ||w - u||_2^2
    where u = (1/p, ..., 1/p).
    """
    w = np.asarray(weights, dtype=np.float64)
    dd = np.asarray(delta_d, dtype=np.float64)
    if dd.ndim != 2 or dd.shape[0] == 0:
        raise ValueError(f"delta_d harus berdimensi (n_triplets, n_features), diterima {dd.shape}.")
    n_triplets, n_features = dd.shape
    if w.shape != (n_features,):
        raise ValueError(f"Panjang weights harus ({n_features},), diterima {w.shape}.")
    if margin < 0.0:
        raise ValueError(f"margin harus >= 0, diterima {margin}.")
    if reg_lambda < 0.0:
        raise ValueError(f"reg_lambda harus >= 0, diterima {reg_lambda}.")

    u = np.full(n_features, 1.0 / n_features, dtype=np.float64)
    raw_hinge = margin + (dd @ w)
    active_mask = raw_hinge > 0.0

    hinge_loss = float(np.where(active_mask, raw_hinge, 0.0).mean())
    diff_u = w - u
    reg_loss = float(reg_lambda * np.dot(diff_u, diff_u))
    total_loss = hinge_loss + reg_loss

    if active_mask.any():
        grad_hinge = dd[active_mask].sum(axis=0) / float(n_triplets)
    else:
        grad_hinge = np.zeros(n_features, dtype=np.float64)
    grad = grad_hinge + 2.0 * reg_lambda * diff_u

    stats = {
        "total_loss": total_loss,
        "hinge_loss": hinge_loss,
        "reg_loss": reg_loss,
        "active_fraction": float(active_mask.mean()),
        "n_active": int(active_mask.sum()),
    }
    return total_loss, grad, stats


def numerical_gradient_check(
    weights: np.ndarray | Sequence[float],
    delta_d: np.ndarray,
    margin: float = 0.2,
    reg_lambda: float = 0.05,
    eps: float = 1e-6,
    tol: float = 1e-4,
) -> dict[str, Any]:
    """Verify analytical gradient against central finite differences."""
    w = np.asarray(weights, dtype=np.float64).copy()
    _, analytical_grad, _ = triplet_margin_loss_and_grad(
        w, delta_d, margin=margin, reg_lambda=reg_lambda
    )

    numerical_grad = np.zeros_like(w)
    for j in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[j] += eps
        w_minus[j] -= eps
        loss_plus, _, _ = triplet_margin_loss_and_grad(
            w_plus, delta_d, margin=margin, reg_lambda=reg_lambda
        )
        loss_minus, _, _ = triplet_margin_loss_and_grad(
            w_minus, delta_d, margin=margin, reg_lambda=reg_lambda
        )
        numerical_grad[j] = (loss_plus - loss_minus) / (2.0 * eps)

    abs_diff = np.abs(analytical_grad - numerical_grad)
    max_abs_err = float(np.max(abs_diff))
    denom = np.maximum(1.0, np.abs(analytical_grad) + np.abs(numerical_grad))
    max_rel_err = float(np.max(abs_diff / denom))

    return {
        "analytical_grad": analytical_grad,
        "numerical_grad": numerical_grad,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "passed": bool(max_abs_err <= tol),
    }
