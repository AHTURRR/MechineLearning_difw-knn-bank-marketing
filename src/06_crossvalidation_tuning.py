"""Stage 06: Joint Stratified 5-Fold Cross-Validation & Hyperparameter Tuning.

Corresponds to the 'Cross-Validation & Hyperparameter Tuning' convergence node on
the research flowchart, receiving:
  - Left branch : Baseline KNN (StandardScaler + OneHot + Euclidean)
  - Right branch: Proposed DIFW-KNN (Mixed-Type Distance + Local-Context Triplet
                  Weights + Local Probability + Cost-Sensitive Decision Rule)

Strictly operates EXCLUSIVELY on Training / Development Data (36,168 observations)
using the canonical 5-fold assignments in data/processed/train_cv_folds.csv.
Hold-out Test Data (9,043 observations) remains 100% untouched until Stage 07.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_knn import (
    METRIC_KEYS,
    compute_classification_metrics,
    run_stratified_cv_grid,
)
from src.cost_sensitive_decision import CostSensitiveDecisionRule, compute_optimal_threshold
from src.feature_weight_optimizer import FeatureWeightOptimizer
from src.mixed_type_distance import MixedTypeDistanceMetric, MixedTypeEncoded, normalize_feature_weights
from src.triplet_learning import BalancedTripletSampler


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cv_tuning"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "Laporan_CrossValidation_Tuning_DIFW_KNN.md"

RANDOM_STATE = 42

# Branch A: Baseline Euclidean KNN grid (20 configs)
BASELINE_K_GRID = [3, 5, 7, 11, 15, 21, 31, 51, 71, 101]
BASELINE_WEIGHTS_GRID = ["uniform", "distance"]

# Branch B1 & B2: Mixed-Type Uniform (E2/E4) & Proposed DIFW-KNN (E3/E5) grids
DIFW_MARGIN_GRID = [0.30, 0.40, 0.50]
DIFW_LAMBDA_GRID = [0.08, 0.15, 0.25]
DIFW_K_GRID = [15, 21, 31, 51, 71]
DIFW_Q_GRID = [0.0, 0.5, 1.0, 2.0]  # q=0.0 corresponds to uniform neighbor voting
N_TRIPLETS_CV = 24000
CANDIDATE_POOL_SIZE = 25
MAX_ITER_OPT = 180

PRIMARY_C_FP = 1.0
PRIMARY_C_FN = 5.0
COST_RATIO_CANDIDATES = [1.0, 2.0, 3.0, 4.0, 5.0, 7.55]


def configure_plot_style() -> None:
    """Configure clean matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
        }
    )


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render DataFrame as GitHub-flavored Markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _fast_top_k_mixed(
    metric: MixedTypeDistanceMetric,
    enc_q: MixedTypeEncoded,
    enc_r: MixedTypeEncoded,
    q_num_scaled: np.ndarray,
    r_num_scaled: np.ndarray,
    weights: np.ndarray,
    max_k: int,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """L3-cache-friendly exact Mixed-Type top-k search using preallocated buffers & SGEMM."""
    w = normalize_feature_weights(weights, n_features=len(metric.feature_names_))
    n_num = len(metric.numeric_cols_)
    n_cat = len(metric.categorical_cols_)
    w_num = w[:n_num].astype(np.float32)
    w_cat = w[n_num:]
    w_cat_sum = np.float32(np.sum(w_cat))

    r_oh = metric.build_weighted_categorical_onehot(enc_r.categorical, w_cat).astype(np.float32)
    q_oh_all = metric.build_weighted_categorical_onehot(enc_q.categorical, w_cat).astype(np.float32)

    n_q = enc_q.n_samples
    n_r = enc_r.n_samples
    top_distances = np.empty((n_q, max_k), dtype=np.float64)
    top_indices = np.empty((n_q, max_k), dtype=np.int64)

    diff_buf = np.empty((batch_size, n_r), dtype=np.float32)

    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        b_len = end - start
        buf = diff_buf[:b_len]

        # Categorical weighted mismatch via OpenBLAS SGEMM
        acc = w_cat_sum - (q_oh_all[start:end] @ r_oh.T)
        np.maximum(acc, 0.0, out=acc)

        # Numeric clipped Q95-Q05 distance using preallocated buffer (zero heap thrashing)
        q_sub = q_num_scaled[start:end]
        for j in range(n_num):
            wj = float(w_num[j])
            if wj <= 0.0:
                continue
            np.subtract(q_sub[:, j : j + 1], r_num_scaled[:, j][None, :], out=buf)
            np.abs(buf, out=buf)
            np.minimum(buf, 1.0, out=buf)
            buf *= wj
            acc += buf

        part_idx = np.argpartition(acc, kth=max_k - 1, axis=1)[:, :max_k]
        part_dist = np.take_along_axis(acc, part_idx, axis=1)
        order = np.argsort(part_dist, axis=1, kind="stable")
        top_indices[start:end] = np.take_along_axis(part_idx, order, axis=1)
        top_distances[start:end] = np.take_along_axis(part_dist, order, axis=1).astype(np.float64)

    return top_distances, top_indices


def _predict_probs_from_topk(
    y_ref: np.ndarray,
    top_distances: np.ndarray,
    top_indices: np.ndarray,
    k: int,
    q: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute local positive probability P_hat(yes | x) for given (k, q)."""
    y_nn = y_ref[top_indices[:, :k]].astype(np.float64)
    if q == 0.0:
        return y_nn.mean(axis=1)
    alpha = np.power(top_distances[:, :k] + eps, -q)
    return (alpha * y_nn).sum(axis=1) / alpha.sum(axis=1)


def run_mixed_and_difw_cv_tuning(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run leak-free 5-fold CV grid search for Mixed-Type Uniform (E2/E4) & DIFW-KNN (E3/E5)."""
    unique_folds = sorted(int(f) for f in np.unique(fold_ids))
    max_k = max(DIFW_K_GRID)
    tau_cs = compute_optimal_threshold(c_fp=PRIMARY_C_FP, c_fn=PRIMARY_C_FN)
    cs_rule = CostSensitiveDecisionRule(c_fp=PRIMARY_C_FP, c_fn=PRIMARY_C_FN)

    # Pre-fit fold encoders and sample fold triplets strictly from training fold
    fold_data: list[dict[str, Any]] = []
    for fid in unique_folds:
        val_mask = fold_ids == fid
        tr_mask = ~val_mask
        X_tr = X_train_df.loc[tr_mask].reset_index(drop=True)
        y_tr = y_train[tr_mask]
        X_va = X_train_df.loc[val_mask].reset_index(drop=True)
        y_va = y_train[val_mask]

        metric = MixedTypeDistanceMetric().fit(X_tr)
        enc_tr = metric.transform(X_tr)
        enc_va = metric.transform(X_va)
        assert metric.ranges_ is not None
        r_num_s = (enc_tr.numeric / metric.ranges_[None, :]).astype(np.float32)
        q_num_s = (enc_va.numeric / metric.ranges_[None, :]).astype(np.float32)

        sampler = BalancedTripletSampler(
            n_triplets=N_TRIPLETS_CV,
            positive_anchor_ratio=0.5,
            mining_strategy="local_context",
            candidate_pool_size=CANDIDATE_POOL_SIZE,
            margin=0.40,
            random_state=RANDOM_STATE + fid,
        )
        triplet_batch = sampler.sample(enc_tr, y_tr, metric)

        fold_data.append(
            {
                "fold_id": fid,
                "val_idx": np.where(val_mask)[0],
                "metric": metric,
                "enc_tr": enc_tr,
                "enc_va": enc_va,
                "r_num_s": r_num_s,
                "q_num_s": q_num_s,
                "y_tr": y_tr,
                "y_va": y_va,
                "triplet_batch": triplet_batch,
            }
        )

    # 1. Mixed-Type Uniform Grid (E2 & E4): 20 configs (5 k x 4 q)
    u_weights = np.full(16, 1.0 / 16.0, dtype=np.float64)
    uniform_oof: dict[tuple[int, float], np.ndarray] = {
        (k, q): np.zeros(len(y_train), dtype=np.float64)
        for k in DIFW_K_GRID
        for q in DIFW_Q_GRID
    }
    u_rows: list[dict[str, Any]] = []

    for fd in fold_data:
        t0 = time.perf_counter()
        top_d, top_i = _fast_top_k_mixed(
            fd["metric"],
            fd["enc_va"],
            fd["enc_tr"],
            fd["q_num_s"],
            fd["r_num_s"],
            weights=u_weights,
            max_k=max_k,
        )
        q_sec = time.perf_counter() - t0

        for k in DIFW_K_GRID:
            for q in DIFW_Q_GRID:
                prob = _predict_probs_from_topk(fd["y_tr"], top_d, top_i, k=k, q=q)
                uniform_oof[(k, q)][fd["val_idx"]] = prob
                m_e2 = compute_classification_metrics(fd["y_va"], prob, threshold=0.5)
                m_e4 = cs_rule.evaluate(fd["y_va"], prob, threshold=tau_cs)
                u_rows.append(
                    {
                        "fold_id": fd["fold_id"],
                        "k": k,
                        "q": q,
                        "query_sec": round(q_sec, 4),
                        "pr_auc": m_e2["pr_auc"],
                        "roc_auc": m_e2["roc_auc"],
                        "e2_f1": m_e2["f1_yes"],
                        "e2_mcc": m_e2["mcc"],
                        "e2_recall": m_e2["recall_yes"],
                        "e2_precision": m_e2["precision_yes"],
                        "e2_accuracy": m_e2["accuracy"],
                        "e2_gmean": m_e2["g_mean"],
                        "e4_f1": m_e4["f1_yes"],
                        "e4_mcc": m_e4["mcc"],
                        "e4_recall": m_e4["recall_yes"],
                        "e4_precision": m_e4["precision_yes"],
                        "e4_accuracy": m_e4["accuracy"],
                        "e4_gmean": m_e4["g_mean"],
                        "e4_mean_cost": m_e4["mean_cost"],
                    }
                )

    u_fold_df = pd.DataFrame(u_rows)
    u_metric_cols = [c for c in u_fold_df.columns if c not in {"fold_id", "k", "q", "query_sec"}]
    uniform_cv_df = (
        u_fold_df.groupby(["k", "q"], as_index=False)
        .agg(
            **{f"{c}_mean": (c, "mean") for c in u_metric_cols},
            **{f"{c}_std": (c, "std") for c in u_metric_cols},
        )
        .sort_values(["pr_auc_mean", "e4_f1_mean", "e2_mcc_mean"], ascending=False)
        .reset_index(drop=True)
    )

    # 2. Proposed DIFW-KNN Grid (E3 & E5): 3 margin x 3 lambda x 5 k x 4 q = 180 configs
    difw_oof: dict[tuple[float, float, int, float], np.ndarray] = {}
    fold_weights_store: dict[tuple[float, float], list[np.ndarray]] = {}
    w_rows: list[dict[str, Any]] = []

    for margin in DIFW_MARGIN_GRID:
        for reg_lam in DIFW_LAMBDA_GRID:
            for k in DIFW_K_GRID:
                for q in DIFW_Q_GRID:
                    difw_oof[(margin, reg_lam, k, q)] = np.zeros(len(y_train), dtype=np.float64)
            w_per_fold: list[np.ndarray] = []

            for fd in fold_data:
                t_opt0 = time.perf_counter()
                opt = FeatureWeightOptimizer(
                    margin=margin,
                    reg_lambda=reg_lam,
                    learning_rate=0.5,
                    max_iter=MAX_ITER_OPT,
                    n_triplets=N_TRIPLETS_CV,
                    mining_strategy="local_context",
                    candidate_pool_size=CANDIDATE_POOL_SIZE,
                    random_state=RANDOM_STATE + fd["fold_id"],
                ).fit(fd["enc_tr"], fd["y_tr"], fd["metric"], triplet_batch=fd["triplet_batch"])
                opt_sec = time.perf_counter() - t_opt0
                assert opt.weights_ is not None
                w_per_fold.append(opt.weights_.copy())

                t_q0 = time.perf_counter()
                top_d, top_i = _fast_top_k_mixed(
                    fd["metric"],
                    fd["enc_va"],
                    fd["enc_tr"],
                    fd["q_num_s"],
                    fd["r_num_s"],
                    weights=opt.weights_,
                    max_k=max_k,
                )
                q_sec = time.perf_counter() - t_q0

                for k in DIFW_K_GRID:
                    for q in DIFW_Q_GRID:
                        prob = _predict_probs_from_topk(fd["y_tr"], top_d, top_i, k=k, q=q)
                        difw_oof[(margin, reg_lam, k, q)][fd["val_idx"]] = prob
                        m_e3 = compute_classification_metrics(fd["y_va"], prob, threshold=0.5)
                        m_e5 = cs_rule.evaluate(fd["y_va"], prob, threshold=tau_cs)
                        w_rows.append(
                            {
                                "fold_id": fd["fold_id"],
                                "margin": margin,
                                "reg_lambda": reg_lam,
                                "k": k,
                                "q": q,
                                "opt_sec": round(opt_sec, 4),
                                "query_sec": round(q_sec, 4),
                                "pr_auc": m_e3["pr_auc"],
                                "roc_auc": m_e3["roc_auc"],
                                "e3_f1": m_e3["f1_yes"],
                                "e3_mcc": m_e3["mcc"],
                                "e3_recall": m_e3["recall_yes"],
                                "e3_precision": m_e3["precision_yes"],
                                "e3_accuracy": m_e3["accuracy"],
                                "e3_gmean": m_e3["g_mean"],
                                "e5_f1": m_e5["f1_yes"],
                                "e5_mcc": m_e5["mcc"],
                                "e5_recall": m_e5["recall_yes"],
                                "e5_precision": m_e5["precision_yes"],
                                "e5_accuracy": m_e5["accuracy"],
                                "e5_gmean": m_e5["g_mean"],
                                "e5_mean_cost": m_e5["mean_cost"],
                            }
                        )
            fold_weights_store[(margin, reg_lam)] = w_per_fold

    w_fold_df = pd.DataFrame(w_rows)
    w_metric_cols = [
        c for c in w_fold_df.columns if c not in {"fold_id", "margin", "reg_lambda", "k", "q", "opt_sec", "query_sec"}
    ]
    difw_cv_df = (
        w_fold_df.groupby(["margin", "reg_lambda", "k", "q"], as_index=False)
        .agg(
            **{f"{c}_mean": (c, "mean") for c in w_metric_cols},
            **{f"{c}_std": (c, "std") for c in w_metric_cols},
        )
        .sort_values(["pr_auc_mean", "e5_mcc_mean", "e5_f1_mean"], ascending=False)
        .reset_index(drop=True)
    )

    best_u = uniform_cv_df.iloc[0]
    best_w = difw_cv_df.iloc[0]
    artifacts = {
        "best_uniform_oof": uniform_oof[(int(best_u["k"]), float(best_u["q"]))],
        "best_difw_oof": difw_oof[
            (float(best_w["margin"]), float(best_w["reg_lambda"]), int(best_w["k"]), float(best_w["q"]))
        ],
        "best_cv_weights_mean": np.mean(
            np.vstack(fold_weights_store[(float(best_w["margin"]), float(best_w["reg_lambda"]))]),
            axis=0,
        ),
        "best_cv_weights_std": np.std(
            np.vstack(fold_weights_store[(float(best_w["margin"]), float(best_w["reg_lambda"]))]),
            axis=0,
            ddof=1,
        ),
        "feature_names": list(fold_data[0]["metric"].feature_names_),
    }
    return uniform_cv_df, difw_cv_df, artifacts


def create_cv_tuning_figures(
    baseline_cv_df: pd.DataFrame,
    uniform_cv_df: pd.DataFrame,
    difw_cv_df: pd.DataFrame,
    best_summary_df: pd.DataFrame,
) -> list[Path]:
    """Generate 3 publication-quality figures for Stage 06 CV & Hyperparameter Tuning."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Baseline KNN vs Mixed-Type Uniform vs Proposed DIFW-KNN PR-AUC & F1 across k
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
    base_d = baseline_cv_df[baseline_cv_df["weights"] == "distance"].sort_values("k")
    base_u = baseline_cv_df[baseline_cv_df["weights"] == "uniform"].sort_values("k")
    axes[0].plot(base_u["k"], base_u["pr_auc_mean"], marker="o", linestyle="--", color="#7F7F7F", label="Baseline Euclidean (uniform)")
    axes[0].plot(base_d["k"], base_d["pr_auc_mean"], marker="s", linestyle="--", color="#4C72B0", label="Baseline Euclidean (distance)")

    best_m = float(difw_cv_df.iloc[0]["margin"])
    best_l = float(difw_cv_df.iloc[0]["reg_lambda"])
    best_q_u = float(uniform_cv_df.iloc[0]["q"])
    best_q_w = float(difw_cv_df.iloc[0]["q"])

    unif_sub = uniform_cv_df[uniform_cv_df["q"] == best_q_u].sort_values("k")
    difw_sub = difw_cv_df[
        (difw_cv_df["margin"] == best_m) & (difw_cv_df["reg_lambda"] == best_l) & (difw_cv_df["q"] == best_q_w)
    ].sort_values("k")

    axes[0].plot(unif_sub["k"], unif_sub["pr_auc_mean"], marker="^", linewidth=2, color="#DD8452", label=f"Mixed-Type Uniform (q={best_q_u:g})")
    axes[0].plot(difw_sub["k"], difw_sub["pr_auc_mean"], marker="D", linewidth=2.3, color="#55A868", label=f"Proposed DIFW-KNN (m={best_m:g}, lam={best_l:g}, q={best_q_w:g})")
    axes[0].set_title("5-Fold CV PR-AUC vs Jumlah Tetangga (k)")
    axes[0].set_xlabel("Nilai k")
    axes[0].set_ylabel("5-Fold CV PR-AUC")
    axes[0].legend(fontsize=8)

    axes[1].plot(base_u["k"], base_u["f1_yes_mean"], marker="o", linestyle="--", color="#7F7F7F", label="E0/E1 Euclidean (tau=0.50)")
    axes[1].plot(unif_sub["k"], unif_sub["e2_f1_mean"], marker="^", linestyle="--", color="#DD8452", label="E2 Mixed Uniform (tau=0.50)")
    axes[1].plot(difw_sub["k"], difw_sub["e3_f1_mean"], marker="s", linewidth=1.9, color="#4C72B0", label="E3 DIFW Weighted (tau=0.50)")
    axes[1].plot(unif_sub["k"], unif_sub["e4_f1_mean"], marker="v", linewidth=2.0, color="#8172B3", label="E4 Cost-Sensitive Mixed (5:1)")
    axes[1].plot(difw_sub["k"], difw_sub["e5_f1_mean"], marker="D", linewidth=2.3, color="#55A868", label="E5 Full DIFW-KNN (5:1)")
    axes[1].set_title("5-Fold CV F1-Score (yes) vs Jumlah Tetangga (k)")
    axes[1].set_xlabel("Nilai k")
    axes[1].set_ylabel("5-Fold CV F1 (yes)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    p1 = FIGURES_DIR / "01_cv_k_curves_baseline_vs_difw.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # 2. Heatmap of (margin, reg_lambda) on PR-AUC and E5 F1 at optimal (k, q)
    best_k_w = int(difw_cv_df.iloc[0]["k"])
    hm_sub = difw_cv_df[(difw_cv_df["k"] == best_k_w) & (difw_cv_df["q"] == best_q_w)]
    piv_pr = hm_sub.pivot(index="margin", columns="reg_lambda", values="pr_auc_mean")
    piv_f1 = hm_sub.pivot(index="margin", columns="reg_lambda", values="e5_f1_mean")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, piv, title in [
        (axes[0], piv_pr, f"5-Fold CV PR-AUC (k={best_k_w}, q={best_q_w:g})"),
        (axes[1], piv_f1, f"5-Fold CV F1-Score E5 @ 5:1 (k={best_k_w}, q={best_q_w:g})"),
    ]:
        arr = piv.to_numpy()
        im = ax.imshow(arr, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{r:g}" for r in piv.index])
        ax.set_xlabel("Regularisasi L2 (lambda)")
        ax.set_ylabel("Triplet Margin (m)")
        ax.set_title(title)
        ax.grid(False)
        mid = (arr.min() + arr.max()) / 2.0
        for r_i in range(arr.shape[0]):
            for c_i in range(arr.shape[1]):
                ax.text(
                    c_i,
                    r_i,
                    f"{arr[r_i, c_i]:.4f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white" if arr[r_i, c_i] > mid else "#222222",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    p2 = FIGURES_DIR / "02_difw_margin_lambda_heatmaps.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    # 3. Tuned 5-Fold CV Winner Summary across E0 .. E5
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9))
    m_ids = best_summary_df["model_id"].tolist()
    x = np.arange(len(m_ids))
    wb = 0.25

    axes[0].bar(x - wb, best_summary_df["f1_yes_mean"], width=wb, color="#4C72B0", label="F1 (yes)")
    axes[0].bar(x, best_summary_df["mcc_mean"], width=wb, color="#55A868", label="MCC")
    axes[0].bar(x + wb, best_summary_df["g_mean_mean"], width=wb, color="#DD8452", label="G-Mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(m_ids)
    axes[0].set_ylim(0, 0.84)
    axes[0].set_title("Perbandingan Model Ter-Tuning pada 5-Fold CV (E0 s.d. E5)")
    axes[0].set_ylabel("Skor Rata-Rata 5-Fold CV")
    axes[0].legend(fontsize=8.5)
    for idx_m, row in best_summary_df.iterrows():
        axes[0].text(idx_m - wb, row["f1_yes_mean"] + 0.012, f"{row['f1_yes_mean']:.3f}", ha="center", fontsize=7.5, rotation=90)
        axes[0].text(idx_m, row["mcc_mean"] + 0.012, f"{row['mcc_mean']:.3f}", ha="center", fontsize=7.5, rotation=90)
        axes[0].text(idx_m + wb, row["g_mean_mean"] + 0.012, f"{row['g_mean_mean']:.3f}", ha="center", fontsize=7.5, rotation=90)

    axes[1].bar(x - 0.18, best_summary_df["pr_auc_mean"], width=0.36, color="#8172B3", label="PR-AUC")
    axes[1].bar(x + 0.18, best_summary_df["roc_auc_mean"], width=0.36, color="#CCB974", label="ROC-AUC")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(m_ids)
    axes[1].set_ylim(0, 0.90)
    axes[1].set_title("Kualitas Ranking Ter-Tuning pada 5-Fold CV (PR-AUC & ROC-AUC)")
    axes[1].set_ylabel("Skor Rata-Rata 5-Fold CV")
    axes[1].legend(fontsize=8.5)
    for idx_m, row in best_summary_df.iterrows():
        axes[1].text(idx_m - 0.18, row["pr_auc_mean"] + 0.012, f"{row['pr_auc_mean']:.4f}", ha="center", fontsize=8)
        axes[1].text(idx_m + 0.18, row["roc_auc_mean"] + 0.012, f"{row['roc_auc_mean']:.4f}", ha="center", fontsize=8)

    fig.tight_layout()
    p3 = FIGURES_DIR / "03_tuned_cv_winners_e0_to_e5.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved.append(p3)

    return saved


def build_report(
    baseline_cv_df: pd.DataFrame,
    uniform_cv_df: pd.DataFrame,
    difw_cv_df: pd.DataFrame,
    best_summary_df: pd.DataFrame,
    cv_weights_df: pd.DataFrame,
    cost_sensitivity_df: pd.DataFrame,
    selected_configs: dict[str, Any],
    figure_paths: list[Path],
) -> str:
    """Build Laporan_CrossValidation_Tuning_DIFW_KNN.md."""
    base_top = baseline_cv_df.sort_values(["f1_yes_mean", "mcc_mean"], ascending=False).head(6)[
        ["weights", "k", "accuracy_mean", "precision_yes_mean", "recall_yes_mean", "f1_yes_mean", "pr_auc_mean", "roc_auc_mean", "mcc_mean"]
    ].round(4)
    unif_top = uniform_cv_df.head(6)[
        ["k", "q", "pr_auc_mean", "roc_auc_mean", "e2_f1_mean", "e2_mcc_mean", "e4_f1_mean", "e4_mcc_mean", "e4_gmean_mean"]
    ].round(4)
    difw_top = difw_cv_df.head(10)[
        ["margin", "reg_lambda", "k", "q", "pr_auc_mean", "roc_auc_mean", "e3_f1_mean", "e3_mcc_mean", "e5_f1_mean", "e5_mcc_mean", "e5_gmean_mean"]
    ].round(4)

    lines = [
        "# LAPORAN TAHAP 06 — CROSS-VALIDATION & HYPERPARAMETER TUNING",
        "",
        "**Skrip CLI:** [`src/06_crossvalidation_tuning.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/06_crossvalidation_tuning.py)  ",
        "**Notebook:** [`notebooks/06_CrossValidation_Hyperparameter_Tuning.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/06_CrossValidation_Hyperparameter_Tuning.ipynb)  ",
        "**Kepatuhan Alur Kerja (*Flowchart*):** Tahap ini mempertemukan cabang **Baseline KNN** (Tahap 04) dan **Proposed DIFW-KNN** (Tahap 05) pada **Stratified 5-Fold Cross-Validation (`36.168` baris `Training / Development Data`)**. `Hold-out Test Data` (`9.043` baris) tetap **dikunci sepenuhnya** dan baru akan dibuka pada **Tahap 07 (`Final Model Evaluation`)**.",
        "",
        "---",
        "",
        "## 1. Ruang Pencarian Hyperparameter (*Search Space*)",
        "",
        "- **Cabang A (`Baseline KNN` — `E1`):** `k` $\\in \\{3, 5, 7, 11, 15, 21, 31, 51, 71, 101\\}$ × `weights` $\\in \\{\\text{'uniform'}, \\text{'distance'}\\}$ (`20` konfigurasi × `5` fold = `100` evaluasi).",
        "- **Cabang B1 (`Mixed-Type Uniform` — `E2` & `E4`):** `k` $\\in \\{15, 21, 31, 51, 71\\}$ × eksponen jarak lokal `q` $\\in \\{0.0, 0.5, 1.0, 2.0\\}$ (`20` konfigurasi × `5` fold = `100` evaluasi).",
        "- **Cabang B2 (`Proposed DIFW-KNN` — `E3` & `E5`):** `margin` $m \\in \\{0.30, 0.40, 0.50\\}$ × `reg_lambda` $\\lambda \\in \\{0.08, 0.15, 0.25\\}$ × `k` $\\in \\{15, 21, 31, 51, 71\\}$ × `q` $\\in \\{0.0, 0.5, 1.0, 2.0\\}$ (`180` konfigurasi × `5` fold = `900` evaluasi).",
        "",
        "---",
        "",
        "## 2. Konfigurasi Hyperparameter Terpilih untuk Tahap 07 (`Final Model Evaluation`)",
        "",
        f"- **`E0 (Standard KNN)`:** `k = 5, weights = 'uniform', threshold = 0.50` *(konfigurasi dasar tanpa tuning)*.",
        f"- **`E1 (Optimized KNN — Best CV F1)`:** `k = {selected_configs['E1']['k']}, weights = '{selected_configs['E1']['weights']}', threshold = 0.50` *(catatan: kandidat Best CV PR-AUC Euclidean adalah `k = {selected_configs['E1_PR']['k']}, weights = '{selected_configs['E1_PR']['weights']}'`)*.",
        f"- **`E2 (Mixed-Type KNN)`:** `k = {selected_configs['E2']['k']}, q = {selected_configs['E2']['q']}, threshold = 0.50`.",
        f"- **`E3 (Feature-Weighted KNN)`:** `margin = {selected_configs['E3']['margin']}, reg_lambda = {selected_configs['E3']['reg_lambda']}, k = {selected_configs['E3']['k']}, q = {selected_configs['E3']['q']}, threshold = 0.50`.",
        f"- **`E4 (Cost-Sensitive Mixed KNN)`:** `k = {selected_configs['E4']['k']}, q = {selected_configs['E4']['q']}, c_fp = {PRIMARY_C_FP}, c_fn = {PRIMARY_C_FN} (threshold = {selected_configs['E4']['threshold']:.4f})`.",
        f"- **`E5 (DIFW-KNN Lengkap)`:** `margin = {selected_configs['E5']['margin']}, reg_lambda = {selected_configs['E5']['reg_lambda']}, k = {selected_configs['E5']['k']}, q = {selected_configs['E5']['q']}, c_fp = {PRIMARY_C_FP}, c_fn = {PRIMARY_C_FN} (threshold = {selected_configs['E5']['threshold']:.4f})`.",
        "",
        "### Ringkasan Performa 5-Fold Cross-Validation Model Terpilih (`E0` s.d. `E5`)",
        "",
        df_to_markdown_table(best_summary_df.round(4)),
        "",
        "---",
        "",
        "## 3. Rincian Hasil Grid Search per Cabang",
        "",
        "### a. Top 6 Konfigurasi Baseline Euclidean KNN (`E1`)",
        "",
        df_to_markdown_table(base_top),
        "",
        "### b. Top 6 Konfigurasi Mixed-Type Uniform (`E2` & `E4`)",
        "",
        df_to_markdown_table(unif_top),
        "",
        "### c. Top 10 Konfigurasi Proposed DIFW-KNN (`E3` & `E5`)",
        "",
        df_to_markdown_table(difw_top),
        "",
        "---",
        "",
        "## 4. Konsistensi Bobot Fitur Optimal Lintas 5 Fold CV",
        "",
        df_to_markdown_table(cv_weights_df.round(5)),
        "",
        "---",
        "",
        "## 5. Sensitivitas Rasio Biaya pada Prediksi Out-of-Fold Model Terpilih (`E5`)",
        "",
        df_to_markdown_table(cost_sensitivity_df.round(4)),
        "",
        "---",
        "",
        "## 6. Artefak Visual",
        "",
    ]
    for p in figure_paths:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        lines.append(f"- [{p.name}](file:///{p.as_posix()}) (`{rel}`)")
    return "\n".join(lines)


def main() -> None:
    """Execute Stage 06 joint 5-fold CV hyperparameter tuning."""
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "train_split.csv")
    folds_df = pd.read_csv(PROCESSED_DIR / "train_cv_folds.csv")
    X_train_df = train_df.drop(columns=["y"])
    y_train = train_df["y"].to_numpy(dtype=np.int64)
    fold_ids = folds_df["fold_id"].to_numpy(dtype=np.int64)

    # 1. Branch A: Baseline Euclidean KNN 5-fold CV grid
    base_fold_df, baseline_cv_df = run_stratified_cv_grid(
        X_train_df=X_train_df,
        y_train=y_train,
        fold_ids=fold_ids,
        k_grid=BASELINE_K_GRID,
        weights_grid=BASELINE_WEIGHTS_GRID,
    )
    base_fold_df.to_csv(OUTPUT_DIR / "baseline_cv_per_fold.csv", index=False)
    baseline_cv_df.to_csv(OUTPUT_DIR / "baseline_cv_grid_results.csv", index=False)

    # 2. Branch B1 & B2: Mixed-Type Uniform & Proposed DIFW-KNN 5-fold CV grids
    uniform_cv_df, difw_cv_df, artifacts = run_mixed_and_difw_cv_tuning(
        X_train_df=X_train_df,
        y_train=y_train,
        fold_ids=fold_ids,
    )
    uniform_cv_df.to_csv(OUTPUT_DIR / "mixed_uniform_cv_grid_results.csv", index=False)
    difw_cv_df.to_csv(OUTPUT_DIR / "difw_knn_cv_grid_results.csv", index=False)

    # 3. Extract optimal hyperparameters strictly from 5-Fold CV
    e0_cv = baseline_cv_df[(baseline_cv_df["k"] == 5) & (baseline_cv_df["weights"] == "uniform")].iloc[0]
    e1_f1_cv = baseline_cv_df.sort_values(["f1_yes_mean", "mcc_mean", "pr_auc_mean"], ascending=False).iloc[0]
    e1_pr_cv = baseline_cv_df.sort_values(["pr_auc_mean", "roc_auc_mean", "f1_yes_mean"], ascending=False).iloc[0]
    best_u = uniform_cv_df.iloc[0]
    best_w = difw_cv_df.iloc[0]

    tau_cs = compute_optimal_threshold(c_fp=PRIMARY_C_FP, c_fn=PRIMARY_C_FN)
    selected_configs = {
        "E0": {"k": 5, "weights": "uniform", "threshold": 0.50},
        "E1": {"k": int(e1_f1_cv["k"]), "weights": str(e1_f1_cv["weights"]), "threshold": 0.50},
        "E1_PR": {"k": int(e1_pr_cv["k"]), "weights": str(e1_pr_cv["weights"]), "threshold": 0.50},
        "E2": {
            "k": int(best_u["k"]),
            "q": float(best_u["q"]),
            "neighbor_weights": "uniform" if float(best_u["q"]) == 0.0 else "distance",
            "threshold": 0.50,
        },
        "E3": {
            "margin": float(best_w["margin"]),
            "reg_lambda": float(best_w["reg_lambda"]),
            "k": int(best_w["k"]),
            "q": float(best_w["q"]),
            "neighbor_weights": "uniform" if float(best_w["q"]) == 0.0 else "distance",
            "mining_strategy": "local_context",
            "n_triplets": N_TRIPLETS_CV,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "max_iter": MAX_ITER_OPT,
            "threshold": 0.50,
        },
        "E4": {
            "k": int(best_u["k"]),
            "q": float(best_u["q"]),
            "neighbor_weights": "uniform" if float(best_u["q"]) == 0.0 else "distance",
            "c_fp": PRIMARY_C_FP,
            "c_fn": PRIMARY_C_FN,
            "threshold": round(tau_cs, 6),
        },
        "E5": {
            "margin": float(best_w["margin"]),
            "reg_lambda": float(best_w["reg_lambda"]),
            "k": int(best_w["k"]),
            "q": float(best_w["q"]),
            "neighbor_weights": "uniform" if float(best_w["q"]) == 0.0 else "distance",
            "mining_strategy": "local_context",
            "n_triplets": N_TRIPLETS_CV,
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "max_iter": MAX_ITER_OPT,
            "c_fp": PRIMARY_C_FP,
            "c_fn": PRIMARY_C_FN,
            "threshold": round(tau_cs, 6),
        },
    }
    (OUTPUT_DIR / "selected_hyperparameters.json").write_text(
        json.dumps(selected_configs, indent=2),
        encoding="utf-8",
    )

    # Summary table across E0 .. E5 on 5-Fold CV
    best_summary_df = pd.DataFrame(
        [
            {
                "model_id": "E0",
                "config_summary": "Euclidean k=5, uniform, tau=0.50",
                "accuracy_mean": e0_cv["accuracy_mean"],
                "precision_yes_mean": e0_cv["precision_yes_mean"],
                "recall_yes_mean": e0_cv["recall_yes_mean"],
                "f1_yes_mean": e0_cv["f1_yes_mean"],
                "pr_auc_mean": e0_cv["pr_auc_mean"],
                "roc_auc_mean": e0_cv["roc_auc_mean"],
                "mcc_mean": e0_cv["mcc_mean"],
                "g_mean_mean": e0_cv["g_mean_mean"],
            },
            {
                "model_id": "E1",
                "config_summary": f"Euclidean k={int(e1_f1_cv['k'])}, {e1_f1_cv['weights']}, tau=0.50",
                "accuracy_mean": e1_f1_cv["accuracy_mean"],
                "precision_yes_mean": e1_f1_cv["precision_yes_mean"],
                "recall_yes_mean": e1_f1_cv["recall_yes_mean"],
                "f1_yes_mean": e1_f1_cv["f1_yes_mean"],
                "pr_auc_mean": e1_f1_cv["pr_auc_mean"],
                "roc_auc_mean": e1_f1_cv["roc_auc_mean"],
                "mcc_mean": e1_f1_cv["mcc_mean"],
                "g_mean_mean": e1_f1_cv["g_mean_mean"],
            },
            {
                "model_id": "E2",
                "config_summary": f"Mixed Uniform k={int(best_u['k'])}, q={best_u['q']:g}, tau=0.50",
                "accuracy_mean": best_u["e2_accuracy_mean"],
                "precision_yes_mean": best_u["e2_precision_mean"],
                "recall_yes_mean": best_u["e2_recall_mean"],
                "f1_yes_mean": best_u["e2_f1_mean"],
                "pr_auc_mean": best_u["pr_auc_mean"],
                "roc_auc_mean": best_u["roc_auc_mean"],
                "mcc_mean": best_u["e2_mcc_mean"],
                "g_mean_mean": best_u["e2_gmean_mean"],
            },
            {
                "model_id": "E3",
                "config_summary": f"DIFW m={best_w['margin']:g}, lam={best_w['reg_lambda']:g}, k={int(best_w['k'])}, q={best_w['q']:g}, tau=0.50",
                "accuracy_mean": best_w["e3_accuracy_mean"],
                "precision_yes_mean": best_w["e3_precision_mean"],
                "recall_yes_mean": best_w["e3_recall_mean"],
                "f1_yes_mean": best_w["e3_f1_mean"],
                "pr_auc_mean": best_w["pr_auc_mean"],
                "roc_auc_mean": best_w["roc_auc_mean"],
                "mcc_mean": best_w["e3_mcc_mean"],
                "g_mean_mean": best_w["e3_gmean_mean"],
            },
            {
                "model_id": "E4",
                "config_summary": f"Mixed Cost k={int(best_u['k'])}, q={best_u['q']:g}, 5:1 (tau={tau_cs:.4f})",
                "accuracy_mean": best_u["e4_accuracy_mean"],
                "precision_yes_mean": best_u["e4_precision_mean"],
                "recall_yes_mean": best_u["e4_recall_mean"],
                "f1_yes_mean": best_u["e4_f1_mean"],
                "pr_auc_mean": best_u["pr_auc_mean"],
                "roc_auc_mean": best_u["roc_auc_mean"],
                "mcc_mean": best_u["e4_mcc_mean"],
                "g_mean_mean": best_u["e4_gmean_mean"],
            },
            {
                "model_id": "E5",
                "config_summary": f"DIFW Lengkap m={best_w['margin']:g}, lam={best_w['reg_lambda']:g}, k={int(best_w['k'])}, q={best_w['q']:g}, 5:1",
                "accuracy_mean": best_w["e5_accuracy_mean"],
                "precision_yes_mean": best_w["e5_precision_mean"],
                "recall_yes_mean": best_w["e5_recall_mean"],
                "f1_yes_mean": best_w["e5_f1_mean"],
                "pr_auc_mean": best_w["pr_auc_mean"],
                "roc_auc_mean": best_w["roc_auc_mean"],
                "mcc_mean": best_w["e5_mcc_mean"],
                "g_mean_mean": best_w["e5_gmean_mean"],
            },
        ]
    )
    best_summary_df.to_csv(OUTPUT_DIR / "cv_selected_models_summary.csv", index=False)

    cv_weights_df = pd.DataFrame(
        {
            "feature": artifacts["feature_names"],
            "cv_weight_mean": artifacts["best_cv_weights_mean"],
            "cv_weight_std": artifacts["best_cv_weights_std"],
            "uniform_prior": 1.0 / len(artifacts["feature_names"]),
            "relative_importance": artifacts["best_cv_weights_mean"] * len(artifacts["feature_names"]),
        }
    ).sort_values("cv_weight_mean", ascending=False).reset_index(drop=True)
    cv_weights_df.to_csv(OUTPUT_DIR / "cv_optimal_feature_weights.csv", index=False)

    cost_rows = []
    for c_fn in COST_RATIO_CANDIDATES:
        rule = CostSensitiveDecisionRule(c_fp=PRIMARY_C_FP, c_fn=c_fn)
        ev = rule.evaluate(y_train, artifacts["best_difw_oof"])
        cost_rows.append(
            {
                "c_fn_ratio": c_fn,
                "threshold": round(ev["threshold"], 4),
                "accuracy": ev["accuracy"],
                "precision_yes": ev["precision_yes"],
                "recall_yes": ev["recall_yes"],
                "f1_yes": ev["f1_yes"],
                "mcc": ev["mcc"],
                "g_mean": ev["g_mean"],
                "mean_cost": ev["mean_cost"],
                "cost_reduction_pct": ev["cost_reduction_pct"],
            }
        )
    cost_sensitivity_df = pd.DataFrame(cost_rows)
    cost_sensitivity_df.to_csv(OUTPUT_DIR / "cv_oof_cost_sensitivity.csv", index=False)

    figure_paths = create_cv_tuning_figures(
        baseline_cv_df=baseline_cv_df,
        uniform_cv_df=uniform_cv_df,
        difw_cv_df=difw_cv_df,
        best_summary_df=best_summary_df,
    )

    report = build_report(
        baseline_cv_df=baseline_cv_df,
        uniform_cv_df=uniform_cv_df,
        difw_cv_df=difw_cv_df,
        best_summary_df=best_summary_df,
        cv_weights_df=cv_weights_df,
        cost_sensitivity_df=cost_sensitivity_df,
        selected_configs=selected_configs,
        figure_paths=figure_paths,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Tahap 06 (Cross-Validation & Hyperparameter Tuning) selesai.")
    print(best_summary_df[["model_id", "config_summary", "f1_yes_mean", "pr_auc_mean", "roc_auc_mean", "mcc_mean", "g_mean_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
