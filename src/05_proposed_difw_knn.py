"""Stage 05: Proposed DIFW-KNN Algorithm Development on Training / Development Data.

Maps 1-to-1 to the 6 sub-stages inside 'Proposed Algorithm Development' on the
research flowchart, operating EXCLUSIVELY on Training / Development Data (36,168 rows;
Hold-out Test Data remains untouched until Stage 07):
  1. Mixed-Type Distance (Q05-Q95 range scaling + mismatch categorical + OpenBLAS check)
  2. Balanced Triplet Sampling (Hybrid Global + Local-Context 50:50 class-balanced)
  3. Triplet Margin Loss (analytic vs central finite-difference gradient check)
  4. Supervised Feature Weight Learning (Projected GD on simplex + 5-seed stability)
  5. Weighted KNN & Local Probability (5-fold leak-free OOF evaluation: E2_dev vs E3_dev)
  6. Cost-Sensitive Decision Rule (Bayes optimal tau* & cost ratio grid: E4_dev vs E5_dev)
"""

from __future__ import annotations

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

from src.baseline_knn import METRIC_KEYS, compute_classification_metrics
from src.cost_sensitive_decision import CostSensitiveDecisionRule, evaluate_cost_ratio_grid
from src.difw_knn import DIFWKNNClassifier
from src.feature_weight_optimizer import FeatureWeightOptimizer, project_onto_simplex
from src.mixed_type_distance import MixedTypeDistanceMetric, MixedTypeEncoded
from src.triplet_learning import BalancedTripletSampler, numerical_gradient_check


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BASELINE_DIR = PROJECT_ROOT / "outputs" / "baseline_knn"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "proposed_difw_knn"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "Laporan_Proposed_DIFW_KNN.md"

RANDOM_STATE = 42
STABILITY_SEEDS = [42, 100, 2026, 7, 999]
DEV_N_TRIPLETS = 30000
DEV_MARGIN = 0.40
DEV_LAMBDA = 0.15
DEV_POOL_SIZE = 25
DEV_K = 31
DEV_Q = 1.0
DEV_C_FP = 1.0
DEV_C_FN = 5.0


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


def run_component_validation_suite(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureWeightOptimizer]:
    """Validate all 6 algorithmic sub-components and 5-seed stability on Training Data."""
    metric = MixedTypeDistanceMetric().fit(X_train_df)
    X_enc = metric.transform(X_train_df)

    rng = np.random.default_rng(RANDOM_STATE)
    idx_sub = rng.choice(X_enc.n_samples, size=160, replace=False)
    sub_enc = MixedTypeEncoded(
        numeric=X_enc.numeric[idx_sub],
        categorical=X_enc.categorical[idx_sub],
        feature_names=X_enc.feature_names,
    )
    d_pair = metric.compute_pairwise_feature_distances(sub_enc, sub_enc)

    # Check 1-3: Distance metric axioms
    diag_max = float(np.max(np.abs(np.diagonal(d_pair, axis1=0, axis2=1))))
    sym_max = float(np.max(np.abs(d_pair - np.transpose(d_pair, (1, 0, 2)))))
    d_min, d_max = float(d_pair.min()), float(d_pair.max())

    # Check 4: OpenBLAS categorical acceleration vs exact pairwise mean
    u = np.full(X_enc.n_features, 1.0 / X_enc.n_features, dtype=np.float64)
    dw_exact = (d_pair * u[None, None, :]).sum(axis=2)
    dw_blas = metric.compute_weighted_distance_matrix(sub_enc, sub_enc, weights=u, batch_size=64)
    blas_diff = float(np.max(np.abs(dw_exact - dw_blas)))

    # Check 5: Class-Balanced Local-Context Triplet Sampling
    sampler = BalancedTripletSampler(
        n_triplets=DEV_N_TRIPLETS,
        positive_anchor_ratio=0.5,
        mining_strategy="local_context",
        candidate_pool_size=DEV_POOL_SIZE,
        margin=DEV_MARGIN,
        random_state=RANDOM_STATE,
    )
    batch = sampler.sample(X_enc, y_train, metric)
    pos_ratio = float(np.mean(batch.anchor_labels == 1))
    valid_pos = bool(np.all(y_train[batch.anchor_idx] == y_train[batch.positive_idx]))
    no_self = bool(np.all(batch.anchor_idx != batch.positive_idx))
    valid_neg = bool(np.all(y_train[batch.anchor_idx] != batch.negative_idx))

    # Check 6: Analytical vs Central Finite-Difference Gradient Check
    probe_w = project_onto_simplex(rng.uniform(0.4, 1.6, size=X_enc.n_features))
    margins = np.abs(DEV_MARGIN + batch.delta_feature_distances @ probe_w)
    eps_gc = 1e-6 if float(np.min(margins)) > 5e-6 else float(np.min(margins)) * 0.2
    gc_res = numerical_gradient_check(
        probe_w,
        batch.delta_feature_distances,
        margin=DEV_MARGIN,
        reg_lambda=DEV_LAMBDA,
        eps=eps_gc,
        tol=1e-7,
    )

    # Check 7-8: Simplex constraints & Monotonic Triplet Margin Loss reduction
    opt = FeatureWeightOptimizer(
        margin=DEV_MARGIN,
        reg_lambda=DEV_LAMBDA,
        learning_rate=0.5,
        max_iter=200,
        n_triplets=DEV_N_TRIPLETS,
        mining_strategy="local_context",
        candidate_pool_size=DEV_POOL_SIZE,
        random_state=RANDOM_STATE,
    ).fit(X_enc, y_train, metric, triplet_batch=batch)
    assert opt.weights_ is not None and opt.initial_loss_ is not None and opt.final_loss_ is not None
    w_min = float(opt.weights_.min())
    w_sum = float(opt.weights_.sum())

    # Check 9: Multi-seed feature weight stability across 5 seeds
    stability_df, stab_stats = opt.evaluate_seed_stability(
        encoded_X=X_enc,
        y=y_train,
        metric=metric,
        seeds=STABILITY_SEEDS,
    )

    checks = [
        {"id": 1, "sub_stage": "Mixed-Type Distance", "check": "Self-distance d_j(x, x) = 0", "value": f"max_diag={diag_max:.1e}", "status": "PASS" if diag_max == 0.0 else "FAIL"},
        {"id": 2, "sub_stage": "Mixed-Type Distance", "check": "Simetri d_j(x, z) = d_j(z, x)", "value": f"max_diff={sym_max:.1e}", "status": "PASS" if sym_max <= 1e-12 else "FAIL"},
        {"id": 3, "sub_stage": "Mixed-Type Distance", "check": "Rentang d_j(x, z) dalam [0, 1]", "value": f"[{d_min:.4f}, {d_max:.4f}]", "status": "PASS" if (0.0 <= d_min and d_max <= 1.0) else "FAIL"},
        {"id": 4, "sub_stage": "Mixed-Type Distance", "check": "Ekuivalensi OpenBLAS vs exact loop", "value": f"max_diff={blas_diff:.2e}", "status": "PASS" if blas_diff <= 1e-12 else "FAIL"},
        {"id": 5, "sub_stage": "Balanced Triplet Sampling", "check": "Proporsi anchor 50:50 & label valid", "value": f"pos_ratio={pos_ratio:.3f}, valid={valid_pos and no_self and valid_neg}", "status": "PASS" if (abs(pos_ratio - 0.5) < 1e-6 and valid_pos and no_self and valid_neg) else "FAIL"},
        {"id": 6, "sub_stage": "Triplet Margin Loss", "check": "Gradient check analitik vs numerik (<1e-7)", "value": f"max_abs_err={gc_res['max_abs_err']:.2e}", "status": "PASS" if gc_res["passed"] else "FAIL"},
        {"id": 7, "sub_stage": "Feature Weight Learning", "check": "Kendala simpleks (w_j >= 0, sum = 1)", "value": f"min={w_min:.5f}, sum={w_sum:.12f}", "status": "PASS" if (w_min >= 0.0 and abs(w_sum - 1.0) <= 1e-9) else "FAIL"},
        {"id": 8, "sub_stage": "Feature Weight Learning", "check": "Penurunan Triplet Margin Loss", "value": f"{opt.initial_loss_:.5f} -> {opt.final_loss_:.5f}", "status": "PASS" if opt.final_loss_ < opt.initial_loss_ else "FAIL"},
        {"id": 9, "sub_stage": "Feature Weight Learning", "check": "Stabilitas bobot lintas 5 random seed", "value": f"Spearman={stab_stats['mean_pairwise_spearman']:.4f}, Cosine={stab_stats['mean_pairwise_cosine']:.4f}", "status": "PASS" if stab_stats["mean_pairwise_spearman"] >= 0.90 else "FAIL"},
    ]
    checks_df = pd.DataFrame(checks)
    if (checks_df["status"] != "PASS").any():
        failed = checks_df.loc[checks_df["status"] != "PASS", "check"].tolist()
        raise AssertionError(f"Validasi komponen gagal pada: {failed}")

    weights_df = opt.get_weights_table(metric.feature_names_)
    return checks_df, weights_df, stability_df, opt


def evaluate_proposed_on_dev_folds(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate E2_dev, E3_dev, E4_dev, E5_dev across 5 development folds (leak-free)."""
    oof_uniform = np.zeros(len(y_train), dtype=np.float64)
    oof_weighted = np.zeros(len(y_train), dtype=np.float64)
    fold_weights: list[np.ndarray] = []

    sym_rule = CostSensitiveDecisionRule(c_fp=1.0, c_fn=1.0)
    cs_rule = CostSensitiveDecisionRule(c_fp=DEV_C_FP, c_fn=DEV_C_FN)
    per_fold_rows: list[dict[str, Any]] = []

    for fold_id in sorted(int(f) for f in np.unique(fold_ids)):
        val_mask = fold_ids == fold_id
        tr_mask = ~val_mask

        X_tr = X_train_df.loc[tr_mask].reset_index(drop=True)
        y_tr = y_train[tr_mask]
        X_va = X_train_df.loc[val_mask].reset_index(drop=True)
        y_va = y_train[val_mask]

        # Fit E2/E4 (Mixed-Type Uniform) strictly on fold training split
        t0 = time.perf_counter()
        clf_u = DIFWKNNClassifier(
            k=DEV_K,
            neighbor_weights="distance",
            distance_power_q=DEV_Q,
            learn_feature_weights=False,
            use_cost_sensitive=False,
            c_fp=DEV_C_FP,
            c_fn=DEV_C_FN,
            batch_size=1024,
            random_state=RANDOM_STATE + fold_id,
        ).fit(X_tr, y_tr)
        p_u = clf_u.predict_proba(X_va)[:, 1]
        u_sec = time.perf_counter() - t0
        oof_uniform[val_mask] = p_u

        # Fit E3/E5 (Proposed DIFW-KNN) strictly on fold training split
        t1 = time.perf_counter()
        clf_w = DIFWKNNClassifier(
            k=DEV_K,
            neighbor_weights="distance",
            distance_power_q=DEV_Q,
            learn_feature_weights=True,
            use_cost_sensitive=True,
            c_fp=DEV_C_FP,
            c_fn=DEV_C_FN,
            margin=DEV_MARGIN,
            reg_lambda=DEV_LAMBDA,
            n_triplets=DEV_N_TRIPLETS,
            mining_strategy="local_context",
            candidate_pool_size=DEV_POOL_SIZE,
            batch_size=1024,
            random_state=RANDOM_STATE + fold_id,
        ).fit(X_tr, y_tr)
        p_w = clf_w.predict_proba(X_va)[:, 1]
        w_sec = time.perf_counter() - t1
        oof_weighted[val_mask] = p_w
        assert clf_w.feature_weights_ is not None
        fold_weights.append(clf_w.feature_weights_.copy())

        for model_id, model_label, prob, tau, sec in [
            ("E2_dev", f"Mixed-Type KNN (k={DEV_K}, uniform, tau=0.50)", p_u, sym_rule.threshold, u_sec),
            ("E3_dev", f"Feature-Weighted KNN (k={DEV_K}, triplet, tau=0.50)", p_w, sym_rule.threshold, w_sec),
            ("E4_dev", f"Cost-Sensitive Mixed KNN (k={DEV_K}, uniform, tau={cs_rule.threshold:.4f})", p_u, cs_rule.threshold, u_sec),
            ("E5_dev", f"Proposed DIFW-KNN (k={DEV_K}, triplet + tau={cs_rule.threshold:.4f})", p_w, cs_rule.threshold, w_sec),
        ]:
            ev = cs_rule.evaluate(y_va, prob, threshold=tau)
            per_fold_rows.append(
                {
                    "model_id": model_id,
                    "model_label": model_label,
                    "fold_id": fold_id,
                    "elapsed_sec": round(sec, 4),
                    **ev,
                }
            )

    per_fold_df = pd.DataFrame(per_fold_rows)
    summary_rows: list[dict[str, Any]] = []
    for model_id in ["E2_dev", "E3_dev", "E4_dev", "E5_dev"]:
        sub = per_fold_df[per_fold_df["model_id"] == model_id]
        row: dict[str, Any] = {
            "model_id": model_id,
            "model_label": str(sub["model_label"].iloc[0]),
            "threshold": round(float(sub["threshold"].iloc[0]), 4),
        }
        for m in list(METRIC_KEYS) + ["mean_cost"]:
            row[f"{m}_mean"] = round(float(sub[m].mean()), 4)
            row[f"{m}_std"] = round(float(sub[m].std(ddof=1)), 4)
        summary_rows.append(row)

    oof_dict = {
        "oof_uniform": oof_uniform,
        "oof_weighted": oof_weighted,
        "fold_mean_weights": np.mean(np.vstack(fold_weights), axis=0),
    }
    return per_fold_df, pd.DataFrame(summary_rows), oof_dict


def create_proposed_figures(
    opt: FeatureWeightOptimizer,
    weights_df: pd.DataFrame,
    dev_progression_df: pd.DataFrame,
    cost_grid_df: pd.DataFrame,
) -> list[Path]:
    """Generate 3 publication-quality figures for Stage 05."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Triplet Margin Loss Convergence & Learned Feature Weights
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    iters = np.arange(len(opt.loss_history_))
    axes[0].plot(iters, opt.loss_history_, color="#4C72B0", linewidth=2.2, label="Total Loss L(w)")
    axes[0].plot(iters, opt.hinge_loss_history_, color="#DD8452", linestyle="--", linewidth=1.8, label="Triplet Hinge Loss")
    axes[0].plot(iters, opt.reg_loss_history_, color="#55A868", linestyle=":", linewidth=1.8, label="L2 Prior Reg ||w - u||^2")
    axes[0].set_title("Konvergensi Triplet Margin Loss (Training Data: 36,168)")
    axes[0].set_xlabel("Iterasi Projected Gradient Descent")
    axes[0].set_ylabel("Nilai Objective Loss")
    axes[0].legend(fontsize=9)

    w_plot = weights_df.sort_values("weight", ascending=True)
    u_val = float(w_plot["uniform_weight"].iloc[0])
    colors = ["#55A868" if v >= u_val else "#8C96A5" for v in w_plot["weight"]]
    bars = axes[1].barh(w_plot["feature"], w_plot["weight"], color=colors)
    axes[1].axvline(u_val, color="#D62728", linestyle="--", linewidth=1.3, label=f"Bobot Seragam u = 1/16 ({u_val:.4f})")
    axes[1].set_title("Bobot Fitur Terpelajar DIFW-KNN (Local-Context Triplets)")
    axes[1].set_xlabel("Bobot Fitur w_j (sum = 1)")
    axes[1].set_xlim(0, max(float(w_plot["weight"].max()) * 1.24, 0.25))
    axes[1].legend(fontsize=8.5, loc="lower right")
    for bar, val in zip(bars, w_plot["weight"]):
        axes[1].text(float(val) + 0.003, bar.get_y() + bar.get_height() / 2, f"{float(val):.4f}", va="center", fontsize=8.5)

    fig.tight_layout()
    p1 = FIGURES_DIR / "01_triplet_loss_and_feature_weights.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # 2. 5-Fold Development Progression (E0 -> E2_dev -> E3_dev -> E4_dev -> E5_dev)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    lbls = dev_progression_df["model_id"].tolist()
    x = np.arange(len(lbls))
    wb = 0.25

    axes[0].bar(x - wb, dev_progression_df["f1_yes_mean"], width=wb, color="#4C72B0", label="F1 (yes)")
    axes[0].bar(x, dev_progression_df["mcc_mean"], width=wb, color="#55A868", label="MCC")
    axes[0].bar(x + wb, dev_progression_df["g_mean_mean"], width=wb, color="#DD8452", label="G-Mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lbls)
    axes[0].set_ylim(0, 0.82)
    axes[0].set_title("Progresi Komponen pada 5-Fold Development Set (Training Only)")
    axes[0].set_ylabel("Skor Rata-Rata 5-Fold Dev")
    axes[0].legend(fontsize=9)
    for idx_m, r in dev_progression_df.iterrows():
        axes[0].text(idx_m - wb, r["f1_yes_mean"] + 0.012, f"{r['f1_yes_mean']:.3f}", ha="center", fontsize=8, rotation=90)
        axes[0].text(idx_m, r["mcc_mean"] + 0.012, f"{r['mcc_mean']:.3f}", ha="center", fontsize=8, rotation=90)
        axes[0].text(idx_m + wb, r["g_mean_mean"] + 0.012, f"{r['g_mean_mean']:.3f}", ha="center", fontsize=8, rotation=90)

    axes[1].bar(x - 0.18, dev_progression_df["pr_auc_mean"], width=0.36, color="#8172B3", label="PR-AUC")
    axes[1].bar(x + 0.18, dev_progression_df["roc_auc_mean"], width=0.36, color="#CCB974", label="ROC-AUC")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(lbls)
    axes[1].set_ylim(0, 0.90)
    axes[1].set_title("Peningkatan Kualitas Ranking Probabilitas Lokal (5-Fold Dev)")
    axes[1].set_ylabel("Skor Rata-Rata 5-Fold Dev")
    axes[1].legend(fontsize=9)
    for idx_m, r in dev_progression_df.iterrows():
        axes[1].text(idx_m - 0.18, r["pr_auc_mean"] + 0.012, f"{r['pr_auc_mean']:.4f}", ha="center", fontsize=8)
        axes[1].text(idx_m + 0.18, r["roc_auc_mean"] + 0.012, f"{r['roc_auc_mean']:.4f}", ha="center", fontsize=8)

    fig.tight_layout()
    p2 = FIGURES_DIR / "02_dev_component_progression.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    # 3. Cost-Sensitive Decision Rule Trade-off on Out-of-Fold Development Predictions
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ratios = cost_grid_df["c_fn"].to_numpy()
    axes[0].plot(ratios, cost_grid_df["precision_yes"], marker="o", linewidth=2, color="#4C72B0", label="Precision (yes)")
    axes[0].plot(ratios, cost_grid_df["recall_yes"], marker="s", linewidth=2, color="#DD8452", label="Recall (yes)")
    axes[0].plot(ratios, cost_grid_df["f1_yes"], marker="^", linewidth=2, color="#55A868", label="F1 (yes)")
    axes[0].plot(ratios, cost_grid_df["mcc"], marker="d", linewidth=2, color="#8172B3", label="MCC")
    axes[0].axvline(DEV_C_FN, color="#333333", linestyle="--", linewidth=1.2, label=f"Default 5:1 (tau={1/(1+DEV_C_FN):.4f})")
    axes[0].set_title("Efek Cost-Sensitive Rule pada Trade-off Metrik (OOF Dev)")
    axes[0].set_xlabel("Rasio Biaya C_FN (dengan C_FP = 1)")
    axes[0].set_ylabel("Skor Out-of-Fold Development")
    axes[0].set_xticks(ratios)
    axes[0].legend(fontsize=8.5)

    axes[1].bar(
        [f"{r:g}:1\n(tau={t:.2f})" for r, t in zip(cost_grid_df["c_fn"], cost_grid_df["threshold"])],
        cost_grid_df["cost_reduction_pct"],
        color="#55A868",
        width=0.58,
    )
    axes[1].set_title("Pengurangan Biaya Kesalahan vs Threshold Standar 0.50 (%)")
    axes[1].set_xlabel("Skenario Rasio Biaya C_FN : C_FP")
    axes[1].set_ylabel("Reduksi Total Biaya (%)")
    axes[1].set_ylim(0, max(float(cost_grid_df["cost_reduction_pct"].max()) * 1.22, 10))
    for i, val in enumerate(cost_grid_df["cost_reduction_pct"]):
        axes[1].text(i, float(val) + 0.8, f"{float(val):.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    p3 = FIGURES_DIR / "03_cost_sensitive_rule_behavior.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved.append(p3)

    return saved


def build_report(
    checks_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    dev_progression_df: pd.DataFrame,
    cost_grid_df: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    """Build Laporan_Proposed_DIFW_KNN.md."""
    w_disp = weights_df.copy()
    for c in ["weight", "uniform_weight", "relative_importance", "weight_pct"]:
        w_disp[c] = w_disp[c].round(4)

    stab_disp = stability_df[["feature", "weight_mean", "weight_std", "weight_min", "weight_max"]].round(5)
    prog_cols = [
        "model_id",
        "model_label",
        "accuracy_mean",
        "precision_yes_mean",
        "recall_yes_mean",
        "f1_yes_mean",
        "pr_auc_mean",
        "roc_auc_mean",
        "mcc_mean",
        "g_mean_mean",
    ]
    cost_disp = cost_grid_df[
        ["c_fn", "threshold", "accuracy", "precision_yes", "recall_yes", "f1_yes", "mcc", "g_mean", "mean_cost", "cost_reduction_pct"]
    ].round(4)

    lines = [
        "# LAPORAN TAHAP 05 — PROPOSED ALGORITHM DEVELOPMENT (`DIFW-KNN`)",
        "",
        "**Modul Kode:** [`src/mixed_type_distance.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/mixed_type_distance.py), [`src/triplet_learning.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/triplet_learning.py), [`src/feature_weight_optimizer.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/feature_weight_optimizer.py), [`src/cost_sensitive_decision.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/cost_sensitive_decision.py), [`src/difw_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/difw_knn.py), [`src/05_proposed_difw_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/05_proposed_difw_knn.py)  ",
        "**Notebook:** [`notebooks/05_Proposed_DIFW_KNN.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/05_Proposed_DIFW_KNN.ipynb)  ",
        "**Kepatuhan Alur Kerja (*Flowchart*):** Seluruh pengembangan dan validasi ke-6 sub-komponen dilakukan **murni pada `Training / Development Data` (`36.168` baris)**; `Hold-out Test Data` (`9.043` baris) tetap dikunci hingga Tahap 07.",
        "",
        "---",
        "",
        "## 1. Validasi 6 Sub-Komponen Algoritma Usulan (`9/9 PASS`)",
        "",
        df_to_markdown_table(checks_df),
        "",
        "---",
        "",
        "## 2. Bobot Fitur Terpelajar (`Supervised Feature Weight Learning`)",
        "",
        f"Konfigurasi inisialisasi pengembangan: `mining_strategy='local_context'`, `n_triplets={DEV_N_TRIPLETS:,}`, `margin={DEV_MARGIN}`, `reg_lambda={DEV_LAMBDA}`.",
        "",
        df_to_markdown_table(w_disp),
        "",
        "### Uji Stabilitas Bobot Fitur Lintas 5 Random Seed (`[42, 100, 2026, 7, 999]`)",
        "",
        df_to_markdown_table(stab_disp),
        "",
        "---",
        "",
        "## 3. Bukti Kontribusi Komponen pada 5-Fold Development Set (Sebelum Hyperparameter Tuning)",
        "",
        df_to_markdown_table(dev_progression_df[prog_cols]),
        "",
        "---",
        "",
        "## 4. Analisis Aturan Keputusan Sensitif Biaya (`Cost-Sensitive Decision Rule`)",
        "",
        df_to_markdown_table(cost_disp),
        "",
        "> **Kesiapan Menuju Tahap 06 (`Cross-Validation & Hyperparameter Tuning`):**",
        "> Seluruh sub-komponen `Proposed DIFW-KNN` telah tervalidasi secara matematis dan empiris pada data pengembangan. Tahap selanjutnya adalah mempertemukan `Baseline KNN` (Tahap 04) dan `Proposed DIFW-KNN` (Tahap 05) di **Tahap 06 (`Cross-Validation & Hyperparameter Tuning`)** untuk mencari konfigurasi *hyperparameter* optimal sebelum membuka `Hold-out Test Data` di Tahap 07.",
        "",
        "---",
        "",
        "## 5. Artefak Visual",
        "",
    ]
    for p in figure_paths:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        lines.append(f"- [{p.name}](file:///{p.as_posix()}) (`{rel}`)")
    return "\n".join(lines)


def main() -> None:
    """Execute Stage 05 Proposed DIFW-KNN development & validation on Training Data."""
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "train_split.csv")
    folds_df = pd.read_csv(PROCESSED_DIR / "train_cv_folds.csv")
    base_summary = pd.read_csv(BASELINE_DIR / "dev_5fold_baseline_summary.csv")

    X_train_df = train_df.drop(columns=["y"])
    y_train = train_df["y"].to_numpy(dtype=np.int64)
    fold_ids = folds_df["fold_id"].to_numpy(dtype=np.int64)

    # 1. Validate all 6 sub-components & 5-seed stability on Training Data
    checks_df, weights_df, stability_df, opt = run_component_validation_suite(X_train_df, y_train)
    checks_df.to_csv(OUTPUT_DIR / "component_validation_checks.csv", index=False)
    weights_df.to_csv(OUTPUT_DIR / "dev_learned_feature_weights.csv", index=False)
    stability_df.to_csv(OUTPUT_DIR / "dev_weight_seed_stability.csv", index=False)
    pd.DataFrame(
        {
            "iteration": np.arange(len(opt.loss_history_)),
            "total_loss": opt.loss_history_,
            "hinge_loss": opt.hinge_loss_history_,
            "reg_loss": opt.reg_loss_history_,
            "active_fraction": opt.active_fraction_history_,
        }
    ).to_csv(OUTPUT_DIR / "dev_triplet_loss_history.csv", index=False)

    # 2. Evaluate E2_dev .. E5_dev across the 5 Development Folds (strictly training/dev)
    per_fold_df, prop_summary_df, oof_dict = evaluate_proposed_on_dev_folds(
        X_train_df=X_train_df,
        y_train=y_train,
        fold_ids=fold_ids,
    )
    per_fold_df.to_csv(OUTPUT_DIR / "dev_5fold_per_fold_metrics.csv", index=False)

    # Combine with E0 from Stage 04 for clear development progression
    e0_row = base_summary.loc[base_summary["variant"] == "E0_Standard_k5_uniform"].iloc[0]
    e0_entry = {
        "model_id": "E0_dev",
        "model_label": "Standard Euclidean KNN (k=5, uniform, tau=0.50)",
        "threshold": 0.50,
    }
    for m in METRIC_KEYS:
        e0_entry[f"{m}_mean"] = float(e0_row[f"{m}_mean"])
        e0_entry[f"{m}_std"] = float(e0_row[f"{m}_std"])
    e0_entry["mean_cost_mean"] = np.nan
    e0_entry["mean_cost_std"] = np.nan

    dev_progression_df = pd.concat(
        [pd.DataFrame([e0_entry]), prop_summary_df],
        ignore_index=True,
    )
    dev_progression_df.to_csv(OUTPUT_DIR / "dev_5fold_component_progression.csv", index=False)

    # 3. Cost-sensitive threshold behavior across cost ratios on OOF predictions
    cost_grid_df = evaluate_cost_ratio_grid(
        y_true=y_train,
        y_prob=oof_dict["oof_weighted"],
        c_fn_grid=(1.0, 2.0, 3.0, 4.0, 5.0, 7.55, 10.0),
        c_fp=DEV_C_FP,
    )
    cost_grid_df.to_csv(OUTPUT_DIR / "dev_cost_ratio_grid.csv", index=False)

    figure_paths = create_proposed_figures(
        opt=opt,
        weights_df=weights_df,
        dev_progression_df=dev_progression_df,
        cost_grid_df=cost_grid_df,
    )

    report = build_report(
        checks_df=checks_df,
        weights_df=weights_df,
        stability_df=stability_df,
        dev_progression_df=dev_progression_df,
        cost_grid_df=cost_grid_df,
        figure_paths=figure_paths,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Tahap 05 (Proposed DIFW-KNN — Training/Development Only) selesai.")
    print(dev_progression_df[["model_id", "accuracy_mean", "precision_yes_mean", "recall_yes_mean", "f1_yes_mean", "pr_auc_mean", "roc_auc_mean", "mcc_mean", "g_mean_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
