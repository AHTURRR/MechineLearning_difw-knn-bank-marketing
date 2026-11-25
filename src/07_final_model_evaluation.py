"""Stage 07: Final Model Evaluation on Hold-out Test Data (9,043 Observations).

Corresponds to the 'Final Model Evaluation' convergence node on the research
flowchart where:
  1. Locked optimal hyperparameters from Stage 06 (outputs/cv_tuning/selected_hyperparameters.json)
     are loaded without modification.
  2. Canonical models E0, E1 (plus E1-PR reference), E2, E3, E4, and E5 are trained
     on the ENTIRE Training / Development Data (36,168 observations).
  3. Models are evaluated ONE TIME objectively on the locked Hold-out Test Data
     (9,043 observations), computing Accuracy, Precision (yes), Recall (yes),
     F1 (yes), PR-AUC, ROC-AUC, MCC, G-Mean, Expected Cost (5:1), Confusion Matrices,
     computational runtime, and paired 95% Bootstrap Confidence Intervals (n_boot=1,000).
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
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_knn import METRIC_KEYS, compute_classification_metrics
from src.cost_sensitive_decision import CostSensitiveDecisionRule
from src.difw_knn import DIFWKNNClassifier
from src.preprocessing import BaselineEuclideanPreprocessor


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CV_DIR = PROJECT_ROOT / "outputs" / "cv_tuning"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_evaluation"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "Laporan_Final_Evaluation_DIFW_KNN.md"

RANDOM_STATE = 42
N_BOOTSTRAP = 1000


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


def compute_paired_bootstrap_cis(
    y_true: np.ndarray,
    model_specs: list[tuple[str, np.ndarray, float]],
    c_fp: float = 1.0,
    c_fn: float = 5.0,
    n_boot: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Compute paired 95% bootstrap confidence intervals across all models simultaneously."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    keys_ext = list(METRIC_KEYS) + ["mean_cost"]
    store: dict[str, dict[str, list[float]]] = {
        mid: {k: [] for k in keys_ext} for mid, _, _ in model_specs
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if yb.min() == yb.max():
            continue
        for mid, prob_arr, tau in model_specs:
            pb = prob_arr[idx]
            m = compute_classification_metrics(yb, pb, threshold=tau)
            mean_c = (c_fp * m["fp"] + c_fn * m["fn"]) / float(n)
            for k in METRIC_KEYS:
                store[mid][k].append(m[k])
            store[mid]["mean_cost"].append(mean_c)

    out: dict[str, dict[str, tuple[float, float]]] = {}
    for mid, _, _ in model_specs:
        out[mid] = {}
        for k in keys_ext:
            arr = np.asarray(store[mid][k], dtype=np.float64)
            out[mid][k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return out


def train_and_predict_all_models(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_test_df: pd.DataFrame,
    selected_cfg: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Fit full-training models using locked Stage 06 hyperparameters and predict on Hold-out Test."""
    outputs: dict[str, dict[str, Any]] = {}

    # 1. Shared Baseline Euclidean Preprocessor (fitted strictly on 36,168 training rows)
    t_prep0 = time.perf_counter()
    euc_prep = BaselineEuclideanPreprocessor()
    X_train_euc = euc_prep.fit_transform(X_train_df)
    prep_fit_sec = time.perf_counter() - t_prep0

    t_prep_te0 = time.perf_counter()
    X_test_euc = euc_prep.transform(X_test_df)
    prep_te_sec = time.perf_counter() - t_prep_te0

    for mid, label in [
        ("E0", "Standard KNN"),
        ("E1", "Optimized KNN (Best CV F1)"),
        ("E1_PR", "Optimized KNN (Best CV PR-AUC)"),
    ]:
        cfg = selected_cfg[mid]
        t_f0 = time.perf_counter()
        clf = KNeighborsClassifier(
            n_neighbors=int(cfg["k"]),
            weights=str(cfg["weights"]),
            metric="euclidean",
            n_jobs=-1,
        )
        clf.fit(X_train_euc, y_train)
        fit_sec = prep_fit_sec + (time.perf_counter() - t_f0)

        t_p0 = time.perf_counter()
        prob = clf.predict_proba(X_test_euc)[:, 1]
        pred_sec = prep_te_sec + (time.perf_counter() - t_p0)

        outputs[mid] = {
            "model_name": label,
            "config_desc": f"Euclidean k={cfg['k']}, {cfg['weights']}, tau={cfg['threshold']:.2f}",
            "prob": prob,
            "threshold": float(cfg["threshold"]),
            "fit_sec": fit_sec,
            "predict_sec": pred_sec,
        }

    # 2. Mixed-Type Uniform Model (shared neighbor search for E2 & E4)
    cfg_e2 = selected_cfg["E2"]
    cfg_e4 = selected_cfg["E4"]
    q_e2 = float(cfg_e2["q"])
    nw_e2 = "uniform" if q_e2 == 0.0 else "distance"
    t_u0 = time.perf_counter()
    model_u = DIFWKNNClassifier(
        k=int(cfg_e2["k"]),
        neighbor_weights=nw_e2,
        distance_power_q=max(q_e2, 1.0) if nw_e2 == "uniform" else q_e2,
        learn_feature_weights=False,
        use_cost_sensitive=False,
        batch_size=256,
        random_state=RANDOM_STATE,
    ).fit(X_train_df, y_train)
    fit_u_sec = time.perf_counter() - t_u0

    t_up0 = time.perf_counter()
    prob_u = model_u.predict_proba(X_test_df)[:, 1]
    pred_u_sec = time.perf_counter() - t_up0

    outputs["E2"] = {
        "model_name": "Mixed-Type KNN",
        "config_desc": f"Mixed Uniform k={cfg_e2['k']}, q={q_e2:g}, tau=0.50",
        "prob": prob_u,
        "threshold": float(cfg_e2["threshold"]),
        "fit_sec": fit_u_sec,
        "predict_sec": pred_u_sec,
    }
    outputs["E4"] = {
        "model_name": "Cost-Sensitive KNN",
        "config_desc": f"Mixed Cost k={cfg_e4['k']}, q={q_e2:g}, {cfg_e4['c_fn']:g}:1 (tau={cfg_e4['threshold']:.4f})",
        "prob": prob_u,
        "threshold": float(cfg_e4["threshold"]),
        "fit_sec": fit_u_sec,
        "predict_sec": pred_u_sec,
    }

    # 3. Proposed Feature-Weighted DIFW-KNN Model (shared neighbor search for E3 & E5)
    cfg_e5 = selected_cfg["E5"]
    q_e5 = float(cfg_e5["q"])
    nw_e5 = "uniform" if q_e5 == 0.0 else "distance"
    t_w0 = time.perf_counter()
    model_w = DIFWKNNClassifier(
        k=int(cfg_e5["k"]),
        neighbor_weights=nw_e5,
        distance_power_q=max(q_e5, 1.0) if nw_e5 == "uniform" else q_e5,
        learn_feature_weights=True,
        use_cost_sensitive=True,
        c_fp=float(cfg_e5["c_fp"]),
        c_fn=float(cfg_e5["c_fn"]),
        margin=float(cfg_e5["margin"]),
        reg_lambda=float(cfg_e5["reg_lambda"]),
        mining_strategy=str(cfg_e5["mining_strategy"]),
        n_triplets=int(cfg_e5["n_triplets"]),
        candidate_pool_size=int(cfg_e5["candidate_pool_size"]),
        max_iter=int(cfg_e5["max_iter"]),
        batch_size=256,
        random_state=RANDOM_STATE,
    ).fit(X_train_df, y_train)
    fit_w_sec = time.perf_counter() - t_w0

    t_wp0 = time.perf_counter()
    prob_w = model_w.predict_proba(X_test_df)[:, 1]
    pred_w_sec = time.perf_counter() - t_wp0

    outputs["E3"] = {
        "model_name": "Feature-Weighted KNN",
        "config_desc": f"DIFW m={cfg_e5['margin']:g}, lam={cfg_e5['reg_lambda']:g}, k={cfg_e5['k']}, q={q_e5:g}, tau=0.50",
        "prob": prob_w,
        "threshold": float(selected_cfg["E3"]["threshold"]),
        "fit_sec": fit_w_sec,
        "predict_sec": pred_w_sec,
    }
    outputs["E5"] = {
        "model_name": "DIFW-KNN Lengkap",
        "config_desc": f"DIFW m={cfg_e5['margin']:g}, lam={cfg_e5['reg_lambda']:g}, k={cfg_e5['k']}, q={q_e5:g}, {cfg_e5['c_fn']:g}:1 (tau={cfg_e5['threshold']:.4f})",
        "prob": prob_w,
        "threshold": float(cfg_e5["threshold"]),
        "fit_sec": fit_w_sec,
        "predict_sec": pred_w_sec,
    }

    weights_df = model_w.get_feature_weights_table()
    return outputs, weights_df


def create_final_evaluation_figures(
    y_test: np.ndarray,
    results_df: pd.DataFrame,
    raw_outputs: dict[str, dict[str, Any]],
) -> list[Path]:
    """Generate 3 publication-quality figures for Hold-out Test Final Evaluation."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Hold-out ROC & Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    curve_specs = [
        ("E0", "E0 Standard KNN (k=5, uniform)", "#7F7F7F", "--"),
        ("E1", "E1 Optimized KNN (k=3, uniform)", "#4C72B0", "-."),
        ("E1_PR", "E1-PR Euclidean Best CV PR-AUC (k=51, distance)", "#8172B3", ":"),
        ("E2", "E2/E4 Mixed-Type Uniform (k=71, q=0.5)", "#DD8452", "--"),
        ("E5", "E3/E5 Proposed DIFW-KNN (k=71, q=0.5, Triplet)", "#2CA02C", "-"),
    ]
    res_lookup = results_df.set_index("model_id")
    for mid, lbl, col, ls in curve_specs:
        prob = raw_outputs[mid]["prob"]
        r_row = res_lookup.loc[mid]
        lw = 2.5 if mid == "E5" else 1.8
        fpr, tpr, _ = roc_curve(y_test, prob)
        prec, rec, _ = precision_recall_curve(y_test, prob)
        axes[0].plot(fpr, tpr, color=col, linestyle=ls, linewidth=lw, label=f"{lbl} | AUC={r_row['roc_auc']:.4f}")
        axes[1].plot(rec, prec, color=col, linestyle=ls, linewidth=lw, label=f"{lbl} | PR-AUC={r_row['pr_auc']:.4f}")

    axes[0].plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=0.9, label="Random (0.5000)")
    axes[0].set_title("Kurva ROC — Hold-out Test Set (9,043 Observasi)")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(fontsize=7.8, loc="lower right")

    pos_rate = float(y_test.mean())
    axes[1].axhline(pos_rate, color="#333333", linestyle="--", linewidth=0.9, label=f"No-Skill ({pos_rate:.4f})")
    axes[1].set_title("Kurva Precision–Recall — Hold-out Test Set (9,043 Observasi)")
    axes[1].set_xlabel("Recall (yes)")
    axes[1].set_ylabel("Precision (yes)")
    axes[1].legend(fontsize=7.8, loc="upper right")

    fig.tight_layout()
    p1 = FIGURES_DIR / "01_holdout_roc_and_pr_curves.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # 2. Canonical E0..E5 Metrics with 95% Bootstrap CI & Expected Misclassification Cost
    canon_df = results_df[results_df["model_id"].isin(["E0", "E1", "E2", "E3", "E4", "E5"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0), gridspec_kw={"width_ratios": [1.65, 1.0]})

    x = np.arange(len(canon_df))
    wb = 0.20
    bar_metrics = [
        ("f1_yes", "F1 (yes)", "#4C72B0", -1.5 * wb),
        ("mcc", "MCC", "#55A868", -0.5 * wb),
        ("pr_auc", "PR-AUC", "#8172B3", 0.5 * wb),
        ("g_mean", "G-Mean", "#DD8452", 1.5 * wb),
    ]
    for m_key, m_lbl, col, offset in bar_metrics:
        vals = canon_df[m_key].to_numpy()
        err_lo = vals - canon_df[f"{m_key}_ci95_low"].to_numpy()
        err_hi = canon_df[f"{m_key}_ci95_high"].to_numpy() - vals
        axes[0].bar(x + offset, vals, width=wb, yerr=[err_lo, err_hi], capsize=2.5, color=col, label=m_lbl)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(canon_df["model_id"])
    axes[0].set_ylim(0, 0.82)
    axes[0].set_title("Metrik Utama pada Hold-out Test Set (±95% Bootstrap CI)")
    axes[0].set_ylabel("Skor Hold-out Test")
    axes[0].legend(fontsize=8.5, ncol=2, loc="upper left")

    cost_vals = canon_df["mean_cost"].to_numpy()
    c_err_lo = cost_vals - canon_df["mean_cost_ci95_low"].to_numpy()
    c_err_hi = canon_df["mean_cost_ci95_high"].to_numpy() - cost_vals
    c_colors = ["#8C96A5", "#8C96A5", "#8C96A5", "#4C72B0", "#DD8452", "#2CA02C"]
    bars_c = axes[1].bar(canon_df["model_id"], cost_vals, yerr=[c_err_lo, c_err_hi], capsize=3, color=c_colors, width=0.55)
    axes[1].set_title("Rata-Rata Biaya Kesalahan per Nasabah\n(Skenario 5:1, lebih rendah = lebih baik)")
    axes[1].set_ylabel("Expected Cost per Sample")
    axes[1].set_ylim(0, max(cost_vals) * 1.22)
    for bar, val in zip(bars_c, cost_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.018, f"{val:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    p2 = FIGURES_DIR / "02_holdout_metrics_with_95ci.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    # 3. 2x3 Confusion Matrix Grid for Canonical Models E0..E5
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    for ax, (_, r) in zip(axes.flat, canon_df.iterrows()):
        cm = np.array([[int(r["tn"]), int(r["fp"])], [int(r["fn"]), int(r["tp"])]])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred no", "Pred yes"])
        ax.set_yticklabels(["Actual no", "Actual yes"])
        ax.set_title(f"{r['model_id']}: {r['model_name']}\n(F1={r['f1_yes']:.4f}, MCC={r['mcc']:.4f})")
        ax.grid(False)
        th = cm.max() / 2.0
        for ri in range(2):
            for ci in range(2):
                ax.text(
                    ci,
                    ri,
                    f"{cm[ri, ci]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[ri, ci] > th else "#111111",
                    fontweight="bold",
                    fontsize=10.5,
                )
    fig.suptitle("Confusion Matrix Hold-out Test Set (9,043 Observasi: 7,985 'no' & 1,058 'yes')", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p3 = FIGURES_DIR / "03_holdout_confusion_matrices_grid.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved.append(p3)

    return saved


def build_report(
    results_df: pd.DataFrame,
    ci_formatted_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    """Build Laporan_Final_Evaluation_DIFW_KNN.md."""
    main_cols = [
        "model_id",
        "model_name",
        "accuracy",
        "precision_yes",
        "recall_yes",
        "f1_yes",
        "pr_auc",
        "roc_auc",
        "mcc",
        "g_mean",
        "mean_cost",
        "cost_reduction_vs_e0_pct",
        "fit_sec",
        "predict_sec",
    ]
    disp_main = results_df[main_cols].copy()
    for c in METRIC_KEYS + ("mean_cost",):
        disp_main[c] = disp_main[c].round(4)
    disp_main["cost_reduction_vs_e0_pct"] = disp_main["cost_reduction_vs_e0_pct"].round(2)
    disp_main["fit_sec"] = disp_main["fit_sec"].round(3)
    disp_main["predict_sec"] = disp_main["predict_sec"].round(3)

    e0 = results_df.loc[results_df["model_id"] == "E0"].iloc[0]
    e5 = results_df.loc[results_df["model_id"] == "E5"].iloc[0]

    lines = [
        "# LAPORAN TAHAP 07 — FINAL MODEL EVALUATION (`HOLD-OUT TEST DATA`)",
        "",
        "**Skrip CLI:** [`src/07_final_model_evaluation.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/07_final_model_evaluation.py)  ",
        "**Notebook:** [`notebooks/07_Final_Model_Evaluation.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/07_Final_Model_Evaluation.ipynb)  ",
        "**Protokol Evaluasi:** Seluruh model dilatih pada `36.168` baris `Training / Development Data` menggunakan *hyperparameter* terkunci dari Tahap 06 ([`selected_hyperparameters.json`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/outputs/cv_tuning/selected_hyperparameters.json)), kemudian dievaluasi **satu kali** pada `9.043` baris **`Hold-out Test Data`** (`7.985` `no` dan `1.058` `yes`).",
        "",
        "---",
        "",
        "## 1. Tabel Performa Final pada Hold-out Test Set (`9.043` Observasi)",
        "",
        df_to_markdown_table(disp_main),
        "",
        "---",
        "",
        "## 2. Interval Kepercayaan Bootstrap Berpasangan 95% (`n_boot = 1.000`)",
        "",
        df_to_markdown_table(ci_formatted_df),
        "",
        "---",
        "",
        "## 3. Confusion Matrix & Biaya Kesalahan (`C_FP = 1.0, C_FN = 5.0`)",
        "",
        df_to_markdown_table(cm_df),
        "",
        "---",
        "",
        "## 4. Bobot Fitur Final `DIFW-KNN` (Dilatih pada Seluruh `36.168` Data Training)",
        "",
        df_to_markdown_table(weights_df.round(5)),
        "",
        "> **Ringkasan Temuan Evaluasi Final (`E5` vs `E0` pada Hold-out Test):**",
        f"> - **`PR-AUC`:** Meningkat dari `{e0['pr_auc']:.4f}` (`E0`) menjadi **`{e5['pr_auc']:.4f}`** (`E5`).",
        f"> - **`ROC-AUC`:** Meningkat dari `{e0['roc_auc']:.4f}` (`E0`) menjadi **`{e5['roc_auc']:.4f}`** (`E5`).",
        f"> - **`F1 (yes)`:** Meningkat dari `{e0['f1_yes']:.4f}` (`E0`) menjadi **`{e5['f1_yes']:.4f}`** (`E5`).",
        f"> - **`MCC`:** Meningkat dari `{e0['mcc']:.4f}` (`E0`) menjadi **`{e5['mcc']:.4f}`** (`E5`).",
        f"> - **`Recall (yes)` & `G-Mean`:** `Recall` melonjak dari `{e0['recall_yes']:.4f}` menjadi **`{e5['recall_yes']:.4f}`** (`G-Mean` dari `{e0['g_mean']:.4f}` menjadi **`{e5['g_mean']:.4f}`**), menurunkan total biaya kesalahan sebesar **`{e5['cost_reduction_vs_e0_pct']:.2f}%`**.",
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
    """Execute Stage 07 one-time Hold-out Test final evaluation."""
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "train_split.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_split.csv")
    selected_cfg = json.loads((CV_DIR / "selected_hyperparameters.json").read_text(encoding="utf-8"))

    X_train_df = train_df.drop(columns=["y"])
    y_train = train_df["y"].to_numpy(dtype=np.int64)
    X_test_df = test_df.drop(columns=["y"])
    y_test = test_df["y"].to_numpy(dtype=np.int64)

    c_fp = float(selected_cfg["E5"]["c_fp"])
    c_fn = float(selected_cfg["E5"]["c_fn"])
    cost_eval_rule = CostSensitiveDecisionRule(c_fp=c_fp, c_fn=c_fn)

    raw_outputs, weights_df = train_and_predict_all_models(
        X_train_df=X_train_df,
        y_train=y_train,
        X_test_df=X_test_df,
        selected_cfg=selected_cfg,
    )
    weights_df.to_csv(OUTPUT_DIR / "final_full_train_feature_weights.csv", index=False)

    ordered_ids = ["E0", "E1", "E1_PR", "E2", "E3", "E4", "E5"]
    boot_specs = [
        (mid, raw_outputs[mid]["prob"], raw_outputs[mid]["threshold"])
        for mid in ordered_ids
    ]
    cis_map = compute_paired_bootstrap_cis(
        y_true=y_test,
        model_specs=boot_specs,
        c_fp=c_fp,
        c_fn=c_fn,
        n_boot=N_BOOTSTRAP,
        random_state=RANDOM_STATE,
    )

    e0_cost_eval = cost_eval_rule.evaluate(
        y_test, raw_outputs["E0"]["prob"], threshold=raw_outputs["E0"]["threshold"]
    )
    e0_total_cost = float(e0_cost_eval["total_cost"])

    rows: list[dict[str, Any]] = []
    ci_fmt_rows: list[dict[str, Any]] = []
    pred_df = pd.DataFrame({"y_true": y_test})

    for mid in ordered_ids:
        info = raw_outputs[mid]
        prob = info["prob"]
        tau = info["threshold"]
        ev = cost_eval_rule.evaluate(y_test, prob, threshold=tau)
        cost_red_vs_e0 = 100.0 * (e0_total_cost - float(ev["total_cost"])) / e0_total_cost

        row: dict[str, Any] = {
            "model_id": mid,
            "model_name": info["model_name"],
            "config_desc": info["config_desc"],
            "threshold": tau,
            "fit_sec": info["fit_sec"],
            "predict_sec": info["predict_sec"],
            **{k: ev[k] for k in METRIC_KEYS},
            "tn": ev["tn"],
            "fp": ev["fp"],
            "fn": ev["fn"],
            "tp": ev["tp"],
            "total_cost": ev["total_cost"],
            "mean_cost": ev["mean_cost"],
            "cost_reduction_vs_e0_pct": cost_red_vs_e0,
        }
        fmt_row: dict[str, Any] = {"model_id": mid, "model_name": info["model_name"]}
        for k in list(METRIC_KEYS) + ["mean_cost"]:
            lo, hi = cis_map[mid][k]
            row[f"{k}_ci95_low"] = lo
            row[f"{k}_ci95_high"] = hi
            fmt_row[k] = f"{row[k]:.4f} [{lo:.4f}, {hi:.4f}]"

        rows.append(row)
        ci_fmt_rows.append(fmt_row)
        pred_df[f"prob_{mid.lower()}"] = prob
        pred_df[f"pred_{mid.lower()}"] = (prob >= tau).astype(np.int64)

    results_df = pd.DataFrame(rows)
    ci_formatted_df = pd.DataFrame(ci_fmt_rows)
    cm_df = results_df[
        ["model_id", "model_name", "threshold", "tn", "fp", "fn", "tp", "total_cost", "mean_cost", "cost_reduction_vs_e0_pct"]
    ].copy()
    cm_df["mean_cost"] = cm_df["mean_cost"].round(4)
    cm_df["cost_reduction_vs_e0_pct"] = cm_df["cost_reduction_vs_e0_pct"].round(2)

    results_df.to_csv(OUTPUT_DIR / "holdout_metrics_all_models.csv", index=False)
    results_df[results_df["model_id"] != "E1_PR"].to_csv(OUTPUT_DIR / "holdout_metrics_e0_to_e5.csv", index=False)
    ci_formatted_df.to_csv(OUTPUT_DIR / "holdout_bootstrap_ci_formatted.csv", index=False)
    cm_df.to_csv(OUTPUT_DIR / "holdout_confusion_matrices.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / "holdout_predictions_all_models.csv", index=False)

    figure_paths = create_final_evaluation_figures(
        y_test=y_test,
        results_df=results_df,
        raw_outputs=raw_outputs,
    )

    report = build_report(
        results_df=results_df,
        ci_formatted_df=ci_formatted_df,
        cm_df=cm_df,
        weights_df=weights_df,
        figure_paths=figure_paths,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Tahap 07 (Final Model Evaluation on Hold-out Test Set) selesai.")
    print(
        results_df[
            ["model_id", "accuracy", "precision_yes", "recall_yes", "f1_yes", "pr_auc", "roc_auc", "mcc", "g_mean", "mean_cost"]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
