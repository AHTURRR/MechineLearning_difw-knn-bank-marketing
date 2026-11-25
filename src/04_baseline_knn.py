"""Stage 04: Baseline KNN Model Development on Training / Development Data.

Strictly adheres to the research flowchart:
- Operates EXCLUSIVELY on Training / Development Data (36,168 observations);
  Hold-out Test Data (9,043 observations) remains locked until Stage 07.
- Verifies exact probability equivalence between fast NearestNeighbors helper and
  scikit-learn's KNeighborsClassifier on Fold 0.
- Evaluates Standard Baseline KNN (E0: k=5, weights='uniform', Euclidean distance on
  StandardScaler + OneHotEncoder representation) and k=5 distance-weighted reference
  across the 5 canonical development folds (Out-of-Fold evaluation).
- Saves development tables, diagnostic figures, and Laporan_Baseline_KNN_DIFW_KNN.md.
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
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_knn import (
    METRIC_KEYS,
    compute_classification_metrics,
    verify_knn_probability_equivalence,
)
from src.preprocessing import BaselineEuclideanPreprocessor


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_knn"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "Laporan_Baseline_KNN_DIFW_KNN.md"


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


def evaluate_baseline_on_dev_folds(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate baseline k=5 Euclidean KNN variants on 5 development folds (leak-free)."""
    configs = [
        ("E0_Standard_k5_uniform", 5, "uniform"),
        ("Ref_Euclidean_k5_distance", 5, "distance"),
    ]
    oof_probs: dict[str, np.ndarray] = {
        name: np.zeros(len(y_train), dtype=np.float64) for name, _, _ in configs
    }
    per_fold_rows: list[dict[str, Any]] = []

    for fold_id in sorted(int(f) for f in np.unique(fold_ids)):
        val_mask = fold_ids == fold_id
        tr_mask = ~val_mask

        X_tr = X_train_df.loc[tr_mask].reset_index(drop=True)
        y_tr = y_train[tr_mask]
        X_va = X_train_df.loc[val_mask].reset_index(drop=True)
        y_va = y_train[val_mask]

        t0 = time.perf_counter()
        prep = BaselineEuclideanPreprocessor()
        X_tr_euc = prep.fit_transform(X_tr)
        X_va_euc = prep.transform(X_va)
        prep_sec = time.perf_counter() - t0

        for name, k_val, w_val in configs:
            t1 = time.perf_counter()
            clf = KNeighborsClassifier(
                n_neighbors=k_val,
                weights=w_val,
                metric="euclidean",
                n_jobs=-1,
            )
            clf.fit(X_tr_euc, y_tr)
            y_prob = clf.predict_proba(X_va_euc)[:, 1]
            knn_sec = time.perf_counter() - t1

            oof_probs[name][val_mask] = y_prob
            metrics = compute_classification_metrics(y_va, y_prob, threshold=0.5)
            per_fold_rows.append(
                {
                    "variant": name,
                    "fold_id": fold_id,
                    "k": k_val,
                    "weights": w_val,
                    "prep_sec": round(prep_sec, 4),
                    "knn_sec": round(knn_sec, 4),
                    **metrics,
                }
            )

    per_fold_df = pd.DataFrame(per_fold_rows)
    summary_rows: list[dict[str, Any]] = []
    for name, k_val, w_val in configs:
        sub = per_fold_df[per_fold_df["variant"] == name]
        oof_m = compute_classification_metrics(y_train, oof_probs[name], threshold=0.5)
        row: dict[str, Any] = {
            "variant": name,
            "k": k_val,
            "weights": w_val,
        }
        for m in METRIC_KEYS:
            row[f"{m}_mean"] = round(float(sub[m].mean()), 4)
            row[f"{m}_std"] = round(float(sub[m].std(ddof=1)), 4)
        row["oof_tn"] = oof_m["tn"]
        row["oof_fp"] = oof_m["fp"]
        row["oof_fn"] = oof_m["fn"]
        row["oof_tp"] = oof_m["tp"]
        summary_rows.append(row)

    return per_fold_df, pd.DataFrame(summary_rows), oof_probs


def create_baseline_dev_figures(
    per_fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    y_train: np.ndarray,
    oof_probs: dict[str, np.ndarray],
) -> list[Path]:
    """Generate development-only figures for Baseline KNN (E0)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Per-fold stability & mean metric profile of E0 Standard KNN
    e0_folds = per_fold_df[per_fold_df["variant"] == "E0_Standard_k5_uniform"].sort_values("fold_id")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    fold_lbls = [f"Fold {int(f)}" for f in e0_folds["fold_id"]]
    for m_col, label, color, marker in [
        ("f1_yes", "F1 (yes)", "#4C72B0", "o"),
        ("mcc", "MCC", "#55A868", "s"),
        ("pr_auc", "PR-AUC", "#DD8452", "^"),
        ("recall_yes", "Recall (yes)", "#C44E52", "d"),
    ]:
        axes[0].plot(fold_lbls, e0_folds[m_col], marker=marker, linewidth=2, color=color, label=label)
    axes[0].set_ylim(0.14, 0.36)
    axes[0].set_title("Konsistensi Metrik E0 Standard KNN (k=5, uniform)\nAcross 5 Development Folds")
    axes[0].set_ylabel("Skor Validasi per Fold")
    axes[0].legend(fontsize=8.5)

    x = np.arange(len(METRIC_KEYS))
    w = 0.36
    r0 = summary_df.iloc[0]
    r1 = summary_df.iloc[1]
    axes[1].bar(
        x - w / 2,
        [r0[f"{m}_mean"] for m in METRIC_KEYS],
        yerr=[r0[f"{m}_std"] for m in METRIC_KEYS],
        width=w,
        capsize=3,
        color="#4C72B0",
        label="E0: k=5, uniform",
    )
    axes[1].bar(
        x + w / 2,
        [r1[f"{m}_mean"] for m in METRIC_KEYS],
        yerr=[r1[f"{m}_std"] for m in METRIC_KEYS],
        width=w,
        capsize=3,
        color="#DD8452",
        label="Ref: k=5, distance",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Acc", "Prec", "Rec", "F1", "PR-AUC", "ROC-AUC", "MCC", "G-Mean"], rotation=20)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Profil Metrik Baseline KNN pada Development Data (±1 SD)")
    axes[1].set_ylabel("Skor Rata-Rata 5-Fold Dev")
    axes[1].legend(fontsize=8.5)

    fig.tight_layout()
    p1 = FIGURES_DIR / "01_baseline_dev_fold_metrics.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # 2. Out-of-Fold ROC, PR Curve, and Confusion Matrix on Training/Dev (36,168)
    fig = plt.figure(figsize=(14.5, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.95])
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_cm = fig.add_subplot(gs[0, 2])

    for (name, label, col) in [
        ("E0_Standard_k5_uniform", "E0 Standard (k=5, uniform)", "#4C72B0"),
        ("Ref_Euclidean_k5_distance", "Ref Euclidean (k=5, distance)", "#DD8452"),
    ]:
        p_oof = oof_probs[name]
        m_oof = compute_classification_metrics(y_train, p_oof, threshold=0.5)
        fpr, tpr, _ = roc_curve(y_train, p_oof)
        prec, rec, _ = precision_recall_curve(y_train, p_oof)
        ax_roc.plot(fpr, tpr, color=col, linewidth=2, label=f"{label}\nROC-AUC={m_oof['roc_auc']:.4f}")
        ax_pr.plot(rec, prec, color=col, linewidth=2, label=f"{label}\nPR-AUC={m_oof['pr_auc']:.4f}")

    ax_roc.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1, label="Random (0.5000)")
    ax_roc.set_title("Out-of-Fold ROC Curve (Training/Dev)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=8, loc="lower right")

    pos_rate = float(y_train.mean())
    ax_pr.axhline(pos_rate, color="#666666", linestyle="--", linewidth=1, label=f"No-Skill ({pos_rate:.4f})")
    ax_pr.set_title("Out-of-Fold PR Curve (Training/Dev)")
    ax_pr.set_xlabel("Recall (yes)")
    ax_pr.set_ylabel("Precision (yes)")
    ax_pr.legend(fontsize=8, loc="upper right")

    cm = np.array([[int(r0["oof_tn"]), int(r0["oof_fp"])], [int(r0["oof_fn"]), int(r0["oof_tp"])]])
    ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred no", "Pred yes"])
    ax_cm.set_yticklabels(["Actual no", "Actual yes"])
    ax_cm.set_title("OOF Confusion Matrix E0\n(Training/Dev: 36,168)")
    th = cm.max() / 2.0
    for r in range(2):
        for c in range(2):
            ax_cm.text(
                c,
                r,
                f"{cm[r, c]:,}",
                ha="center",
                va="center",
                color="white" if cm[r, c] > th else "black",
                fontweight="bold",
                fontsize=10.5,
            )

    fig.tight_layout()
    p2 = FIGURES_DIR / "02_baseline_oof_roc_pr_confusion.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    return saved


def build_report(
    eq_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    """Build Laporan_Baseline_KNN_DIFW_KNN.md strictly scoped to Development Data."""
    e0 = summary_df.iloc[0]
    disp_cols = [
        "variant",
        "k",
        "weights",
        "accuracy_mean",
        "precision_yes_mean",
        "recall_yes_mean",
        "f1_yes_mean",
        "pr_auc_mean",
        "roc_auc_mean",
        "mcc_mean",
        "g_mean_mean",
    ]
    cm_cols = ["variant", "oof_tn", "oof_fp", "oof_fn", "oof_tp"]

    lines = [
        "# LAPORAN TAHAP 04 — BASELINE KNN (`TRAINING / DEVELOPMENT DATA`)",
        "",
        "**Modul Kode:** [`src/baseline_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/baseline_knn.py), [`src/04_baseline_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/04_baseline_knn.py)  ",
        "**Notebook:** [`notebooks/04_Baseline_KNN.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/04_Baseline_KNN.ipynb)  ",
        "**Kepatuhan Alur Kerja (*Flowchart*):** Tahap ini **hanya menggunakan `Training / Development Data` (`36.168` baris)**; `Hold-out Test Data` (`9.043` baris) tetap dikunci dan **belum diakses** hingga Tahap 07 (*Final Model Evaluation*).",
        "",
        "---",
        "",
        "## 1. Spesifikasi Arsitektur Baseline KNN (`E0`)",
        "",
        "- **Representasi Fitur:** `StandardScaler` pada 6 fitur numerik + `OneHotEncoder(handle_unknown='ignore')` pada 10 fitur kategorikal/biner (`52` dimensi setelah transformasi per *fold* latih).",
        "- ** Pengukuran Jarak:** Jarak Euclidean standar tanpa pembobotan fitur domain.",
        "- **Konfigurasi Dasar (`E0`):** `k = 5`, `weights = 'uniform'`, *decision threshold* $\\tau = 0,50$.",
        "",
        "---",
        "",
        "## 2. Unit Test Ekuivalensi Implementasi Pencarian Tetangga (Fold 0)",
        "",
        df_to_markdown_table(eq_df),
        "",
        "---",
        "",
        "## 3. Profil Kinerja Baseline pada 5-Fold Development Set (`36.168` Observasi)",
        "",
        df_to_markdown_table(summary_df[disp_cols]),
        "",
        "### Out-of-Fold Confusion Matrix (36.168 Baris Training/Development)",
        "",
        df_to_markdown_table(summary_df[cm_cols]),
        "",
        "> **Diagnosis Kelemahan Baseline KNN pada Data Training:**",
        f"> - Pada konfigurasi standar `E0 (k=5, uniform)`, nilai `Accuracy` tampak tinggi (`{e0['accuracy_mean']:.4f}`) semata-mata karena kelas mayoritas `no` mendominasi `88,30%` data.",
        f"> - Namun untuk kelas target `yes`, `Recall` hanya mencapai **`{e0['recall_yes_mean']:.4f}`** (`{int(e0['oof_fn']):,}` dari `{int(e0['oof_fn'] + e0['oof_tp']):,}` nasabah potensial gagal terdeteksi!), dengan `F1 = {e0['f1_yes_mean']:.4f}`, `PR-AUC = {e0['pr_auc_mean']:.4f}`, dan `MCC = {e0['mcc_mean']:.4f}`.",
        "> - Kelemahan ini timbul karena: (1) jarak Euclidean di atas `52` kolom *one-hot + z-score* terdistorsi oleh outlier numerik dan dilusi fitur berdaya pisah rendah, serta (2) ambang batas $\\tau = 0,50$ tidak memperhitungkan ketimpangan biaya kesalahan (*false negative* vs *false positive*).",
        "> - Model Baseline KNN selanjutnya akan dibawa ke **Tahap 06 (`Cross-Validation & Hyperparameter Tuning`)** bersama **Tahap 05 (`Proposed DIFW-KNN`)**.",
        "",
        "---",
        "",
        "## 4. Artefak Visual",
        "",
    ]
    for p in figure_paths:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        lines.append(f"- [{p.name}](file:///{p.as_posix()}) (`{rel}`)")
    return "\n".join(lines)


def main() -> None:
    """Execute Stage 04 Baseline KNN strictly on Training / Development Data."""
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "train_split.csv")
    folds_df = pd.read_csv(PROCESSED_DIR / "train_cv_folds.csv")

    X_train_df = train_df.drop(columns=["y"])
    y_train = train_df["y"].to_numpy(dtype=np.int64)
    fold_ids = folds_df["fold_id"].to_numpy(dtype=np.int64)

    # 1. Equivalence unit check on Fold 0
    val0 = fold_ids == 0
    prep0 = BaselineEuclideanPreprocessor()
    X_tr0 = prep0.fit_transform(X_train_df.loc[~val0].reset_index(drop=True))
    X_va0 = prep0.transform(X_train_df.loc[val0].reset_index(drop=True))
    eq_df = verify_knn_probability_equivalence(X_tr0, y_train[~val0], X_va0)
    eq_df.to_csv(OUTPUT_DIR / "unit_test_equivalence.csv", index=False)

    # 2. Evaluate Baseline KNN on 5-Fold Development splits
    per_fold_df, summary_df, oof_probs = evaluate_baseline_on_dev_folds(
        X_train_df=X_train_df,
        y_train=y_train,
        fold_ids=fold_ids,
    )
    per_fold_df.to_csv(OUTPUT_DIR / "dev_5fold_per_fold_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "dev_5fold_baseline_summary.csv", index=False)

    figure_paths = create_baseline_dev_figures(
        per_fold_df=per_fold_df,
        summary_df=summary_df,
        y_train=y_train,
        oof_probs=oof_probs,
    )

    report = build_report(
        eq_df=eq_df,
        summary_df=summary_df,
        figure_paths=figure_paths,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Tahap 04 (Baseline KNN — Training/Development Only) selesai.")
    print(summary_df[["variant", "accuracy_mean", "precision_yes_mean", "recall_yes_mean", "f1_yes_mean", "pr_auc_mean", "roc_auc_mean", "mcc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
