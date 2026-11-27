"""Stage 09: Explainability with LIME & Final Research Conclusions (DIFW-KNN).

Applies LIME (Local Interpretable Model-agnostic Explanations) to the locked
final DIFW-KNN model (E5, trained on 36,168 Training/Dev observations using
locked Stage 06 hyperparameters and Stage 07 learned feature weights) on the
Hold-out Test Set (N = 9,043):
  1. Verifies exact probability reproduction against Stage 07 predictions.
  2. Explains 8 representative Hold-out Test cases (TP recovered from E0, TP high,
     FP borderline, FP high, FN missed, FN borderline, TN clear, TN borderline).
  3. Aggregates local |LIME| feature attributions across 40 stratified test cases
     and measures alignment (Spearman rho) with global Triplet feature weights.
  4. Generates publication-ready figures and the final Stage 09 report with
     research conclusions (RM1-RM3, H1-H3), limitations, and future work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.difw_knn import DIFWKNNClassifier
from src.preprocessing import (
    DOMAIN_CATEGORICAL_COLS,
    DOMAIN_NUMERIC_COLS,
    DomainFeatureEngineer,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CV_DIR = PROJECT_ROOT / "outputs" / "cv_tuning"
FINAL_DIR = PROJECT_ROOT / "outputs" / "final_evaluation"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "explainability_lime"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "Laporan_Explainability_LIME_Conclusions_DIFW_KNN.md"

RANDOM_STATE = 42


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
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
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


def build_locked_e5_model(
    X_train_eng: pd.DataFrame,
    y_train: np.ndarray,
    selected_cfg: dict[str, Any],
    weights_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> DIFWKNNClassifier:
    """Reconstruct locked E5 model on engineered 16-feature space."""
    e5_cfg = selected_cfg["E5"]
    feature_order = numeric_cols + categorical_cols
    w_map = dict(zip(weights_df["feature"], weights_df["weight"]))
    fixed_weights = np.array([float(w_map[f]) for f in feature_order], dtype=np.float64)

    model = DIFWKNNClassifier(
        k=int(e5_cfg["k"]),
        neighbor_weights=str(e5_cfg["neighbor_weights"]),
        distance_power_q=float(e5_cfg["q"]),
        learn_feature_weights=False,
        fixed_feature_weights=fixed_weights,
        use_cost_sensitive=True,
        c_fp=float(e5_cfg["c_fp"]),
        c_fn=float(e5_cfg["c_fn"]),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        auto_engineer_pdays=False,
        batch_size=256,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_eng[feature_order], y_train)
    return model


def encode_for_lime(
    X_eng: pd.DataFrame,
    feature_order: list[str],
    categorical_cols: list[str],
    cat_mappings: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Encode engineered mixed-type DataFrame into numeric array for LimeTabularExplainer."""
    X_out = X_eng[feature_order].copy()
    mappings: dict[str, list[str]] = {} if cat_mappings is None else cat_mappings

    for col in categorical_cols:
        vals = X_out[col].astype(str)
        if cat_mappings is None:
            cats = sorted(vals.unique().tolist())
            mappings[col] = cats
        else:
            cats = mappings[col]
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        X_out[col] = vals.map(lambda v: cat_to_idx.get(v, 0)).astype(np.float64)

    for col in feature_order:
        if col not in categorical_cols:
            X_out[col] = X_out[col].astype(np.float64)

    return X_out.to_numpy(dtype=np.float64), mappings


def decode_from_lime(
    X_arr: np.ndarray,
    feature_order: list[str],
    categorical_cols: list[str],
    cat_mappings: dict[str, list[str]],
) -> pd.DataFrame:
    """Decode LIME numeric matrix back to engineered mixed-type DataFrame."""
    df = pd.DataFrame(X_arr, columns=feature_order)
    for col in categorical_cols:
        cats = cat_mappings[col]
        n_cats = len(cats)
        idx = np.clip(np.rint(df[col].to_numpy(dtype=np.float64)).astype(int), 0, n_cats - 1)
        df[col] = [cats[i] for i in idx]
    return df


def select_representative_cases(preds_df: pd.DataFrame) -> list[tuple[str, int]]:
    """Select 8 representative Hold-out Test indices across TP, FP, FN, TN."""
    y = preds_df["y_true"].to_numpy()
    p5 = preds_df["prob_e5"].to_numpy()
    yhat5 = preds_df["pred_e5"].to_numpy()
    yhat0 = preds_df["pred_e0"].to_numpy()

    cases: list[tuple[str, int]] = []

    # 1. TP recovered by E5 where E0 missed (FN in E0 -> TP in E5)
    rec_mask = (y == 1) & (yhat5 == 1) & (yhat0 == 0)
    rec_idx = np.where(rec_mask)[0]
    cases.append(("TP_recovered_from_E0", int(rec_idx[np.argmax(p5[rec_idx])])))

    # 2. TP high confidence
    tp_idx = np.where((y == 1) & (yhat5 == 1))[0]
    cases.append(("TP_high_conf", int(tp_idx[np.argmax(p5[tp_idx])])))

    # 3. FP borderline (just above tau=0.1667)
    fp_idx = np.where((y == 0) & (yhat5 == 1))[0]
    fp_border = fp_idx[np.argmin(np.abs(p5[fp_idx] - (1.0 / 6.0)))]
    cases.append(("FP_borderline", int(fp_border)))

    # 4. FP high confidence
    cases.append(("FP_high_prob", int(fp_idx[np.argmax(p5[fp_idx])])))

    # 5. FN borderline (just below tau=0.1667)
    fn_idx = np.where((y == 1) & (yhat5 == 0))[0]
    fn_border = fn_idx[np.argmin(np.abs(p5[fn_idx] - (1.0 / 6.0)))]
    cases.append(("FN_borderline", int(fn_border)))

    # 6. FN clear miss (lowest prob among actual YES)
    cases.append(("FN_clear_miss", int(fn_idx[np.argmin(p5[fn_idx])])))

    # 7. TN clear (prob = 0)
    tn_idx = np.where((y == 0) & (yhat5 == 0))[0]
    cases.append(("TN_clear", int(tn_idx[np.argmin(p5[tn_idx])])))

    # 8. TN borderline (just below tau=0.1667)
    tn_border = tn_idx[np.argmax(p5[tn_idx])]
    cases.append(("TN_borderline", int(tn_border)))

    return cases


def plot_representative_lime_cases(
    contrib_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot 4-panel horizontal bar chart of LIME local explanations."""
    configure_plot_style()
    selected_types = ["TP_recovered_from_E0", "FP_high_prob", "FN_clear_miss", "TN_clear"]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))

    for ax, ctype in zip(axes.ravel(), selected_types):
        sub = contrib_df[contrib_df["case_type"] == ctype].head(8).iloc[::-1]
        meta = summary_df[summary_df["case_type"] == ctype].iloc[0]
        vals = sub["lime_weight"].to_numpy(dtype=np.float64)
        labels = sub["feature_rule"].tolist()
        colors = ["#1E8449" if v > 0 else "#C0392B" for v in vals]

        ax.barh(labels, vals, color=colors, edgecolor="#2C3E50", linewidth=0.6)
        ax.axvline(0.0, color="#333333", linewidth=0.9)
        ax.set_title(
            f"{ctype} (idx={meta['test_index']}, y={meta['y_true']}, "
            f"P_e5={meta['prob_e5']:.3f}, pred_e5={meta['pred_e5']})"
        )
        ax.set_xlabel("LIME Local Contribution toward P(yes)")

    fig.suptitle(
        "Stage 09: LIME Local Explanations on Representative Hold-out Test Cases (E5 DIFW-KNN)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def plot_global_vs_local_alignment(align_df: pd.DataFrame, rho: float, p_val: float, out_path: Path) -> None:
    """Plot Global Triplet Weights vs Mean Local |LIME| Importance."""
    configure_plot_style()
    sub = align_df.sort_values("global_triplet_weight", ascending=True)
    y_pos = np.arange(len(sub))
    height = 0.38

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.barh(
        y_pos + height / 2.0,
        sub["global_triplet_weight_norm"],
        height=height,
        label="Global Triplet Weight (Normalized)",
        color="#2980B9",
        edgecolor="#2C3E50",
        linewidth=0.6,
    )
    ax.barh(
        y_pos - height / 2.0,
        sub["mean_abs_lime_norm"],
        height=height,
        label="Mean Local |LIME| Attribution (40 Test Cases, Normalized)",
        color="#E67E22",
        edgecolor="#2C3E50",
        linewidth=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["feature"])
    ax.set_xlabel("Normalized Relative Importance (Sum = 1.0)")
    ax.set_title(
        f"Global Triplet Feature Weights vs Mean Local LIME Importance (Spearman rho = {rho:.3f}, p = {p_val:.4e})"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_stage_09() -> dict[str, Any]:
    """Execute Stage 09 LIME Explainability and Final Conclusions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "train_split.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_split.csv")
    preds_df = pd.read_csv(FINAL_DIR / "holdout_predictions_all_models.csv")
    weights_df = pd.read_csv(FINAL_DIR / "final_full_train_feature_weights.csv")
    selected_cfg = json.loads((CV_DIR / "selected_hyperparameters.json").read_text(encoding="utf-8"))

    y_train = train_df["y"].to_numpy(dtype=np.int64)
    X_train_raw = train_df.drop(columns=["y"])
    X_test_raw = test_df.drop(columns=["y"])

    # Engineer features strictly on Training fit
    prep = DomainFeatureEngineer()
    X_train_eng = prep.fit_transform(X_train_raw)
    X_test_eng = prep.transform(X_test_raw)

    numeric_cols = list(DOMAIN_NUMERIC_COLS)
    categorical_cols = list(DOMAIN_CATEGORICAL_COLS)
    feature_order = numeric_cols + categorical_cols

    e5_model = build_locked_e5_model(
        X_train_eng, y_train, selected_cfg, weights_df, numeric_cols, categorical_cols
    )

    # Verify exact reproduction against Stage 07 Hold-out predictions
    prob_e5_repro = e5_model.predict_proba(X_test_eng[feature_order])[:, 1]
    max_repro_diff = float(np.max(np.abs(prob_e5_repro - preds_df["prob_e5"].to_numpy())))
    if max_repro_diff > 1e-5:
        raise RuntimeError(f"E5 reproduction mismatch: max_diff={max_repro_diff:.6e}")

    # Encode for LIME
    X_train_lime, cat_mappings = encode_for_lime(X_train_eng, feature_order, categorical_cols)
    X_test_lime, _ = encode_for_lime(X_test_eng, feature_order, categorical_cols, cat_mappings=cat_mappings)

    cat_indices = [feature_order.index(c) for c in categorical_cols]
    cat_names = {feature_order.index(c): cat_mappings[c] for c in categorical_cols}

    explainer = LimeTabularExplainer(
        training_data=X_train_lime,
        feature_names=feature_order,
        class_names=["no", "yes"],
        categorical_features=cat_indices,
        categorical_names=cat_names,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    def lime_predict_fn(X_arr: np.ndarray) -> np.ndarray:
        df_dec = decode_from_lime(X_arr, feature_order, categorical_cols, cat_mappings)
        return e5_model.predict_proba(df_dec)

    # 1. Explain 8 Representative Cases
    rep_cases = select_representative_cases(preds_df)
    summary_rows = []
    contrib_rows = []

    for ctype, idx in rep_cases:
        exp = explainer.explain_instance(
            X_test_lime[idx],
            lime_predict_fn,
            num_features=len(feature_order),
            num_samples=500,
            labels=(1,),
        )
        local_list = exp.as_list(label=1)
        top_pos = [f"{r} ({w:+.4f})" for r, w in local_list if w > 0][:3]
        top_neg = [f"{r} ({w:+.4f})" for r, w in local_list if w < 0][:3]
        summary_rows.append(
            {
                "case_type": ctype,
                "test_index": idx,
                "y_true": int(preds_df.loc[idx, "y_true"]),
                "prob_e0": round(float(preds_df.loc[idx, "prob_e0"]), 4),
                "pred_e0": int(preds_df.loc[idx, "pred_e0"]),
                "prob_e5": round(float(preds_df.loc[idx, "prob_e5"]), 4),
                "pred_e5": int(preds_df.loc[idx, "pred_e5"]),
                "lime_r2_fidelity": round(float(exp.score), 4),
                "top_positive_drivers": "; ".join(top_pos),
                "top_negative_drivers": "; ".join(top_neg),
            }
        )
        for rank, (rule, weight) in enumerate(local_list, start=1):
            contrib_rows.append(
                {
                    "case_type": ctype,
                    "test_index": idx,
                    "rank": rank,
                    "feature_rule": rule,
                    "lime_weight": round(float(weight), 6),
                    "abs_lime_weight": round(abs(float(weight)), 6),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    contrib_df = pd.DataFrame(contrib_rows)
    summary_df.to_csv(OUTPUT_DIR / "lime_case_studies_summary.csv", index=False)
    contrib_df.to_csv(OUTPUT_DIR / "lime_case_studies_contributions.csv", index=False)

    # 2. Global-vs-Local Alignment across 40 stratified Hold-out Test cases (20 yes, 20 no)
    rng = np.random.default_rng(RANDOM_STATE)
    y_test = preds_df["y_true"].to_numpy()
    pos_sample = rng.choice(np.where(y_test == 1)[0], size=20, replace=False)
    neg_sample = rng.choice(np.where(y_test == 0)[0], size=20, replace=False)
    sample_indices = np.concatenate([pos_sample, neg_sample])

    abs_lime_acc = np.zeros(len(feature_order), dtype=np.float64)
    fidelities = []
    for idx in sample_indices:
        exp = explainer.explain_instance(
            X_test_lime[int(idx)],
            lime_predict_fn,
            num_features=len(feature_order),
            num_samples=300,
            labels=(1,),
        )
        fidelities.append(float(exp.score))
        for feat_idx, w in exp.local_exp[1]:
            abs_lime_acc[int(feat_idx)] += abs(float(w))

    mean_abs_lime = abs_lime_acc / float(len(sample_indices))
    mean_abs_lime_norm = mean_abs_lime / float(np.sum(mean_abs_lime))
    w_map = dict(zip(weights_df["feature"], weights_df["weight"]))
    global_w = np.array([float(w_map[f]) for f in feature_order], dtype=np.float64)
    global_w_norm = global_w / float(np.sum(global_w))

    rho, p_rho = stats.spearmanr(global_w_norm, mean_abs_lime_norm)

    align_df = pd.DataFrame(
        {
            "feature": feature_order,
            "global_triplet_weight": np.round(global_w, 6),
            "global_triplet_weight_norm": np.round(global_w_norm, 6),
            "mean_abs_lime": np.round(mean_abs_lime, 6),
            "mean_abs_lime_norm": np.round(mean_abs_lime_norm, 6),
        }
    ).sort_values("global_triplet_weight", ascending=False).reset_index(drop=True)
    align_df.to_csv(OUTPUT_DIR / "lime_global_vs_local_alignment.csv", index=False)

    # 3. Figures
    plot_representative_lime_cases(
        contrib_df, summary_df, FIGURES_DIR / "01_lime_representative_cases.png"
    )
    plot_global_vs_local_alignment(
        align_df, float(rho), float(p_rho), FIGURES_DIR / "02_lime_vs_triplet_weights.png"
    )

    # 4. Write Final Stage 09 Report & Conclusions
    mean_fidelity = float(np.mean(fidelities))
    report_lines = [
        "# Laporan Tahap 09: Explainability with LIME & Kesimpulan Akhir Penelitian (DIFW-KNN)",
        "",
        f"Verifikasi reproduksi model `E5` pada Hold-out Test Set (`N = {len(X_test_raw):,}`): selisih probabilitas maksimum terhadap Tahap 07 = `{max_repro_diff:.3e}` (identik).",
        "",
        "## 1. Interpretabilitas Lokal dengan LIME (8 Kasus Representatif)",
        "",
        "Tabel berikut merangkum penjelasan lokal LIME (`num_samples = 500`) pada kasus nyata Hold-out Test Set (`TP`, `FP`, `FN`, `TN`), termasuk kasus `TP_recovered_from_E0` di mana `E5` berhasil mendeteksi nasabah berlangganan yang gagal ditangkap `E0`:",
        "",
        df_to_markdown_table(summary_df),
        "",
        "## 2. Konsistensi Global (Triplet Weights) vs Lokal (Mean |LIME| pada 40 Sampel)",
        "",
        f"- **Rata-rata Local Fidelity ($R^2$ LIME pada 40 sampel):** `{mean_fidelity:.4f}`",
        f"- **Korelasi Peringkat Global vs Lokal (Spearman $\\rho$):** `{float(rho):.4f}` ($p = {float(p_rho):.4e}$)",
        "",
        df_to_markdown_table(align_df),
        "",
        "## 3. Kesimpulan Akhir Penelitian",
        "",
        "### Jawaban Rumusan Masalah (RM1–RM3) & Hipotesis (H1–H3)",
        "1. **RM1 & H1 (Penanganan Data Campuran & Sentinel `pdays = -1`):** Dekomposisi `pdays = -1` (81,74% data) menjadi `previously_contacted` dan `pdays_recency` (`M_pos_train = 195.0`) bersama metrik jarak campuran berbasis rentang kuantil $Q_{95}-Q_{05}$ (`E2`) terbukti menaikkan `PR-AUC` dari `0.2921` (`E0`) menjadi `0.4160` dan `ROC-AUC` dari `0.7014` menjadi `0.7733`.",
        "2. **RM2 & H2 (Pembelajaran Bobot Fitur Berbasis Triplet pada Data Imbalanced Multimodal):** Strategi *Hybrid 50% Global + 50% Local-Context Triplet Sampling* mengatasi kegagalan sampling global acak dan menghasilkan bobot fitur stabil (`5-seed Spearman = 0.9906`). Fitur paling dominan adalah `month` (`0.3073`), `poutcome` (`0.1030`), `day` (`0.0971`), `age` (`0.0913`), dan `previously_contacted` (`0.0866`), yang menaikkan `PR-AUC` ke `0.4479` dan `ROC-AUC` ke `0.7888` (`E3`/`E5`, DeLong $p_{\\text{Holm}} = 0.0399$ vs `E2`).",
        "3. **RM3 & H3 (Aturan Keputusan *Cost-Sensitive* 5:1 & Integrasi Penuh `E5`):** Mengintegrasikan aturan Bayes *cost-sensitive* ($\\tau = 0.1667$) pada `E5` (DIFW-KNN Lengkap) menaikkan `Recall` dari `20.13%` (`213 TP`) menjadi `53.02%` (`561 TP`), `F1-Score` dari `0.2969` menjadi `0.4748`, `MCC` dari `0.2907` menjadi `0.3998`, dan menekan total biaya mis-klasifikasi sebesar **26,43%** (`4.389 -> 3.229`, paired bootstrap $p_{\\text{Holm}} < 0.0001$).",
        "",
        "### Keterbatasan & Saran Penelitian Lanjutan",
        "- **Komputasi Inferensi:** Karena KNN bersifat *lazy learner*, waktu prediksi `E5` pada `9.043` data uji adalah `7.779` detik (masih sangat layak untuk *batch scoring* harian, namun dapat dipercepat dengan *prototype condensation* atau *approximate nearest neighbors* untuk skenario *real-time*).",
        "- **Asumsi Rasio Biaya Tetap (5:1):** Nilai $C_{FN}:C_{FP} = 5:1$ dapat disesuaikan secara dinamis oleh manajemen bank mengikuti perubahan margin suku bunga deposito dan biaya operasional *call center*.",
        "- **Generalisasi Domain:** Kerangka DIFW-KNN dapat diuji pada dataset tabular campuran *imbalanced* lain (seperti *credit scoring* atau *churn prediction*).",
        "",
        "## 4. Artefak Grafik Tahap 09",
        "- `outputs/explainability_lime/figures/01_lime_representative_cases.png`",
        "- `outputs/explainability_lime/figures/02_lime_vs_triplet_weights.png`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "max_repro_diff": max_repro_diff,
        "mean_fidelity": mean_fidelity,
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "summary_df": summary_df,
        "align_df": align_df,
    }


def main() -> None:
    res = run_stage_09()
    print(f"Stage 09 complete. E5 reproduction max_diff = {res['max_repro_diff']:.3e}")
    print(
        f"40-sample LIME mean R2 fidelity = {res['mean_fidelity']:.4f}, "
        f"Global-vs-Local Spearman rho = {res['spearman_rho']:.4f} (p = {res['spearman_p']:.4e})"
    )
    print("\nRepresentative Cases Summary:")
    print(
        res["summary_df"][
            ["case_type", "y_true", "prob_e0", "pred_e0", "prob_e5", "pred_e5", "lime_r2_fidelity"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
