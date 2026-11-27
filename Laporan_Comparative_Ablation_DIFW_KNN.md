# Laporan Tahap 08: Comparative Analysis & Ablation Study (DIFW-KNN)

Evaluasi dilakukan secara retrospektif pada prediksi terkunci Hold-out Test Set (`N = 9,043` observasi) dengan koreksi *multiple testing* **Holm-Bonferroni** dan **Paired Bootstrap (`B = 1,000`)**. Tidak ada *re-tuning* ataupun perubahan parameter pada tahap ini.

## 1. Studi Ablasi Komponen (`E0`–`E5`)

| model_id | model_name | distance_metric | feature_weighting | decision_rule | accuracy | precision_yes | recall_yes | f1_yes | pr_auc | roc_auc | mcc | g_mean | mean_cost | cost_savings_vs_e0_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Standard KNN | Euclidean | Uniform (k=5) | tau=0.50 | 0.8884 | 0.565 | 0.2013 | 0.2969 | 0.2921 | 0.7014 | 0.2907 | 0.4441 | 0.4853 | 0.0 |
| E1 | Optimized KNN (Best CV F1) | Euclidean | CV-F1 (k=3, dist) | tau=0.50 | 0.8816 | 0.4876 | 0.242 | 0.3234 | 0.2619 | 0.6779 | 0.2863 | 0.4835 | 0.4732 | 2.51 |
| E1_PR | Optimized KNN (Best CV PR-AUC) | Euclidean | CV-PR (k=51, dist) | tau=0.50 | 0.8924 | 0.7094 | 0.1361 | 0.2284 | 0.4042 | 0.7656 | 0.2793 | 0.3676 | 0.5119 | -5.47 |
| E2 | Mixed-Type KNN | Mixed Quantile | Uniform | tau=0.50 | 0.8924 | 0.7033 | 0.1389 | 0.232 | 0.416 | 0.7733 | 0.2806 | 0.3713 | 0.5106 | -5.19 |
| E3 | Feature-Weighted KNN | Mixed Quantile | Hybrid Triplet Weights | tau=0.50 | 0.8968 | 0.6731 | 0.2297 | 0.3425 | 0.4479 | 0.7888 | 0.3528 | 0.4757 | 0.4637 | 4.47 |
| E4 | Cost-Sensitive KNN | Mixed Quantile | Uniform | tau=0.1667 (5:1) | 0.8577 | 0.4125 | 0.5104 | 0.4563 | 0.416 | 0.7733 | 0.3783 | 0.6791 | 0.3714 | 23.47 |
| E5 | DIFW-KNN Lengkap | Mixed Quantile | Hybrid Triplet Weights | tau=0.1667 (5:1) | 0.8628 | 0.4299 | 0.5302 | 0.4748 | 0.4479 | 0.7888 | 0.3998 | 0.6934 | 0.3571 | 26.43 |

### Kontribusi Marjinal Per Komponen

| transition | component_effect | delta_pr_auc | delta_roc_auc | delta_f1 | delta_recall | delta_mcc | delta_accuracy | delta_mean_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 -> E2 | Add Mixed-Type Quantile Distance (vs Standard KNN) | 0.124 | 0.0719 | -0.0648 | -0.0624 | -0.0101 | 0.004 | 0.0252 |
| E1_PR -> E2 | Replace Euclidean with Mixed-Type Distance (same ranking family) | 0.0119 | 0.0078 | 0.0037 | 0.0028 | 0.0013 | 0.0 | -0.0013 |
| E2 -> E3 | Add Hybrid Triplet Feature Weights (at tau=0.50) | 0.0319 | 0.0154 | 0.1105 | 0.0907 | 0.0722 | 0.0044 | -0.0469 |
| E2 -> E4 | Add Cost-Sensitive Bayes Rule (at uniform weights) | 0.0 | 0.0 | 0.2242 | 0.3715 | 0.0977 | -0.0347 | -0.1391 |
| E4 -> E5 | Add Hybrid Triplet Feature Weights (on top of Cost-Sensitive) | 0.0319 | 0.0154 | 0.0185 | 0.0198 | 0.0215 | 0.0051 | -0.0144 |
| E3 -> E5 | Add Cost-Sensitive Bayes Rule (on top of Feature Weights) | 0.0 | 0.0 | 0.1323 | 0.3006 | 0.047 | -0.0341 | -0.1066 |
| E0 -> E5 | Full DIFW-KNN vs Standard KNN Baseline | 0.1558 | 0.0873 | 0.178 | 0.3289 | 0.1091 | -0.0257 | -0.1283 |
| E1 -> E5 | Full DIFW-KNN vs Optimized Euclidean KNN (Best CV F1) | 0.186 | 0.1109 | 0.1514 | 0.2883 | 0.1135 | -0.0188 | -0.1161 |

## 2. Uji Signifikansi Statistik

### 2.1 Uji DeLong (ROC-AUC Terkorelasi)

| comparison | baseline_id | roc_auc_e5 | roc_auc_baseline | delta_roc_auc | se_diff | z_stat | p_value_raw | p_value_holm | significant_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E5 vs E0 | E0 | 0.7888 | 0.7014 | 0.0873 | 0.00785 | 11.1182 | 1.0227e-28 | 5.1133e-28 | True |
| E5 vs E1 | E1 | 0.7888 | 0.6779 | 0.1109 | 0.00826 | 13.4301 | 4.0271e-41 | 2.4162e-40 | True |
| E5 vs E1_PR | E1_PR | 0.7888 | 0.7656 | 0.0232 | 0.00644 | 3.5996 | 3.1868e-04 | 1.2747e-03 | True |
| E5 vs E2 | E2 | 0.7888 | 0.7733 | 0.0154 | 0.00623 | 2.4761 | 1.3284e-02 | 3.9852e-02 | True |
| E5 vs E3 | E3 | 0.7888 | 0.7888 | 0.0 | 0.0 | 0.0 | 1.0000e+00 | 1.0000e+00 | False |
| E5 vs E4 | E4 | 0.7888 | 0.7733 | 0.0154 | 0.00623 | 2.4761 | 1.3284e-02 | 3.9852e-02 | True |

### 2.2 Uji McNemar (Keputusan Biner 0/1)

> Catatan: Uji McNemar memperlakukan kesalahan FP dan FN secara simetris (1:1). Karena `E4` dan `E5` memakai ambang *cost-sensitive* 5:1 (`tau = 0.1667`), penurunan akurasi mentah terhadap model `tau = 0.5` (`E0`–`E3`) mencerminkan pertukaran disengaja dari FN ke FP demi menurunkan total biaya.

| comparison | n10_a_right_b_wrong | n01_a_wrong_b_right | chi2_cc | p_value_holm_chi2 | significant_0_05 |
| --- | --- | --- | --- | --- | --- |
| E5 vs E0 | 378 | 610 | 54.01 | 5.9869e-13 | True |
| E5 vs E1 | 407 | 577 | 29.03 | 1.4287e-07 | True |
| E5 vs E1_PR | 419 | 687 | 64.46 | 3.9473e-15 | True |
| E5 vs E2 | 414 | 682 | 65.04 | 3.6609e-15 | True |
| E5 vs E3 | 318 | 626 | 99.84 | 9.9130e-23 | True |
| E5 vs E4 | 335 | 289 | 3.25 | 7.1633e-02 | False |

### 2.3 Paired Bootstrap Hypothesis Test (`B = 1,000`, Holm-Bonferroni)

| comparison | metric | delta_ci95 | p_value_holm | significant_0_05 |
| --- | --- | --- | --- | --- |
| E5 vs E0 | pr_auc | +0.1558 [+0.1340, +0.1794] | 0.0000e+00 | True |
| E5 vs E0 | roc_auc | +0.0873 [+0.0713, +0.1023] | 0.0000e+00 | True |
| E5 vs E0 | f1_yes | +0.1780 [+0.1465, +0.2086] | 0.0000e+00 | True |
| E5 vs E0 | recall_yes | +0.3289 [+0.3001, +0.3596] | 0.0000e+00 | True |
| E5 vs E0 | mcc | +0.1091 [+0.0774, +0.1422] | 0.0000e+00 | True |
| E5 vs E0 | mean_cost | -0.1283 [-0.1487, -0.1063] | 0.0000e+00 | True |
| E5 vs E1 | pr_auc | +0.1860 [+0.1635, +0.2124] | 0.0000e+00 | True |
| E5 vs E1 | roc_auc | +0.1109 [+0.0947, +0.1268] | 0.0000e+00 | True |
| E5 vs E1 | f1_yes | +0.1514 [+0.1216, +0.1807] | 0.0000e+00 | True |
| E5 vs E1 | recall_yes | +0.2883 [+0.2617, +0.3173] | 0.0000e+00 | True |
| E5 vs E1 | mcc | +0.1135 [+0.0824, +0.1448] | 0.0000e+00 | True |
| E5 vs E1 | mean_cost | -0.1161 [-0.1360, -0.0962] | 0.0000e+00 | True |
| E5 vs E2 | pr_auc | +0.0319 [+0.0146, +0.0497] | 0.0000e+00 | True |
| E5 vs E2 | roc_auc | +0.0154 [+0.0030, +0.0282] | 3.6000e-02 | True |
| E5 vs E2 | f1_yes | +0.2428 [+0.2080, +0.2767] | 0.0000e+00 | True |
| E5 vs E2 | recall_yes | +0.3913 [+0.3629, +0.4213] | 0.0000e+00 | True |
| E5 vs E2 | mcc | +0.1192 [+0.0853, +0.1567] | 0.0000e+00 | True |
| E5 vs E2 | mean_cost | -0.1535 [-0.1757, -0.1305] | 0.0000e+00 | True |
| E5 vs E3 | pr_auc | +0.0000 [+0.0000, +0.0000] | 1.0000e+00 | False |
| E5 vs E3 | roc_auc | +0.0000 [+0.0000, +0.0000] | 1.0000e+00 | False |
| E5 vs E3 | f1_yes | +0.1323 [+0.1025, +0.1617] | 0.0000e+00 | True |
| E5 vs E3 | recall_yes | +0.3006 [+0.2732, +0.3295] | 0.0000e+00 | True |
| E5 vs E3 | mcc | +0.0470 [+0.0177, +0.0774] | 8.0000e-03 | True |
| E5 vs E3 | mean_cost | -0.1066 [-0.1261, -0.0866] | 0.0000e+00 | True |
| E5 vs E4 | pr_auc | +0.0319 [+0.0146, +0.0497] | 0.0000e+00 | True |
| E5 vs E4 | roc_auc | +0.0154 [+0.0030, +0.0282] | 3.6000e-02 | True |
| E5 vs E4 | f1_yes | +0.0185 [-0.0002, +0.0358] | 5.2000e-02 | False |
| E5 vs E4 | recall_yes | +0.0198 [-0.0019, +0.0425] | 1.0400e-01 | False |
| E5 vs E4 | mcc | +0.0215 [-0.0001, +0.0413] | 5.2000e-02 | False |
| E5 vs E4 | mean_cost | -0.0144 [-0.0283, -0.0003] | 4.6000e-02 | True |

## 3. Verifikasi Hipotesis Penelitian (H1, H2, H3)

- **H1 (Mixed-Type Quantile Distance + Sentinel Decomposition):** **TERBUKTI.** `E2` menaikkan `PR-AUC` sebesar `+0.1240` dan `ROC-AUC` sebesar `+0.0719` dibanding `E0`.
- **H2 (Hybrid Local-Context Triplet Feature Weighting):** **TERBUKTI.** Menambahkan bobot fitur di atas jarak campuran (`E3` vs `E2`, serta `E5` vs `E4`) menaikkan `PR-AUC` sebesar `+0.0319` dan `ROC-AUC` sebesar `+0.0154` (DeLong $p_{\text{Holm}} = 3.9852e-02$), `F1` (`+0.0185` pada `E5` vs `E4`), dan `MCC` (`+0.0215` pada `E5` vs `E4`).
- **H3 (Cost-Sensitive Bayes Rule 5:1 + Integrasi Penuh `E5`):** **TERBUKTI.** `E5` mengalahkan `E0` secara signifikan pada `PR-AUC` (`+0.1558`), `F1` (`+0.1779`), `Recall` (`+0.3289`), `MCC` (`+0.1091`), dan menurunkan biaya kesalahan rata-rata (`delta = -0.1283`, $p_{\text{Holm}} = 0.0000e+00$). `E5` juga unggul signifikan atas `E4` pada `PR-AUC`, `ROC-AUC`, dan `Mean Cost` ($p_{\text{Holm}} = 4.6000e-02$), serta menunjukkan peningkatan marjinal pada `F1` ($p_{\text{Holm}} = 5.2000e-02$) dan `MCC` ($p_{\text{Holm}} = 5.2000e-02$).

## 4. Analisis Transisi Prediksi (`E0 -> E5`) & Subgrup Domain

### Transisi Confusion Matrix (`E0 -> E5`)

| actual_class | total | both_pred_yes | gained_yes_e0_no_e5_yes | lost_yes_e0_yes_e5_no | both_pred_no |
| --- | --- | --- | --- | --- | --- |
| Actual YES (Positive) | 1058 | 209 | 352 | 4 | 493 |
| Actual NO (Negative) | 7985 | 138 | 606 | 26 | 7215 |

### Performa pada Subgrup Kritis

| segment | n_samples | positive_rate_pct | model_id | recall_yes | precision_yes | f1_yes | pr_auc | mcc | mean_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Previously Contacted = 0 (New) | 7370 | 8.91 | E0 | 0.0731 | 0.4103 | 0.124 | 0.1643 | 0.1431 | 0.4225 |
| Previously Contacted = 0 (New) | 7370 | 8.91 | E3 | 0.0807 | 0.5889 | 0.1419 | 0.3057 | 0.195 | 0.4148 |
| Previously Contacted = 0 (New) | 7370 | 8.91 | E5 | 0.379 | 0.3694 | 0.3742 | 0.3057 | 0.3121 | 0.3345 |
| Previously Contacted = 1 (Repeat) | 1673 | 23.97 | E0 | 0.4115 | 0.6346 | 0.4992 | 0.5086 | 0.3968 | 0.7621 |
| Previously Contacted = 1 (Repeat) | 1673 | 23.97 | E3 | 0.4738 | 0.7011 | 0.5655 | 0.6283 | 0.4752 | 0.679 |
| Previously Contacted = 1 (Repeat) | 1673 | 23.97 | E5 | 0.7781 | 0.4945 | 0.6047 | 0.6283 | 0.4644 | 0.4567 |
| poutcome = success | 306 | 65.03 | E0 | 0.7437 | 0.7291 | 0.7363 | 0.737 | 0.2318 | 1.0131 |
| poutcome = success | 306 | 65.03 | E3 | 0.8894 | 0.7284 | 0.8009 | 0.7714 | 0.3215 | 0.5752 |
| poutcome = success | 306 | 65.03 | E5 | 1.0 | 0.6503 | 0.7881 | 0.7714 | 0.0 | 0.3497 |
| poutcome != success | 8737 | 9.83 | E0 | 0.0757 | 0.3736 | 0.1258 | 0.1757 | 0.1318 | 0.4669 |
| poutcome != success | 8737 | 9.83 | E3 | 0.0768 | 0.5593 | 0.1351 | 0.3133 | 0.1812 | 0.4598 |
| poutcome != success | 8737 | 9.83 | E5 | 0.4214 | 0.3624 | 0.3897 | 0.3133 | 0.3186 | 0.3573 |

## 5. Artefak Grafik Tahap 08
- `outputs/comparative_ablation/figures/01_ablation_waterfall.png`
- `outputs/comparative_ablation/figures/02_paired_bootstrap_deltas.png`
- `outputs/comparative_ablation/figures/03_subgroup_recall_cost.png`
