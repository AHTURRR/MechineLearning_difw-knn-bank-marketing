# LAPORAN TAHAP 07 — FINAL MODEL EVALUATION (`HOLD-OUT TEST DATA`)

**Skrip CLI:** [`src/07_final_model_evaluation.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/07_final_model_evaluation.py)  
**Notebook:** [`notebooks/07_Final_Model_Evaluation.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/07_Final_Model_Evaluation.ipynb)  
**Protokol Evaluasi:** Seluruh model dilatih pada `36.168` baris `Training / Development Data` menggunakan *hyperparameter* terkunci dari Tahap 06 ([`selected_hyperparameters.json`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/outputs/cv_tuning/selected_hyperparameters.json)), kemudian dievaluasi **satu kali** pada `9.043` baris **`Hold-out Test Data`** (`7.985` `no` dan `1.058` `yes`).

---

## 1. Tabel Performa Final pada Hold-out Test Set (`9.043` Observasi)

| model_id | model_name | accuracy | precision_yes | recall_yes | f1_yes | pr_auc | roc_auc | mcc | g_mean | mean_cost | cost_reduction_vs_e0_pct | fit_sec | predict_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Standard KNN | 0.8884 | 0.565 | 0.2013 | 0.2969 | 0.2921 | 0.7014 | 0.2907 | 0.4441 | 0.4853 | 0.0 | 0.377 | 2.573 |
| E1 | Optimized KNN (Best CV F1) | 0.8816 | 0.4876 | 0.242 | 0.3234 | 0.2619 | 0.6779 | 0.2863 | 0.4835 | 0.4732 | 2.51 | 0.378 | 0.617 |
| E1_PR | Optimized KNN (Best CV PR-AUC) | 0.8924 | 0.7094 | 0.1361 | 0.2284 | 0.4042 | 0.7656 | 0.2793 | 0.3676 | 0.5119 | -5.47 | 0.377 | 0.727 |
| E2 | Mixed-Type KNN | 0.8924 | 0.7033 | 0.1389 | 0.232 | 0.416 | 0.7733 | 0.2806 | 0.3713 | 0.5106 | -5.19 | 0.193 | 7.831 |
| E3 | Feature-Weighted KNN | 0.8968 | 0.6731 | 0.2297 | 0.3425 | 0.4479 | 0.7888 | 0.3528 | 0.4757 | 0.4637 | 4.47 | 1.978 | 7.779 |
| E4 | Cost-Sensitive KNN | 0.8577 | 0.4125 | 0.5104 | 0.4563 | 0.416 | 0.7733 | 0.3783 | 0.6791 | 0.3714 | 23.47 | 0.193 | 7.831 |
| E5 | DIFW-KNN Lengkap | 0.8628 | 0.4299 | 0.5302 | 0.4748 | 0.4479 | 0.7888 | 0.3998 | 0.6934 | 0.3571 | 26.43 | 1.978 | 7.779 |

---

## 2. Interval Kepercayaan Bootstrap Berpasangan 95% (`n_boot = 1.000`)

| model_id | model_name | accuracy | precision_yes | recall_yes | f1_yes | pr_auc | roc_auc | mcc | g_mean | mean_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Standard KNN | 0.8884 [0.8823, 0.8946] | 0.5650 [0.5153, 0.6121] | 0.2013 [0.1775, 0.2247] | 0.2969 [0.2658, 0.3257] | 0.2921 [0.2651, 0.3184] | 0.7014 [0.6858, 0.7186] | 0.2907 [0.2584, 0.3204] | 0.4441 [0.4170, 0.4692] | 0.4853 [0.4577, 0.5143] |
| E1 | Optimized KNN (Best CV F1) | 0.8816 [0.8750, 0.8875] | 0.4876 [0.4465, 0.5269] | 0.2420 [0.2167, 0.2671] | 0.3234 [0.2934, 0.3502] | 0.2619 [0.2398, 0.2866] | 0.6779 [0.6626, 0.6941] | 0.2863 [0.2550, 0.3145] | 0.4835 [0.4575, 0.5079] | 0.4732 [0.4460, 0.5019] |
| E1_PR | Optimized KNN (Best CV PR-AUC) | 0.8924 [0.8862, 0.8983] | 0.7094 [0.6473, 0.7747] | 0.1361 [0.1151, 0.1565] | 0.2284 [0.1970, 0.2591] | 0.4042 [0.3729, 0.4383] | 0.7656 [0.7478, 0.7827] | 0.2793 [0.2460, 0.3135] | 0.3676 [0.3380, 0.3941] | 0.5119 [0.4825, 0.5431] |
| E2 | Mixed-Type KNN | 0.8924 [0.8861, 0.8985] | 0.7033 [0.6417, 0.7647] | 0.1389 [0.1183, 0.1604] | 0.2320 [0.2010, 0.2633] | 0.4160 [0.3860, 0.4490] | 0.7733 [0.7560, 0.7895] | 0.2806 [0.2473, 0.3131] | 0.3713 [0.3427, 0.3990] | 0.5106 [0.4806, 0.5410] |
| E3 | Feature-Weighted KNN | 0.8968 [0.8907, 0.9027] | 0.6731 [0.6237, 0.7185] | 0.2297 [0.2054, 0.2554] | 0.3425 [0.3118, 0.3734] | 0.4479 [0.4174, 0.4811] | 0.7888 [0.7720, 0.8056] | 0.3528 [0.3207, 0.3835] | 0.4757 [0.4497, 0.5013] | 0.4637 [0.4351, 0.4928] |
| E4 | Cost-Sensitive KNN | 0.8577 [0.8508, 0.8644] | 0.4125 [0.3864, 0.4390] | 0.5104 [0.4794, 0.5406] | 0.4563 [0.4299, 0.4809] | 0.4160 [0.3860, 0.4490] | 0.7733 [0.7560, 0.7895] | 0.3783 [0.3507, 0.4061] | 0.6791 [0.6594, 0.6991] | 0.3714 [0.3495, 0.3950] |
| E5 | DIFW-KNN Lengkap | 0.8628 [0.8558, 0.8694] | 0.4299 [0.4006, 0.4564] | 0.5302 [0.5000, 0.5596] | 0.4748 [0.4483, 0.4981] | 0.4479 [0.4174, 0.4811] | 0.7888 [0.7720, 0.8056] | 0.3998 [0.3711, 0.4250] | 0.6934 [0.6733, 0.7123] | 0.3571 [0.3347, 0.3805] |

---

## 3. Confusion Matrix & Biaya Kesalahan (`C_FP = 1.0, C_FN = 5.0`)

| model_id | model_name | threshold | tn | fp | fn | tp | total_cost | mean_cost | cost_reduction_vs_e0_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Standard KNN | 0.5 | 7821 | 164 | 845 | 213 | 4389.0 | 0.4853 | 0.0 |
| E1 | Optimized KNN (Best CV F1) | 0.5 | 7716 | 269 | 802 | 256 | 4279.0 | 0.4732 | 2.51 |
| E1_PR | Optimized KNN (Best CV PR-AUC) | 0.5 | 7926 | 59 | 914 | 144 | 4629.0 | 0.5119 | -5.47 |
| E2 | Mixed-Type KNN | 0.5 | 7923 | 62 | 911 | 147 | 4617.0 | 0.5106 | -5.19 |
| E3 | Feature-Weighted KNN | 0.5 | 7867 | 118 | 815 | 243 | 4193.0 | 0.4637 | 4.47 |
| E4 | Cost-Sensitive KNN | 0.166667 | 7216 | 769 | 518 | 540 | 3359.0 | 0.3714 | 23.47 |
| E5 | DIFW-KNN Lengkap | 0.166667 | 7241 | 744 | 497 | 561 | 3229.0 | 0.3571 | 26.43 |

---

## 4. Bobot Fitur Final `DIFW-KNN` (Dilatih pada Seluruh `36.168` Data Training)

| feature | weight | uniform_weight | relative_importance | weight_pct |
| --- | --- | --- | --- | --- |
| month | 0.3073 | 0.0625 | 4.91684 | 30.73026 |
| poutcome | 0.10295 | 0.0625 | 1.64727 | 10.29545 |
| day | 0.0971 | 0.0625 | 1.5536 | 9.70999 |
| age | 0.09125 | 0.0625 | 1.46002 | 9.12515 |
| previously_contacted | 0.08656 | 0.0625 | 1.38503 | 8.65643 |
| campaign | 0.06682 | 0.0625 | 1.06906 | 6.68163 |
| balance | 0.0624 | 0.0625 | 0.99845 | 6.24029 |
| pdays_recency | 0.05691 | 0.0625 | 0.91051 | 5.69066 |
| contact | 0.05078 | 0.0625 | 0.81254 | 5.07835 |
| housing | 0.03404 | 0.0625 | 0.54456 | 3.40353 |
| previous | 0.02485 | 0.0625 | 0.39766 | 2.48535 |
| marital | 0.00819 | 0.0625 | 0.13097 | 0.81856 |
| loan | 0.00577 | 0.0625 | 0.09235 | 0.57721 |
| job | 0.00507 | 0.0625 | 0.08114 | 0.50713 |
| education | 0.0 | 0.0625 | 0.0 | 0.0 |
| default | 0.0 | 0.0625 | 0.0 | 0.0 |

> **Ringkasan Temuan Evaluasi Final (`E5` vs `E0` pada Hold-out Test):**
> - **`PR-AUC`:** Meningkat dari `0.2921` (`E0`) menjadi **`0.4479`** (`E5`).
> - **`ROC-AUC`:** Meningkat dari `0.7014` (`E0`) menjadi **`0.7888`** (`E5`).
> - **`F1 (yes)`:** Meningkat dari `0.2969` (`E0`) menjadi **`0.4748`** (`E5`).
> - **`MCC`:** Meningkat dari `0.2907` (`E0`) menjadi **`0.3998`** (`E5`).
> - **`Recall (yes)` & `G-Mean`:** `Recall` melonjak dari `0.2013` menjadi **`0.5302`** (`G-Mean` dari `0.4441` menjadi **`0.6934`**), menurunkan total biaya kesalahan sebesar **`26.43%`**.

---

## 5. Artefak Visual

- [01_holdout_roc_and_pr_curves.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/final_evaluation/figures/01_holdout_roc_and_pr_curves.png) (`outputs/final_evaluation/figures/01_holdout_roc_and_pr_curves.png`)
- [02_holdout_metrics_with_95ci.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/final_evaluation/figures/02_holdout_metrics_with_95ci.png) (`outputs/final_evaluation/figures/02_holdout_metrics_with_95ci.png`)
- [03_holdout_confusion_matrices_grid.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/final_evaluation/figures/03_holdout_confusion_matrices_grid.png) (`outputs/final_evaluation/figures/03_holdout_confusion_matrices_grid.png`)