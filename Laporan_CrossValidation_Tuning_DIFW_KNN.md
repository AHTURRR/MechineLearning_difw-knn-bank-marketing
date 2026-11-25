# LAPORAN TAHAP 06 — CROSS-VALIDATION & HYPERPARAMETER TUNING

**Skrip CLI:** [`src/06_crossvalidation_tuning.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/06_crossvalidation_tuning.py)  
**Notebook:** [`notebooks/06_CrossValidation_Hyperparameter_Tuning.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/06_CrossValidation_Hyperparameter_Tuning.ipynb)  
**Kepatuhan Alur Kerja (*Flowchart*):** Tahap ini mempertemukan cabang **Baseline KNN** (Tahap 04) dan **Proposed DIFW-KNN** (Tahap 05) pada **Stratified 5-Fold Cross-Validation (`36.168` baris `Training / Development Data`)**. `Hold-out Test Data` (`9.043` baris) tetap **dikunci sepenuhnya** dan baru akan dibuka pada **Tahap 07 (`Final Model Evaluation`)**.

---

## 1. Ruang Pencarian Hyperparameter (*Search Space*)

- **Cabang A (`Baseline KNN` — `E1`):** `k` $\in \{3, 5, 7, 11, 15, 21, 31, 51, 71, 101\}$ × `weights` $\in \{\text{'uniform'}, \text{'distance'}\}$ (`20` konfigurasi × `5` fold = `100` evaluasi).
- **Cabang B1 (`Mixed-Type Uniform` — `E2` & `E4`):** `k` $\in \{15, 21, 31, 51, 71\}$ × eksponen jarak lokal `q` $\in \{0.0, 0.5, 1.0, 2.0\}$ (`20` konfigurasi × `5` fold = `100` evaluasi).
- **Cabang B2 (`Proposed DIFW-KNN` — `E3` & `E5`):** `margin` $m \in \{0.30, 0.40, 0.50\}$ × `reg_lambda` $\lambda \in \{0.08, 0.15, 0.25\}$ × `k` $\in \{15, 21, 31, 51, 71\}$ × `q` $\in \{0.0, 0.5, 1.0, 2.0\}$ (`180` konfigurasi × `5` fold = `900` evaluasi).

---

## 2. Konfigurasi Hyperparameter Terpilih untuk Tahap 07 (`Final Model Evaluation`)

- **`E0 (Standard KNN)`:** `k = 5, weights = 'uniform', threshold = 0.50` *(konfigurasi dasar tanpa tuning)*.
- **`E1 (Optimized KNN — Best CV F1)`:** `k = 3, weights = 'uniform', threshold = 0.50` *(catatan: kandidat Best CV PR-AUC Euclidean adalah `k = 51, weights = 'distance'`)*.
- **`E2 (Mixed-Type KNN)`:** `k = 71, q = 0.5, threshold = 0.50`.
- **`E3 (Feature-Weighted KNN)`:** `margin = 0.4, reg_lambda = 0.15, k = 71, q = 0.5, threshold = 0.50`.
- **`E4 (Cost-Sensitive Mixed KNN)`:** `k = 71, q = 0.5, c_fp = 1.0, c_fn = 5.0 (threshold = 0.1667)`.
- **`E5 (DIFW-KNN Lengkap)`:** `margin = 0.4, reg_lambda = 0.15, k = 71, q = 0.5, c_fp = 1.0, c_fn = 5.0 (threshold = 0.1667)`.

### Ringkasan Performa 5-Fold Cross-Validation Model Terpilih (`E0` s.d. `E5`)

| model_id | config_summary | accuracy_mean | precision_yes_mean | recall_yes_mean | f1_yes_mean | pr_auc_mean | roc_auc_mean | mcc_mean | g_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Euclidean k=5, uniform, tau=0.50 | 0.8863 | 0.5364 | 0.2026 | 0.2938 | 0.2823 | 0.6896 | 0.2804 | 0.4444 |
| E1 | Euclidean k=3, uniform, tau=0.50 | 0.8799 | 0.4729 | 0.2342 | 0.3131 | 0.2474 | 0.6664 | 0.2746 | 0.4751 |
| E2 | Mixed Uniform k=71, q=0.5, tau=0.50 | 0.8925 | 0.6983 | 0.1428 | 0.237 | 0.4049 | 0.7673 | 0.283 | 0.3761 |
| E3 | DIFW m=0.4, lam=0.15, k=71, q=0.5, tau=0.50 | 0.8928 | 0.6249 | 0.2087 | 0.3126 | 0.426 | 0.7786 | 0.3184 | 0.4527 |
| E4 | Mixed Cost k=71, q=0.5, 5:1 (tau=0.1667) | 0.8539 | 0.3986 | 0.489 | 0.4391 | 0.4049 | 0.7673 | 0.3586 | 0.6641 |
| E5 | DIFW Lengkap m=0.4, lam=0.15, k=71, q=0.5, 5:1 | 0.8641 | 0.432 | 0.5138 | 0.4692 | 0.426 | 0.7786 | 0.394 | 0.6838 |

---

## 3. Rincian Hasil Grid Search per Cabang

### a. Top 6 Konfigurasi Baseline Euclidean KNN (`E1`)

| weights | k | accuracy_mean | precision_yes_mean | recall_yes_mean | f1_yes_mean | pr_auc_mean | roc_auc_mean | mcc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 3 | 0.8799 | 0.4729 | 0.2342 | 0.3131 | 0.2474 | 0.6664 | 0.2746 |
| distance | 3 | 0.8753 | 0.4394 | 0.2401 | 0.3104 | 0.2709 | 0.6679 | 0.2622 |
| distance | 5 | 0.8838 | 0.5078 | 0.2113 | 0.2982 | 0.3105 | 0.6925 | 0.275 |
| uniform | 5 | 0.8863 | 0.5364 | 0.2026 | 0.2938 | 0.2823 | 0.6896 | 0.2804 |
| distance | 7 | 0.8878 | 0.5563 | 0.1997 | 0.2938 | 0.3315 | 0.7092 | 0.2862 |
| uniform | 7 | 0.8893 | 0.5803 | 0.1945 | 0.2913 | 0.3042 | 0.7062 | 0.2912 |

### b. Top 6 Konfigurasi Mixed-Type Uniform (`E2` & `E4`)

| k | q | pr_auc_mean | roc_auc_mean | e2_f1_mean | e2_mcc_mean | e4_f1_mean | e4_mcc_mean | e4_gmean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 71.0 | 0.5 | 0.4049 | 0.7673 | 0.237 | 0.283 | 0.4391 | 0.3586 | 0.6641 |
| 51.0 | 0.5 | 0.4038 | 0.7635 | 0.245 | 0.2865 | 0.4331 | 0.3509 | 0.6651 |
| 71.0 | 1.0 | 0.4008 | 0.7663 | 0.2393 | 0.2811 | 0.4368 | 0.3554 | 0.6665 |
| 51.0 | 1.0 | 0.3983 | 0.762 | 0.2492 | 0.2873 | 0.4292 | 0.346 | 0.6668 |
| 31.0 | 0.5 | 0.3977 | 0.7533 | 0.2622 | 0.2922 | 0.4187 | 0.3334 | 0.6607 |
| 71.0 | 0.0 | 0.3938 | 0.7661 | 0.2333 | 0.2797 | 0.4361 | 0.3547 | 0.6651 |

### c. Top 10 Konfigurasi Proposed DIFW-KNN (`E3` & `E5`)

| margin | reg_lambda | k | q | pr_auc_mean | roc_auc_mean | e3_f1_mean | e3_mcc_mean | e5_f1_mean | e5_mcc_mean | e5_gmean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 0.15 | 71.0 | 0.5 | 0.426 | 0.7786 | 0.3126 | 0.3184 | 0.4692 | 0.394 | 0.6838 |
| 0.3 | 0.08 | 51.0 | 0.5 | 0.4259 | 0.7757 | 0.3215 | 0.3261 | 0.4715 | 0.3959 | 0.6931 |
| 0.4 | 0.15 | 51.0 | 0.5 | 0.4256 | 0.7753 | 0.3304 | 0.3326 | 0.4664 | 0.3898 | 0.6902 |
| 0.4 | 0.25 | 51.0 | 0.5 | 0.4248 | 0.7755 | 0.3292 | 0.3311 | 0.4705 | 0.3947 | 0.6934 |
| 0.5 | 0.25 | 51.0 | 0.5 | 0.4243 | 0.7759 | 0.3336 | 0.335 | 0.4682 | 0.3919 | 0.6917 |
| 0.5 | 0.15 | 51.0 | 0.5 | 0.424 | 0.7748 | 0.3344 | 0.3348 | 0.4681 | 0.3917 | 0.6935 |
| 0.4 | 0.15 | 71.0 | 1.0 | 0.4239 | 0.7786 | 0.3208 | 0.3251 | 0.4686 | 0.3929 | 0.6867 |
| 0.3 | 0.08 | 51.0 | 1.0 | 0.4237 | 0.7756 | 0.3292 | 0.3327 | 0.4699 | 0.3938 | 0.6955 |
| 0.3 | 0.08 | 71.0 | 0.5 | 0.4235 | 0.7803 | 0.3016 | 0.3093 | 0.471 | 0.3959 | 0.6859 |
| 0.4 | 0.08 | 51.0 | 0.5 | 0.4235 | 0.7756 | 0.3312 | 0.3326 | 0.4662 | 0.3896 | 0.691 |

---

## 4. Konsistensi Bobot Fitur Optimal Lintas 5 Fold CV

| feature | cv_weight_mean | cv_weight_std | uniform_prior | relative_importance |
| --- | --- | --- | --- | --- |
| month | 0.31145 | 0.00387 | 0.0625 | 4.98318 |
| poutcome | 0.10824 | 0.00376 | 0.0625 | 1.73189 |
| previously_contacted | 0.09687 | 0.00347 | 0.0625 | 1.54985 |
| age | 0.0906 | 0.00423 | 0.0625 | 1.44956 |
| day | 0.08909 | 0.00428 | 0.0625 | 1.42542 |
| campaign | 0.06681 | 0.00146 | 0.0625 | 1.0689 |
| pdays_recency | 0.06191 | 0.00358 | 0.0625 | 0.99054 |
| balance | 0.05857 | 0.00483 | 0.0625 | 0.93708 |
| contact | 0.0507 | 0.00235 | 0.0625 | 0.81113 |
| housing | 0.02852 | 0.00268 | 0.0625 | 0.45629 |
| previous | 0.02355 | 0.00296 | 0.0625 | 0.37679 |
| marital | 0.00839 | 0.00282 | 0.0625 | 0.13419 |
| job | 0.00439 | 0.001 | 0.0625 | 0.07022 |
| loan | 0.00055 | 0.00076 | 0.0625 | 0.00887 |
| education | 0.00038 | 0.00058 | 0.0625 | 0.00608 |
| default | 0.0 | 0.0 | 0.0625 | 0.0 |

---

## 5. Sensitivitas Rasio Biaya pada Prediksi Out-of-Fold Model Terpilih (`E5`)

| c_fn_ratio | threshold | accuracy | precision_yes | recall_yes | f1_yes | mcc | g_mean | mean_cost | cost_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.5 | 0.8928 | 0.6245 | 0.2087 | 0.3128 | 0.3185 | 0.453 | 0.1072 | 0.0 |
| 2.0 | 0.3333 | 0.8837 | 0.5035 | 0.4042 | 0.4484 | 0.3872 | 0.6187 | 0.186 | 6.9047 |
| 3.0 | 0.25 | 0.8763 | 0.4704 | 0.4569 | 0.4635 | 0.3937 | 0.6525 | 0.2508 | 14.2317 |
| 4.0 | 0.2 | 0.872 | 0.4556 | 0.4838 | 0.4693 | 0.3968 | 0.6684 | 0.3092 | 19.6868 |
| 5.0 | 0.1667 | 0.8641 | 0.4319 | 0.5138 | 0.4693 | 0.394 | 0.684 | 0.3634 | 23.8955 |
| 7.55 | 0.117 | 0.8217 | 0.3476 | 0.5977 | 0.4396 | 0.3601 | 0.7134 | 0.4865 | 31.8203 |

---

## 6. Artefak Visual

- [01_cv_k_curves_baseline_vs_difw.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/cv_tuning/figures/01_cv_k_curves_baseline_vs_difw.png) (`outputs/cv_tuning/figures/01_cv_k_curves_baseline_vs_difw.png`)
- [02_difw_margin_lambda_heatmaps.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/cv_tuning/figures/02_difw_margin_lambda_heatmaps.png) (`outputs/cv_tuning/figures/02_difw_margin_lambda_heatmaps.png`)
- [03_tuned_cv_winners_e0_to_e5.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/cv_tuning/figures/03_tuned_cv_winners_e0_to_e5.png) (`outputs/cv_tuning/figures/03_tuned_cv_winners_e0_to_e5.png`)