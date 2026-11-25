# LAPORAN TAHAP 05 — PROPOSED ALGORITHM DEVELOPMENT (`DIFW-KNN`)

**Modul Kode:** [`src/mixed_type_distance.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/mixed_type_distance.py), [`src/triplet_learning.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/triplet_learning.py), [`src/feature_weight_optimizer.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/feature_weight_optimizer.py), [`src/cost_sensitive_decision.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/cost_sensitive_decision.py), [`src/difw_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/difw_knn.py), [`src/05_proposed_difw_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/05_proposed_difw_knn.py)  
**Notebook:** [`notebooks/05_Proposed_DIFW_KNN.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/05_Proposed_DIFW_KNN.ipynb)  
**Kepatuhan Alur Kerja (*Flowchart*):** Seluruh pengembangan dan validasi ke-6 sub-komponen dilakukan **murni pada `Training / Development Data` (`36.168` baris)**; `Hold-out Test Data` (`9.043` baris) tetap dikunci hingga Tahap 07.

---

## 1. Validasi 6 Sub-Komponen Algoritma Usulan (`9/9 PASS`)

| id | sub_stage | check | value | status |
| --- | --- | --- | --- | --- |
| 1 | Mixed-Type Distance | Self-distance d_j(x, x) = 0 | max_diag=0.0e+00 | PASS |
| 2 | Mixed-Type Distance | Simetri d_j(x, z) = d_j(z, x) | max_diff=0.0e+00 | PASS |
| 3 | Mixed-Type Distance | Rentang d_j(x, z) dalam [0, 1] | [0.0000, 1.0000] | PASS |
| 4 | Mixed-Type Distance | Ekuivalensi OpenBLAS vs exact loop | max_diff=2.22e-16 | PASS |
| 5 | Balanced Triplet Sampling | Proporsi anchor 50:50 & label valid | pos_ratio=0.500, valid=True | PASS |
| 6 | Triplet Margin Loss | Gradient check analitik vs numerik (<1e-7) | max_abs_err=3.96e-11 | PASS |
| 7 | Feature Weight Learning | Kendala simpleks (w_j >= 0, sum = 1) | min=0.00000, sum=1.000000000000 | PASS |
| 8 | Feature Weight Learning | Penurunan Triplet Margin Loss | 0.28153 -> 0.22732 | PASS |
| 9 | Feature Weight Learning | Stabilitas bobot lintas 5 random seed | Spearman=0.9906, Cosine=0.9993 | PASS |

---

## 2. Bobot Fitur Terpelajar (`Supervised Feature Weight Learning`)

Konfigurasi inisialisasi pengembangan: `mining_strategy='local_context'`, `n_triplets=30,000`, `margin=0.4`, `reg_lambda=0.15`.

| feature | weight | uniform_weight | relative_importance | weight_pct |
| --- | --- | --- | --- | --- |
| month | 0.3079 | 0.0625 | 4.9262 | 30.7888 |
| poutcome | 0.1046 | 0.0625 | 1.6734 | 10.4589 |
| day | 0.1002 | 0.0625 | 1.603 | 10.0185 |
| previously_contacted | 0.09 | 0.0625 | 1.4403 | 9.0021 |
| age | 0.0868 | 0.0625 | 1.3894 | 8.6836 |
| campaign | 0.0736 | 0.0625 | 1.1776 | 7.3598 |
| pdays_recency | 0.0571 | 0.0625 | 0.9136 | 5.7101 |
| balance | 0.053 | 0.0625 | 0.8472 | 5.2951 |
| contact | 0.0524 | 0.0625 | 0.8377 | 5.2357 |
| previous | 0.0299 | 0.0625 | 0.4791 | 2.9942 |
| housing | 0.0281 | 0.0625 | 0.4501 | 2.8132 |
| marital | 0.0123 | 0.0625 | 0.1969 | 1.2305 |
| job | 0.0023 | 0.0625 | 0.0362 | 0.226 |
| loan | 0.0018 | 0.0625 | 0.0294 | 0.1837 |
| education | 0.0 | 0.0625 | 0.0 | 0.0 |
| default | 0.0 | 0.0625 | 0.0 | 0.0 |

### Uji Stabilitas Bobot Fitur Lintas 5 Random Seed (`[42, 100, 2026, 7, 999]`)

| feature | weight_mean | weight_std | weight_min | weight_max |
| --- | --- | --- | --- | --- |
| month | 0.30747 | 0.00178 | 0.30436 | 0.30968 |
| poutcome | 0.10303 | 0.00289 | 0.10015 | 0.10784 |
| day | 0.09661 | 0.0026 | 0.09366 | 0.10018 |
| age | 0.09098 | 0.00306 | 0.08684 | 0.09436 |
| previously_contacted | 0.09009 | 0.00222 | 0.08734 | 0.09324 |
| campaign | 0.06995 | 0.00293 | 0.06672 | 0.0736 |
| pdays_recency | 0.05816 | 0.00221 | 0.05564 | 0.0611 |
| balance | 0.05731 | 0.00326 | 0.05295 | 0.06257 |
| contact | 0.05054 | 0.00179 | 0.04765 | 0.05236 |
| housing | 0.02921 | 0.0013 | 0.02738 | 0.03079 |
| previous | 0.02871 | 0.0013 | 0.02636 | 0.02994 |
| marital | 0.00914 | 0.00298 | 0.00472 | 0.01231 |
| job | 0.00424 | 0.00165 | 0.00226 | 0.00646 |
| loan | 0.00379 | 0.00238 | 0.00178 | 0.00826 |
| education | 0.00077 | 0.00155 | 0.0 | 0.00387 |
| default | 0.0 | 0.0 | 0.0 | 0.0 |

---

## 3. Bukti Kontribusi Komponen pada 5-Fold Development Set (Sebelum Hyperparameter Tuning)

| model_id | model_label | accuracy_mean | precision_yes_mean | recall_yes_mean | f1_yes_mean | pr_auc_mean | roc_auc_mean | mcc_mean | g_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_dev | Standard Euclidean KNN (k=5, uniform, tau=0.50) | 0.8863 | 0.5364 | 0.2026 | 0.2938 | 0.2823 | 0.6896 | 0.2804 | 0.4444 |
| E2_dev | Mixed-Type KNN (k=31, uniform, tau=0.50) | 0.8923 | 0.6549 | 0.1676 | 0.2668 | 0.3917 | 0.752 | 0.294 | 0.4068 |
| E3_dev | Feature-Weighted KNN (k=31, triplet, tau=0.50) | 0.8941 | 0.6224 | 0.2408 | 0.347 | 0.4184 | 0.7706 | 0.3422 | 0.4858 |
| E4_dev | Cost-Sensitive Mixed KNN (k=31, uniform, tau=0.1667) | 0.8332 | 0.3516 | 0.5041 | 0.4142 | 0.3917 | 0.752 | 0.3277 | 0.6648 |
| E5_dev | Proposed DIFW-KNN (k=31, triplet + tau=0.1667) | 0.8466 | 0.3902 | 0.5521 | 0.4571 | 0.4184 | 0.7706 | 0.3785 | 0.6992 |

---

## 4. Analisis Aturan Keputusan Sensitif Biaya (`Cost-Sensitive Decision Rule`)

| c_fn | threshold | accuracy | precision_yes | recall_yes | f1_yes | mcc | g_mean | mean_cost | cost_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.5 | 0.894 | 0.6217 | 0.2408 | 0.3472 | 0.3422 | 0.486 | 0.106 | 0.0 |
| 2.0 | 0.3333 | 0.8832 | 0.5007 | 0.4035 | 0.4469 | 0.3852 | 0.618 | 0.1866 | 4.1738 |
| 3.0 | 0.25 | 0.8745 | 0.4646 | 0.4758 | 0.4701 | 0.399 | 0.6642 | 0.2481 | 12.5 |
| 4.0 | 0.2 | 0.8626 | 0.4274 | 0.5138 | 0.4667 | 0.3908 | 0.6834 | 0.308 | 17.2854 |
| 5.0 | 0.1667 | 0.8466 | 0.3901 | 0.5521 | 0.4572 | 0.3785 | 0.6993 | 0.3629 | 21.301 |
| 7.55 | 0.117 | 0.7841 | 0.3001 | 0.6346 | 0.4075 | 0.3267 | 0.7143 | 0.4958 | 27.8936 |
| 10.0 | 0.0909 | 0.7187 | 0.2488 | 0.6953 | 0.3664 | 0.2858 | 0.7085 | 0.602 | 33.4942 |

> **Kesiapan Menuju Tahap 06 (`Cross-Validation & Hyperparameter Tuning`):**
> Seluruh sub-komponen `Proposed DIFW-KNN` telah tervalidasi secara matematis dan empiris pada data pengembangan. Tahap selanjutnya adalah mempertemukan `Baseline KNN` (Tahap 04) dan `Proposed DIFW-KNN` (Tahap 05) di **Tahap 06 (`Cross-Validation & Hyperparameter Tuning`)** untuk mencari konfigurasi *hyperparameter* optimal sebelum membuka `Hold-out Test Data` di Tahap 07.

---

## 5. Artefak Visual

- [01_triplet_loss_and_feature_weights.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/proposed_difw_knn/figures/01_triplet_loss_and_feature_weights.png) (`outputs/proposed_difw_knn/figures/01_triplet_loss_and_feature_weights.png`)
- [02_dev_component_progression.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/proposed_difw_knn/figures/02_dev_component_progression.png) (`outputs/proposed_difw_knn/figures/02_dev_component_progression.png`)
- [03_cost_sensitive_rule_behavior.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/proposed_difw_knn/figures/03_cost_sensitive_rule_behavior.png) (`outputs/proposed_difw_knn/figures/03_cost_sensitive_rule_behavior.png`)