# LAPORAN TAHAP 04 — BASELINE KNN (`TRAINING / DEVELOPMENT DATA`)

**Modul Kode:** [`src/baseline_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/baseline_knn.py), [`src/04_baseline_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/04_baseline_knn.py)  
**Notebook:** [`notebooks/04_Baseline_KNN.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/04_Baseline_KNN.ipynb)  
**Kepatuhan Alur Kerja (*Flowchart*):** Tahap ini **hanya menggunakan `Training / Development Data` (`36.168` baris)**; `Hold-out Test Data` (`9.043` baris) tetap dikunci dan **belum diakses** hingga Tahap 07 (*Final Model Evaluation*).

---

## 1. Spesifikasi Arsitektur Baseline KNN (`E0`)

- **Representasi Fitur:** `StandardScaler` pada 6 fitur numerik + `OneHotEncoder(handle_unknown='ignore')` pada 10 fitur kategorikal/biner (`52` dimensi setelah transformasi per *fold* latih).
- ** Pengukuran Jarak:** Jarak Euclidean standar tanpa pembobotan fitur domain.
- **Konfigurasi Dasar (`E0`):** `k = 5`, `weights = 'uniform'`, *decision threshold* $\tau = 0,50$.

---

## 2. Unit Test Ekuivalensi Implementasi Pencarian Tetangga (Fold 0)

| k | weights | max_prob_abs_diff_k | max_dist_trunc_diff | n_boundary_tie_rows_maxk | status |
| --- | --- | --- | --- | --- | --- |
| 5 | uniform | 0.0 | 0.0 | 2 | PASS |
| 5 | distance | 2.220446049250313e-16 | 0.0 | 2 | PASS |
| 21 | uniform | 0.0 | 0.0 | 0 | PASS |
| 21 | distance | 3.3306690738754696e-16 | 0.0 | 0 | PASS |

---

## 3. Profil Kinerja Baseline pada 5-Fold Development Set (`36.168` Observasi)

| variant | k | weights | accuracy_mean | precision_yes_mean | recall_yes_mean | f1_yes_mean | pr_auc_mean | roc_auc_mean | mcc_mean | g_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_Standard_k5_uniform | 5 | uniform | 0.8863 | 0.5364 | 0.2026 | 0.2938 | 0.2823 | 0.6896 | 0.2804 | 0.4444 |
| Ref_Euclidean_k5_distance | 5 | distance | 0.8838 | 0.5078 | 0.2113 | 0.2982 | 0.3105 | 0.6925 | 0.275 | 0.4531 |

### Out-of-Fold Confusion Matrix (36.168 Baris Training/Development)

| variant | oof_tn | oof_fp | oof_fn | oof_tp |
| --- | --- | --- | --- | --- |
| E0_Standard_k5_uniform | 31197 | 740 | 3374 | 857 |
| Ref_Euclidean_k5_distance | 31072 | 865 | 3337 | 894 |

> **Diagnosis Kelemahan Baseline KNN pada Data Training:**
> - Pada konfigurasi standar `E0 (k=5, uniform)`, nilai `Accuracy` tampak tinggi (`0.8863`) semata-mata karena kelas mayoritas `no` mendominasi `88,30%` data.
> - Namun untuk kelas target `yes`, `Recall` hanya mencapai **`0.2026`** (`3,374` dari `4,231` nasabah potensial gagal terdeteksi!), dengan `F1 = 0.2938`, `PR-AUC = 0.2823`, dan `MCC = 0.2804`.
> - Kelemahan ini timbul karena: (1) jarak Euclidean di atas `52` kolom *one-hot + z-score* terdistorsi oleh outlier numerik dan dilusi fitur berdaya pisah rendah, serta (2) ambang batas $\tau = 0,50$ tidak memperhitungkan ketimpangan biaya kesalahan (*false negative* vs *false positive*).
> - Model Baseline KNN selanjutnya akan dibawa ke **Tahap 06 (`Cross-Validation & Hyperparameter Tuning`)** bersama **Tahap 05 (`Proposed DIFW-KNN`)**.

---

## 4. Artefak Visual

- [01_baseline_dev_fold_metrics.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/baseline_knn/figures/01_baseline_dev_fold_metrics.png) (`outputs/baseline_knn/figures/01_baseline_dev_fold_metrics.png`)
- [02_baseline_oof_roc_pr_confusion.png](file:///C:/Users/Advan Workplus/Downloads/jurnak/outputs/baseline_knn/figures/02_baseline_oof_roc_pr_confusion.png) (`outputs/baseline_knn/figures/02_baseline_oof_roc_pr_confusion.png`)