# DIFW-KNN: Domain-Informed Feature-Weighted K-Nearest Neighbors for Bank Marketing Classification

Repositori penelitian kuantitatif eksperimental untuk pengembangan algoritma **Domain-Informed Feature-Weighted K-Nearest Neighbors (`DIFW-KNN`)** menggunakan *Mixed-Type Distance*, *Supervised Triplet Feature Weighting*, dan *Cost-Sensitive Decision Rule* pada dataset **UCI Bank Marketing (`data/bank-full.csv`, ID 222)**.

---

## Ringkasan Metodologi

- **Dataset:** `data/bank-full.csv` (`45.211` observasi, `16` prediktor asli, target biner `y`: `88,30%` `no` vs `11,70%` `yes`, *Imbalance Ratio* `7,55 : 1`).
- **Skenario Prediksi (*Pre-Call Targeting*):** Fitur `duration` dihapus sebelum pemodelan karena merupakan *post-call target leakage* (*single-feature ROC-AUC* = `0,8076`).
- **Protokol Anti-*Data Leakage*:**
  - *Stratified Train-Test Split* `80:20` (`random_state=42`) menghasilkan `36.168` data latih dan `9.043` data uji.
  - Seluruh parameter statistik (`M_pos_train` untuk rekayasa fitur `pdays`, `StandardScaler`, `OneHotEncoder`, serta kuantil `Q05` dan `Q95` pada `MixedTypePreprocessor`) dipelajari **hanya dari data latih**.
- **16 Fitur Domain:**
  - **6 Numerik:** `age`, `balance`, `day`, `campaign`, `previous`, `pdays_recency`
  - **10 Kategorikal/Biner:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, `previously_contacted` (label `"unknown"` dipertahankan sebagai kategori valid).

---

## Struktur Repositori

```text
DIFW-KNN/
│
├── data/
│   ├── bank-full.csv                                 # Dataset UCI Bank Marketing (ID 222)
│   └── processed/
│       ├── train_split.csv                           # Stratified train split (36.168 baris)
│       ├── test_split.csv                            # Stratified test split (9.043 baris)
│       └── train_cv_folds.csv                        # Indeks kanonik Stratified 5-Fold CV
│
├── src/
│   ├── __init__.py
│   ├── mixed_type_distance.py                        # Developer 1: Mixed-Type Distance (Q05-Q95 + mismatch)
│   ├── triplet_learning.py                           # Developer 2: Balanced Triplet Sampling & Margin Loss
│   ├── feature_weight_optimizer.py                   # Developer 2: Simplex Projected GD Feature Weight Optimizer
│   ├── difw_knn.py                                   # Developer 3: DIFW-KNN Classifier & Ablation Suite (E2-E5)
│   ├── cost_sensitive_decision.py                    # Developer 3: Cost-Sensitive Bayes Rule & Trade-off Grid
│   ├── preprocessing.py                              # Modul DomainFeatureEngineer & BaselineEuclideanPreprocessor
│   ├── baseline_knn.py                               # Modul reusable evaluasi & CV Baseline KNN (E0 & E1)
│   ├── 01_eda.py                                     # Tahap 01: Data Understanding & EDA
│   ├── 02_preprocessing.py                           # Tahap 02: Preprocessing & Feature Engineering
│   ├── 03_stratified_split.py                        # Tahap 03: Stratified Train-Test Split & 5-Fold Setup
│   ├── 04_baseline_knn.py                            # Tahap 04: Baseline KNN (Training / Development Only)
│   ├── 05_proposed_difw_knn.py                       # Tahap 05: Proposed DIFW-KNN (6 Sub-Komponen pada Training Data)
│   ├── 06_crossvalidation_tuning.py                  # Tahap 06: 5-Fold CV & Hyperparameter Tuning (Baseline + DIFW-KNN)
│   └── 07_final_model_evaluation.py                  # Tahap 07: Evaluasi Final Hold-out Test Data + 95% Bootstrap CI
│
├── tests/
│   ├── test_mixed_distance.py                        # Unit test metrik jarak Mixed-Type, OpenBLAS, & pdays
│   ├── test_triplet_learning.py                      # Unit test BalancedTripletSampler (random/semi_hard/local_context)
│   └── test_difw_knn.py                              # Unit & integration test Cost-Sensitive Rule & DIFW-KNN
│
├── notebooks/
│   ├── 01_EDA.ipynb                                  # Tahap 01: Data Understanding & EDA
│   ├── 02_Preprocessing_FeatureEngineering.ipynb     # Tahap 02: Preprocessing & Feature Engineering
│   ├── 03_Stratified_Train_Test_Split.ipynb          # Tahap 03: Stratified Split (Hold-out Test dikunci)
│   ├── 04_Baseline_KNN.ipynb                         # Tahap 04: Baseline KNN (Training / Development Data)
│   ├── 05_Proposed_DIFW_KNN.ipynb                    # Tahap 05: Proposed DIFW-KNN (6 Sub-Komponen Algoritma Usulan)
│   ├── 06_CrossValidation_Hyperparameter_Tuning.ipynb # Tahap 06: 5-Fold CV & Hyperparameter Tuning Gabungan
│   └── 07_Final_Model_Evaluation.ipynb               # Tahap 07: Final Model Evaluation pada Hold-out Test Data
│
├── outputs/
│   ├── eda/                                          # Tabel statistik & 5 grafik EDA
│   ├── preprocessing/                                # Tabel parameter kuantil, unit checks, & 3 grafik
│   ├── split/                                        # Uji KS/Chi2 covariate balance, ringkasan CV, & 2 grafik
│   ├── baseline_knn/                                 # Profil 5-Fold Development & OOF Baseline KNN (E0)
│   ├── proposed_difw_knn/                            # Validasi 9 check komponen, bobot fitur, 5-seed stability, & grafik
│   ├── cv_tuning/                                    # Grid search 5-Fold CV, hyperparameter terpilih, & 3 grafik
│   └── final_evaluation/                             # Metrik Hold-out Test E0-E5, 95% Bootstrap CI, & 3 grafik
│   └── comparative_ablation/                         # Uji signifikansi (DeLong, McNemar, Bootstrap) & studi ablasi
│   └── explainability_lime/                          # Penjelasan lokal LIME, alignment global-lokal, & grafik
│
├── Desain_Penelitian_DIFW_KNN.md                     # Dokumen desain & protokol penelitian
├── Laporan_EDA_DIFW_KNN.md                           # Laporan Tahap 01: Data Understanding & EDA
├── Laporan_Preprocessing_DIFW_KNN.md                 # Laporan Tahap 02: Preprocessing & Feature Engineering
├── Laporan_Stratified_Split_DIFW_KNN.md              # Laporan Tahap 03: Stratified Split & 5-Fold Setup
├── Laporan_Baseline_KNN_DIFW_KNN.md                  # Laporan Tahap 04: Baseline KNN (Training/Dev Only)
├── Laporan_Proposed_DIFW_KNN.md                      # Laporan Tahap 05: Proposed DIFW-KNN (Training/Dev Only)
├── Laporan_CrossValidation_Tuning_DIFW_KNN.md        # Laporan Tahap 06: 5-Fold CV & Hyperparameter Tuning
├── Laporan_Final_Evaluation_DIFW_KNN.md              # Laporan Tahap 07: Final Model Evaluation (Hold-out Test)
├── Laporan_Comparative_Ablation_DIFW_KNN.md          # Laporan Tahap 08: Comparative Analysis & Ablation Study
├── Laporan_Explainability_LIME_Conclusions_DIFW_KNN.md # Laporan Tahap 09: Explainability with LIME & Conclusions
├── README.md
└── requirements.txt                                  # Dependensi Python teruji (Python 3.13.7)
```

---

## Cara Menjalankan

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Unit & Integration Tests (`tests/`)
```bash
pytest -v
```

### 3. Menjalankan Skrip Python (`.py`) Sesuai Urutan Flowchart
```bash
# Tahap 01: Data Understanding & EDA
python src/01_eda.py

# Tahap 02: Preprocessing, Feature Engineering, & Distance Unit Checks
python src/02_preprocessing.py

# Tahap 03: Stratified Train-Test Split (80:20) & 5-Fold Development Setup
python src/03_stratified_split.py

# Tahap 04: Baseline KNN (Training / Development Data Only)
python src/04_baseline_knn.py

# Tahap 05: Proposed DIFW-KNN Algorithm Development (Training / Development Data Only)
python src/05_proposed_difw_knn.py

# Tahap 06: Joint 5-Fold Cross-Validation & Hyperparameter Tuning (Training Data Only)
python src/06_crossvalidation_tuning.py

# Tahap 07: Final Model Evaluation on Hold-out Test Data + 95% Bootstrap CI
python src/07_final_model_evaluation.py

# Tahap 08: Comparative Analysis, Statistical Significance & Ablation Study
python src/08_comparative_analysis_ablation.py

# Tahap 09: Explainability with LIME & Final Research Conclusions
python src/09_explainability_lime_conclusions.py
```

### 4. Membuka Notebook Interaktif (`.ipynb`)
Buka file berikut menggunakan VS Code atau JupyterLab (seluruh sel sudah dieksekusi dan menyertakan output tabel serta grafik *inline*):
- `notebooks/01_EDA.ipynb`
- `notebooks/02_Preprocessing_FeatureEngineering.ipynb`
- `notebooks/03_Stratified_Train_Test_Split.ipynb`
- `notebooks/04_Baseline_KNN.ipynb`
- `notebooks/05_Proposed_DIFW_KNN.ipynb`
- `notebooks/06_CrossValidation_Hyperparameter_Tuning.ipynb`
- `notebooks/07_Final_Model_Evaluation.ipynb`
- `notebooks/08_Comparative_Analysis_Ablation.ipynb`
- `notebooks/09_Explainability_LIME_Conclusions.ipynb`

---

## Status Tahapan Penelitian (Sesuai Flowchart 01–09)

- [x] **Tahap 01 — Data Understanding & EDA:** (`src/01_eda.py`, `notebooks/01_EDA.ipynb`)
- [x] **Tahap 02 — Preprocessing & Feature Engineering:** (`src/preprocessing.py`, `src/02_preprocessing.py`, `notebooks/02_Preprocessing_FeatureEngineering.ipynb`)
- [x] **Tahap 03 — Stratified Train-Test Split (`Training/Dev` vs `Hold-out Test` Dikunci):** (`src/03_stratified_split.py`, `notebooks/03_Stratified_Train_Test_Split.ipynb`)
- [x] **Tahap 04 — Baseline KNN (`Training / Development Data` Only):** (`src/baseline_knn.py`, `src/04_baseline_knn.py`, `notebooks/04_Baseline_KNN.ipynb`)
- [x] **Tahap 05 — Proposed DIFW-KNN (`Mixed-Type Distance` $\rightarrow$ `Balanced Triplet Sampling` $\rightarrow$ `Triplet Margin Loss` $\rightarrow$ `Supervised Feature Weight Learning` $\rightarrow$ `Weighted KNN & Local Probability` $\rightarrow$ `Cost-Sensitive Decision Rule` pada `Training Data`):** (`src/05_proposed_difw_knn.py`, `notebooks/05_Proposed_DIFW_KNN.ipynb`)
- [x] **Tahap 06 — Cross-Validation & Hyperparameter Tuning (Gabungan Baseline KNN & Proposed DIFW-KNN pada Training Data):** (`src/06_crossvalidation_tuning.py`, `notebooks/06_CrossValidation_Hyperparameter_Tuning.ipynb`)
- [x] **Tahap 07 — Final Model Evaluation (Membuka `Hold-out Test Data` + 95% Bootstrap CI):** (`src/07_final_model_evaluation.py`, `notebooks/07_Final_Model_Evaluation.ipynb`)
- [x] **Tahap 08 — Comparative Analysis & Ablation Study (`E0` s.d. `E5`):** (`src/08_comparative_analysis_ablation.py`, `notebooks/08_Comparative_Analysis_Ablation.ipynb`)
- [x] **Tahap 09 — Explainability with LIME & Conclusions:** (`src/09_explainability_lime_conclusions.py`, `notebooks/09_Explainability_LIME_Conclusions.ipynb`)
