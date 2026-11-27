# Laporan Tahap 09: Explainability with LIME & Kesimpulan Akhir Penelitian (DIFW-KNN)

Verifikasi reproduksi model `E5` pada Hold-out Test Set (`N = 9,043`): selisih probabilitas maksimum terhadap Tahap 07 = `9.888e-17` (identik).

## 1. Interpretabilitas Lokal dengan LIME (8 Kasus Representatif)

Tabel berikut merangkum penjelasan lokal LIME (`num_samples = 500`) pada kasus nyata Hold-out Test Set (`TP`, `FP`, `FN`, `TN`), termasuk kasus `TP_recovered_from_E0` di mana `E5` berhasil mendeteksi nasabah berlangganan yang gagal ditangkap `E0`:

| case_type | test_index | y_true | prob_e0 | pred_e0 | prob_e5 | pred_e5 | lime_r2_fidelity | top_positive_drivers | top_negative_drivers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TP_recovered_from_E0 | 8201 | 1 | 0.4 | 0 | 0.7953 | 1 | 0.2823 | month=sep (+0.2233); poutcome=success (+0.2059); contact=cellular (+0.0565) | housing=yes (-0.0531); age > 48.00 (-0.0179); 74.00 < balance <= 451.00 (-0.0139) |
| TP_high_conf | 2058 | 1 | 0.8 | 1 | 0.8304 | 1 | 0.258 | poutcome=success (+0.2032); month=sep (+0.1842); previously_contacted=1 (+0.0488) | housing=yes (-0.0538); 8.00 < day <= 16.00 (-0.0129); job=management (-0.0043) |
| FP_borderline | 3752 | 0 | 0.2 | 0 | 0.1667 | 1 | 0.1388 | contact=cellular (+0.0511); housing=no (+0.0489); age <= 33.00 (+0.0272) | month=jul (-0.0735); default=no (-0.0467); poutcome=unknown (-0.0387) |
| FP_high_prob | 5281 | 0 | 0.8 | 1 | 0.7996 | 1 | 0.2158 | month=sep (+0.2419); poutcome=success (+0.0860); contact=cellular (+0.0511) | default=no (-0.0511); 39.00 < age <= 48.00 (-0.0133); loan=no (-0.0112) |
| FN_borderline | 215 | 1 | 0.2 | 0 | 0.1664 | 0 | 0.177 | housing=no (+0.0651); contact=cellular (+0.0630); default=no (+0.0614) | month=nov (-0.0490); 74.00 < balance <= 451.00 (-0.0167); loan=no (-0.0139) |
| FN_clear_miss | 2240 | 1 | 0.0 | 0 | 0.0 | 0 | 0.1669 | contact=cellular (+0.0460); default=no (+0.0454); previous <= 0.00 (+0.0167) | month=may (-0.0729); housing=yes (-0.0347); previously_contacted=0 (-0.0249) |
| TN_clear | 11 | 0 | 0.0 | 0 | 0.0 | 0 | 0.1253 | contact=cellular (+0.0637); 1.00 < campaign <= 2.00 (+0.0337); default=no (+0.0251) | housing=yes (-0.0298); month=jan (-0.0242); job=services (-0.0213) |
| TN_borderline | 7370 | 0 | 0.2 | 0 | 0.1666 | 0 | 0.1339 | month=jun (+0.0569); housing=no (+0.0552); age <= 33.00 (+0.0296) | default=no (-0.0893); poutcome=unknown (-0.0504); previously_contacted=0 (-0.0396) |

## 2. Konsistensi Global (Triplet Weights) vs Lokal (Mean |LIME| pada 40 Sampel)

- **Rata-rata Local Fidelity ($R^2$ LIME pada 40 sampel):** `0.1825`
- **Korelasi Peringkat Global vs Lokal (Spearman $\rho$):** `0.2796` ($p = 2.9425e-01$)

| feature | global_triplet_weight | global_triplet_weight_norm | mean_abs_lime | mean_abs_lime_norm |
| --- | --- | --- | --- | --- |
| month | 0.307303 | 0.307303 | 0.079048 | 0.184498 |
| poutcome | 0.102955 | 0.102955 | 0.034038 | 0.079444 |
| day | 0.0971 | 0.0971 | 0.010716 | 0.025011 |
| age | 0.091251 | 0.091251 | 0.019018 | 0.044387 |
| previously_contacted | 0.086564 | 0.086564 | 0.028658 | 0.066888 |
| campaign | 0.066816 | 0.066816 | 0.015914 | 0.037144 |
| balance | 0.062403 | 0.062403 | 0.019343 | 0.045147 |
| pdays_recency | 0.056907 | 0.056907 | 0.014996 | 0.035 |
| contact | 0.050783 | 0.050783 | 0.05311 | 0.12396 |
| housing | 0.034035 | 0.034035 | 0.043546 | 0.101636 |
| previous | 0.024854 | 0.024854 | 0.016443 | 0.038377 |
| marital | 0.008186 | 0.008186 | 0.016495 | 0.0385 |
| loan | 0.005772 | 0.005772 | 0.014511 | 0.033869 |
| job | 0.005071 | 0.005071 | 0.017955 | 0.041906 |
| education | 0.0 | 0.0 | 0.012746 | 0.029749 |
| default | 0.0 | 0.0 | 0.031913 | 0.074484 |

## 3. Kesimpulan Akhir Penelitian

### Jawaban Rumusan Masalah (RM1–RM3) & Hipotesis (H1–H3)
1. **RM1 & H1 (Penanganan Data Campuran & Sentinel `pdays = -1`):** Dekomposisi `pdays = -1` (81,74% data) menjadi `previously_contacted` dan `pdays_recency` (`M_pos_train = 195.0`) bersama metrik jarak campuran berbasis rentang kuantil $Q_{95}-Q_{05}$ (`E2`) terbukti menaikkan `PR-AUC` dari `0.2921` (`E0`) menjadi `0.4160` dan `ROC-AUC` dari `0.7014` menjadi `0.7733`.
2. **RM2 & H2 (Pembelajaran Bobot Fitur Berbasis Triplet pada Data Imbalanced Multimodal):** Strategi *Hybrid 50% Global + 50% Local-Context Triplet Sampling* mengatasi kegagalan sampling global acak dan menghasilkan bobot fitur stabil (`5-seed Spearman = 0.9906`). Fitur paling dominan adalah `month` (`0.3073`), `poutcome` (`0.1030`), `day` (`0.0971`), `age` (`0.0913`), dan `previously_contacted` (`0.0866`), yang menaikkan `PR-AUC` ke `0.4479` dan `ROC-AUC` ke `0.7888` (`E3`/`E5`, DeLong $p_{\text{Holm}} = 0.0399$ vs `E2`).
3. **RM3 & H3 (Aturan Keputusan *Cost-Sensitive* 5:1 & Integrasi Penuh `E5`):** Mengintegrasikan aturan Bayes *cost-sensitive* ($\tau = 0.1667$) pada `E5` (DIFW-KNN Lengkap) menaikkan `Recall` dari `20.13%` (`213 TP`) menjadi `53.02%` (`561 TP`), `F1-Score` dari `0.2969` menjadi `0.4748`, `MCC` dari `0.2907` menjadi `0.3998`, dan menekan total biaya mis-klasifikasi sebesar **26,43%** (`4.389 -> 3.229`, paired bootstrap $p_{\text{Holm}} < 0.0001$).

### Keterbatasan & Saran Penelitian Lanjutan
- **Komputasi Inferensi:** Karena KNN bersifat *lazy learner*, waktu prediksi `E5` pada `9.043` data uji adalah `7.779` detik (masih sangat layak untuk *batch scoring* harian, namun dapat dipercepat dengan *prototype condensation* atau *approximate nearest neighbors* untuk skenario *real-time*).
- **Asumsi Rasio Biaya Tetap (5:1):** Nilai $C_{FN}:C_{FP} = 5:1$ dapat disesuaikan secara dinamis oleh manajemen bank mengikuti perubahan margin suku bunga deposito dan biaya operasional *call center*.
- **Generalisasi Domain:** Kerangka DIFW-KNN dapat diuji pada dataset tabular campuran *imbalanced* lain (seperti *credit scoring* atau *churn prediction*).

## 4. Artefak Grafik Tahap 09
- `outputs/explainability_lime/figures/01_lime_representative_cases.png`
- `outputs/explainability_lime/figures/02_lime_vs_triplet_weights.png`
