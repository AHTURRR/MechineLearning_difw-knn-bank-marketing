# DESAIN PENELITIAN DIFW-KNN

**Pengembangan Domain-Informed Feature-Weighted K-Nearest Neighbors untuk Klasifikasi Bank Marketing**

> Draft desain penelitian untuk konsultasi dosen. Belum memuat hasil eksperimen pada dataset asli.

## 1. Identitas dan Tujuan Penelitian

- Judul (sementara): Development of a Domain-Informed Feature-Weighted K-Nearest Neighbors Algorithm Using Mixed-Type Distance and Cost-Sensitive Decision for Bank Marketing Classification.
- Pendekatan: penelitian kuantitatif eksperimental dengan tugas supervised binary classification.
- Dataset: Bank Marketing, UCI Machine Learning Repository (dataset ID 222), menggunakan bank-full.csv: 45.211 observasi, 16 prediktor asli, dan target y (yes/no).
- Tujuan: menguji apakah pengukuran jarak berdasarkan tipe fitur, pembelajaran bobot fitur berbasis triplet, dan keputusan sensitif biaya meningkatkan performa model dibandingkan KNN standar.

## 2. Rumusan Masalah dan Hipotesis

- RM1: Apakah mixed-type distance menghasilkan performa klasifikasi lebih baik daripada KNN dengan jarak Euclidean pada representasi fitur yang sesuai?
- RM2: Apakah bobot fitur yang dipelajari dari triplet training memberikan peningkatan dibanding bobot seragam?
- RM3: Bagaimana cost-sensitive decision mengubah trade-off precision, recall, dan biaya kesalahan pada kelas yes?
- H1: Mixed-type distance dapat memperbaiki kualitas jarak dibandingkan baseline yang diuji. H2: Supervised feature weighting dapat meningkatkan metrik kelas positif. H3: Cost-sensitive decision menghasilkan trade-off yang lebih sesuai skenario biaya. Semua hipotesis wajib diuji, bukan diasumsikan terbukti.

## 3. Desain Alur Penelitian

- Literature Review & Research Gap → Dataset Acquisition → Data Understanding & EDA → Stratified Train/Test Split → Training-only Preprocessing & Feature Engineering → Dua jalur pengembangan: Baseline KNN dan Proposed DIFW-KNN → Cross-Validation & Hyperparameter Tuning pada training → Evaluasi pada Hold-out Test → Ablation Study, Perbandingan, Interpretasi LIME → Kesimpulan.
- Jalur DIFW-KNN: Mixed-Type Dissimilarity → Balanced Triplet Sampling → Triplet Margin Loss → Supervised Feature Weight Learning → Feature-Weighted KNN & Local Probability → Cost-Sensitive Decision Rule.

## 4. Tahap Pengolahan Dataset

- Periksa tipe data, missing values, kategori unknown, target y, dan distribusi kelas. Untuk prediksi sebelum panggilan, hapus fitur duration karena nilainya hanya diketahui setelah telepon selesai.
- Transformasikan pdays = -1 menjadi dua fitur: previously_contacted dan pdays_recency (0 untuk belum pernah dihubungi). Pastikan definisi waktu prediksi konsisten untuk campaign, day, month, dan variabel lainnya.
- Gunakan stratified 80:20 split sebagai contoh protokol awal (36.168 training dan 9.043 testing). Pilih model melalui stratified cross-validation hanya di training set. Parameter skala, imputasi, triplet, bobot, dan biaya keputusan tidak boleh dipelajari dari test set.

## 5. Formulasi Algoritma Usulan

- Mixed-Type Distance: d_j(x,z) = min(1, |x_j-z_j| / (Q95_j - Q05_j)) untuk fitur numerik; d_j(x,z)=1 jika fitur kategorikal berbeda, dan 0 jika sama.
- Jarak berbobot: D_w(x,z) = Σ_j w_j d_j(x,z), dengan w_j ≥ 0 dan Σ_j w_j = 1. Seluruh parameter scaling ditaksir menggunakan training set.
- Triplet Margin Loss: L(w) = mean[max(0, m + D_w(a,p) - D_w(a,n))] + λ||w-u||², dengan anchor dan positive berlabel sama; negative berlabel berbeda; u adalah bobot seragam.
- Prediksi probabilitas lokal: P_hat(yes|x) = Σ_{i∈N_k(x)} α_i I(y_i=yes) / Σ_{i∈N_k(x)} α_i, dengan α_i = (D_w(x,x_i)+ε)^(-q).
- Cost-Sensitive Decision: pilih yes bila P_hat(yes|x) ≥ C_FP/(C_FP+C_FN); rasio biaya adalah skenario eksperimen, bukan biaya bisnis aktual kecuali telah tersedia data biaya yang terukur.

## 6. Desain Pengujian dan Validasi

- Fase validasi komponen: uji self-distance = 0; simetri; jarak pada [0,1]; bobot ≥ 0 dan jumlah = 1; triplet positive/negative benar; gradient check; penurunan loss pada data training; sensitivitas bobot terhadap random seed.
- Fase integrasi: gunakan jarak dan bobot hasil belajar untuk mencari k tetangga terdekat, estimasi probabilitas lokal, lalu aturan keputusan berbasis biaya.
- Fase evaluasi: gunakan data testing yang sama untuk seluruh model, laporkan interval kepercayaan dan lakukan ablation study. Jangan mengklaim peningkatan performa hanya berdasarkan penurunan triplet loss.

## 7. Matriks Eksperimen

| ID | Model | Tujuan |
|---|---|---|
| E0 | Standard KNN | One-hot/scaling dan Euclidean, sebagai baseline |
| E1 | Optimized KNN | Mengukur efek tuning k dan distance weighting |
| E2 | Mixed-Type KNN | Mengukur efek tipe jarak |
| E3 | Feature-Weighted KNN | Mengukur efek supervised triplet weights |
| E4 | Cost-Sensitive KNN | Mengukur efek aturan keputusan pada KNN tanpa bobot terpelajar |
| E5 | DIFW-KNN lengkap | Integrasi mixed distance + feature weighting + cost rule |

## 8. Output yang Diharapkan

- Tabel profil dataset dan distribusi kelas; laporan preprocessing dan fitur yang dikeluarkan; hasil unit test mixed distance dan triplet loss; tabel learned feature weights; grafik loss dan bobot fitur.
- Tabel performa komparatif: Accuracy, Precision (yes), Recall (yes), F1 (yes), PR-AUC, ROC-AUC, MCC, G-Mean; confusion matrix; kurva Precision–Recall dan ROC; analisis waktu komputasi.
- Output interpretasi: contoh penjelasan LIME dan pemeriksaan stabilitas bobot fitur. Output akhir adalah bukti apakah pendekatan usulan meningkatkan metrik tertentu serta trade-off yang muncul.

## 9. Batasan dan Risiko

- Model usulan dan kebaruannya masih berupa hipotesis ilmiah. Triplet learning, weighted distance, cost-sensitive classification, dan LIME bukan konsep baru secara terpisah; novelty perlu diverifikasi lebih luas.
- Perbandingan dengan angka artikel terdahulu tidak langsung setara apabila varian dataset, duration, split, encoding atau sampling berbeda. Baseline utama harus dilatih ulang pada protokol eksperimen yang sama.
- Kinerja KNN pada 45 ribu data dapat membutuhkan waktu signifikan saat pencarian tetangga; catat runtime dan gunakan pendekatan komputasi efisien bila diperlukan.

## 10. Referensi Awal

- UCI Machine Learning Repository. Bank Marketing (Dataset 222). https://archive.ics.uci.edu/dataset/222/bank+marketing
- Moro, S., Rita, P., & Cortez, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems. https://doi.org/10.1016/j.dss.2014.03.001
- Koumétio & Toulni (2021). Improving KNN Model for Direct Marketing Prediction in Smart Cities. https://doi.org/10.1007/978-3-030-72065-0_7
- Yu & Huo (2022). Classification of Imbalanced Data Set in Financial Field Based on Combined Algorithm. https://doi.org/10.1155/2022/1839204
- Yusof & Fajila (2026). Conditional Tabular GAN-Based Synthetic Data Generation for Model Generalisation Improvement. https://doi.org/10.32890/jict2026.25.1.1

## Format tabel hasil akhir (diisi setelah eksperimen)

| Model | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Standard KNN (E0: k=5, uniform, tau=0.50) | 0.8884 | 0.5650 | 0.2013 | 0.2969 | 0.2921 | 0.2907 |
| Optimized KNN (E1: k=3, uniform, tau=0.50) | 0.8816 | 0.4876 | 0.2420 | 0.3234 | 0.2619 | 0.2863 |
| Mixed-Type KNN (E2: k=71, q=0.5, tau=0.50) | 0.8924 | 0.7033 | 0.1389 | 0.2320 | 0.4160 | 0.2806 |
| Feature-Weighted KNN (E3: m=0.4, lam=0.15, k=71, q=0.5, tau=0.50) | 0.8968 | 0.6731 | 0.2297 | 0.3425 | 0.4479 | 0.3528 |
| Cost-Sensitive KNN (E4: k=71, q=0.5, 5:1, tau=0.1667) | 0.8577 | 0.4125 | 0.5104 | 0.4563 | 0.4160 | 0.3783 |
| DIFW-KNN lengkap (E5: m=0.4, lam=0.15, k=71, q=0.5, 5:1, tau=0.1667) | 0.8628 | 0.4299 | 0.5302 | 0.4748 | 0.4479 | 0.3998 |

**Catatan:** Seluruh metrik `E0`–`E5` diukur satu kali pada `Hold-out Test Set` (`9.043` observasi) menggunakan *hyperparameter* terkunci hasil 5-Fold CV Tahap 06 ([`Laporan_Final_Evaluation_DIFW_KNN.md`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/Laporan_Final_Evaluation_DIFW_KNN.md)). Varian referensi `E1_PR` (Best CV PR-AUC: `k=51, distance`) mencapai `PR-AUC = 0.4042`, `ROC-AUC = 0.7656`, `F1 = 0.2284`, `MCC = 0.2793`. Pada `E5`, `ROC-AUC = 0.7888`, `G-Mean = 0.6934`, dan rata-rata biaya kesalahan turun `26.43%` dibanding `E0` (`0.3571` vs `0.4853`).