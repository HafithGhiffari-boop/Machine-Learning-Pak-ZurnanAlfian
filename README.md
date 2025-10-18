# Machine-Learning-Pak-ZurnanAlfian

## MachineLearning
Prediksi Kelulusan Mahasiswa dengan Machine Learning
Proyek ini mengimplementasikan analisis data dan model Machine Learning untuk memprediksi kelulusan mahasiswa berdasarkan data akademik seperti IPK, jumlah absensi, dan waktu belajar. Melalui tahapan data cleaning, exploratory data analysis (EDA), feature engineering, model building, tuning, dan evaluasi, proyek ini menghasilkan model Random Forest yang mampu memprediksi kelulusan secara akurat dan siap digunakan untuk proses inference otomatis.

## Tahapan Proyek

## Data Collection & Cleaning
Membaca dataset kelulusan_mahasiswa.csv menggunakan Pandas, menghapus duplikasi, menangani nilai kosong, dan melakukan visualisasi awal menggunakan Seaborn serta Matplotlib untuk mendeteksi outlier.

## Exploratory Data Analysis (EDA)
Melakukan analisis deskriptif dan visualisasi hubungan antar variabel menggunakan histogram, scatter plot, dan heatmap korelasi untuk memahami pola yang memengaruhi kelulusan mahasiswa.

## Feature Engineering
Menambahkan fitur baru seperti Rasio_Absensi dan IPK_x_Study untuk meningkatkan performa model. Dataset hasil olahan disimpan sebagai processed_kelulusan.csv.

## Dataset Splitting
Membagi dataset menjadi train, validation, dan test set dengan rasio 70/15/15 menggunakan train_test_split dengan stratify untuk menjaga distribusi label.

## Model Development
- Baseline Model: Logistic Regression dengan preprocessing pipeline menggunakan SimpleImputer dan StandardScaler.
- Model Alternatif: Random Forest Classifier dengan parameter class_weight="balanced" untuk menangani ketidakseimbangan kelas.

## Model Evaluation & Hyperparameter Tuning
Melakukan validasi silang (StratifiedKFold) dan tuning hyperparameter menggunakan GridSearchCV. Evaluasi dilakukan dengan metrik F1-score (macro), classification report, dan ROC-AUC curve.

## Feature Importance
Menampilkan peringkat kepentingan fitur (feature importance) dari model Random Forest untuk interpretasi hasil.

## Model Deployment Preparation
Menyimpan model terbaik ke file rf_model.pkl menggunakan joblib dan menambahkan contoh prediksi lokal (inference) dengan input fiktif.

## Hasil Akhir
Model Random Forest memberikan performa terbaik dengan nilai F1-score dan ROC-AUC yang tinggi pada data validasi dan data uji. Pipeline ini dapat digunakan kembali untuk memprediksi kelulusan mahasiswa baru secara otomatis dengan hasil yang cepat dan konsisten.
