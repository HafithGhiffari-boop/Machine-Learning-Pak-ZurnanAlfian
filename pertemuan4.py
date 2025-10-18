# === LANGKAH 1: Import Library ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# === LANGKAH 2: Buat Dataset ===
data = {
    "IPK": [3.8,2.5,3.4,2.1,3.9,2.8,3.2,2.7,3.6,2.3,
            3.5,2.4,3.7,2.6,3.1,2.2,3.0,2.9,3.85,2.15,
            3.45,2.55,3.75,2.35,3.65,2.45,3.95,2.05,3.25,2.75,
            3.55,2.65,3.15,2.85,3.05,2.95,3.90,2.10,3.40,2.60],
    "Jumlah_Absensi": [3,8,4,12,2,6,5,7,4,9,
                       3,8,5,11,6,10,7,8,2,12,
                       5,9,3,11,4,10,2,12,6,8,
                       5,9,7,11,4,10,2,12,3,8],
    "Waktu_Belajar_Jam": [10,5,7,2,12,4,8,3,9,4,
                          11,5,10,3,8,2,7,4,13,1,
                          9,6,11,2,10,3,12,1,8,4,
                          7,5,9,3,12,2,10,4,11,3],
    "Lulus": [1,0,1,0,1,0,1,0,1,0,
              1,0,1,0,1,0,1,0,1,0,
              1,0,1,0,1,0,1,0,1,0,
              1,0,1,0,1,0,1,0,1,0]
}

# collection
df = pd.DataFrame(data)
df.to_csv("kelulusan_mahasiswa.csv", index=False)

print("=== Dataset Awal ===")
print(df.head())

print("\n=== Info Dataset ===")
print(df.info())
print(df.describe())

# cleaning
print("\n=== Missing Value Check ===")
print(df.isnull().sum())

print("\n=== Duplikasi Data ===")
print(df.duplicated().sum())

# boxplot
sns.boxplot(x=df["IPK"])
plt.title("Boxplot IPK")
plt.show()

#  EDA 
sns.histplot(df["IPK"], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.show()

sns.scatterplot(x="IPK", y="Waktu_Belajar_Jam", data=df, hue="Lulus")
plt.title("Scatterplot IPK vs Waktu Belajar")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()

# Feature Engineering 
df["Rasio_Absensi"] = df["Jumlah_Absensi"] / 14
df["IPK_x_Study"] = df["IPK"] * df["Waktu_Belajar_Jam"]

df.to_csv("processed_kelulusan.csv", index=False)

print("\n=== Dataset dengan Fitur Turunan ===")
print(df.head())

# Splitting Dataset
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\n=== Ukuran Dataset ===")
print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)

# === LANGKAH 8: Evaluasi Model ===
# Model tanpa fitur turunan
X_awal = df[["IPK", "Jumlah_Absensi", "Waktu_Belajar_Jam"]]
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_awal, y, test_size=0.3, stratify=y, random_state=42)

model_awal = LogisticRegression()
model_awal.fit(X_train_a, y_train_a)
y_pred_a = model_awal.predict(X_test_a)
acc_awal = accuracy_score(y_test_a, y_pred_a)

# Model dengan fitur turunan
X_full = df.drop("Lulus", axis=1)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.3, stratify=y, random_state=42)

model_full = LogisticRegression()
model_full.fit(X_train_f, y_train_f)
y_pred_f = model_full.predict(X_test_f)
acc_full = accuracy_score(y_test_f, y_pred_f)

print("\n=== Perbandingan Akurasi ===")
print("Tanpa fitur turunan :", acc_awal)
print("Dengan fitur turunan:", acc_full)
