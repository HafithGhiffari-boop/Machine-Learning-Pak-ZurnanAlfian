import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

RANDOM_STATE = 42
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (6,4)

# LANGKAH 1: Muat data (Pilihan A: processed_kelulusan.csv)
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split stratified 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

print("Shapes ->", "Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# LANGKAH 2: Pipeline & Baseline Random Forest
num_cols = X_train.select_dtypes(include="number").columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=RANDOM_STATE
)

pipe = Pipeline([("pre", pre), ("clf", rf)])

start = time()
pipe.fit(X_train, y_train)
train_time = time() - start

y_val_pred = pipe.predict(X_val)
f1_val = f1_score(y_val, y_val_pred, average="macro")
print("\nBaseline RF — F1(val):", f1_val)
print("Validation classification report:\n", classification_report(y_val, y_val_pred, digits=3))

# LANGKAH 3: Validasi silang (CV) pada training set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro (train): %.3f ± %.3f" % (cv_scores.mean(), cv_scores.std()))

# LANGKAH 4: Tuning ringkas (GridSearchCV)
param_grid = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe, param_grid=param_grid, cv=skf, scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("\nGridSearchCV done.")
print("Best params:", gs.best_params_)
print("Best CV F1 (on train):", gs.best_score_)

best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
print("Best RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))
print("Validation classification report (best):\n", classification_report(y_val, y_val_best, digits=3))

# LANGKAH 5: Evaluasi akhir (Test set)
final_model = best_model  # pilih model final
y_test_pred = final_model.predict(X_test)

print("\n=== Evaluasi Akhir (Test Set) ===")
f1_test = f1_score(y_test, y_test_pred, average="macro")
print("F1(test):", f1_test)
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):\n", confusion_matrix(y_test, y_test_pred))

# ROC-AUC & curves (jika predict_proba tersedia)
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    try:
        auc = roc_auc_score(y_test, y_test_proba)
        print("ROC-AUC(test):", auc)
    except Exception:
        auc = None

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}" if auc is not None else "")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)"); plt.legend(); plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)
    plt.show()

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure(); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (Test Set)")
    plt.tight_layout(); plt.savefig("pr_test.png", dpi=120)
    plt.show()
else:
    print("Model has no predict_proba; skip ROC/PR curves.")

# LANGKAH 6: Feature importance (native)
try:
    importances = final_model.named_steps["clf"].feature_importances_
    # get feature names after preprocessing
    feature_names = final_model.named_steps["pre"].get_feature_names_out()
    fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop feature importances:")
    for name, val in fi:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print("Feature importance not available:", e)

# LANGKAH 7: Simpan model
joblib.dump(final_model, "rf_model.pkl")
print("\nModel disimpan sebagai rf_model.pkl")
print("Waktu latih baseline (s):", round(train_time,3))
