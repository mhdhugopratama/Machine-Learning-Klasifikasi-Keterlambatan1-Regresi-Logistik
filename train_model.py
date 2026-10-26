from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_ekspedisi.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Baca Dataset
df = pd.read_csv(DATA_PATH)

# 2. Transformasi Fitur (Label Encoding)
LAYANAN_MAP = {
    "Ekspres": 0,
    "Kargo": 1,
    "Reguler": 2,
    "Sameday": 3
}

df["X1"] = df["Jarak_Pengiriman_KM"]
df["X2"] = df["Jumlah_Transit"]
df["X3"] = df["Jenis_Layanan"].map(LAYANAN_MAP)
df["Y"] = (df["Status_Keterlambatan"] == "Terlambat").astype(int)

# 3. Pembagian Data Sejati (Split 80% Latih : 20% Uji)
X = df[["X1", "X2", "X3"]]
y = df["Y"]

train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=y
)

df_train = df.loc[train_idx].copy()
df_test = df.loc[test_idx].copy()

X_train = df_train[["X1", "X2", "X3"]]
y_train = df_train["Y"]

X_test = df_test[["X1", "X2", "X3"]]
y_test = df_test["Y"]

# 4. Pelatihan Model Regresi Logistik Sejati (Fit pada Data Latih 80% = 400 Data)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Parameter yang Dihasilkan dari Pelatihan (Bukan Hardcoded)
B0 = float(model.intercept_[0])
B1 = float(model.coef_[0][0])
B2 = float(model.coef_[0][1])
B3 = float(model.coef_[0][2])

# Fungsi evaluasi metrik
def evaluate_dataset(x_input, y_true):
    prob = model.predict_proba(x_input)[:, 1]
    pred = model.predict(x_input)
    cm = confusion_matrix(y_true, pred, labels=[1, 0])
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": {
            "TP": int(cm[0, 0]),
            "FN": int(cm[0, 1]),
            "FP": int(cm[1, 0]),
            "TN": int(cm[1, 1])
        },
        "prob_max": float(prob.max()),
        "prob_min": float(prob.min()),
        "prob_mean": float(prob.mean())
    }

eval_train = evaluate_dataset(X_train, y_train)
eval_test = evaluate_dataset(X_test, y_test)
eval_all = evaluate_dataset(X, y)

# 5. Simpan Hasil Split Data Lengkap ke CSV untuk Digunakan di Streamlit
def process_table(df_subset):
    df_sub = df_subset.copy()
    z = B0 + (B1 * df_sub["X1"]) + (B2 * df_sub["X2"]) + (B3 * df_sub["X3"])
    p = 1.0 / (1.0 + np.exp(-z))
    pred = (p >= 0.5).astype(int)
    df_sub["Skor_z"] = np.round(z, 4)
    df_sub["Probabilitas_Terlambat"] = np.round(p, 4)
    df_sub["Prediksi_Model"] = np.where(pred == 1, "Terlambat (1)", "Tepat Waktu (0)")
    
    conditions = [
        (df_sub["Y"] == 1) & (pred == 1),
        (df_sub["Y"] == 0) & (pred == 0),
        (df_sub["Y"] == 0) & (pred == 1),
        (df_sub["Y"] == 1) & (pred == 0)
    ]
    choices = ["True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"]
    df_sub["Hasil_Evaluasi"] = np.select(conditions, choices, default="-")
    
    kolom = [
        "ID_Pengiriman", "Jarak_Pengiriman_KM", "Jumlah_Transit", "Jenis_Layanan",
        "Status_Keterlambatan", "Skor_z", "Probabilitas_Terlambat", "Prediksi_Model", "Hasil_Evaluasi"
    ]
    return df_sub[kolom]

df_train_proc = process_table(df_train)
df_test_proc = process_table(df_test)

df_train_proc.to_csv(MODEL_DIR / "data_latih_split_400.csv", index=False)
df_test_proc.to_csv(MODEL_DIR / "data_uji_split_100.csv", index=False)

# 6. Simpan Model & Metadata Evaluasi
payload = {
    "parameters": {
        "beta_0": round(B0, 6),
        "beta_1": round(B1, 6),
        "beta_2": round(B2, 6),
        "beta_3": round(B3, 6),
        "persamaan_logit": f"z = {B0:.4f} + ({B1:.6f} * X1) + ({B2:.6f} * X2) + ({B3:.6f} * X3)"
    },
    "layanan_encoding": LAYANAN_MAP,
    "split_info": {
        "metode": "train_test_split (Scikit-Learn)",
        "train_size": 0.8,
        "test_size": 0.2,
        "random_state": 42,
        "stratify": "Status_Keterlambatan",
        "n_train": len(df_train),
        "n_test": len(df_test)
    },
    "threshold": 0.5,
    "evaluasi_data_latih": eval_train,
    "evaluasi_data_uji": eval_test,
    "evaluasi_seluruh_data": eval_all
}

joblib.dump(model, MODEL_DIR / "logistic_regression.pkl")
with open(MODEL_DIR / "evaluation.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("=" * 70)
print("PELATIHAN REGRESI LOGISTIK SEJATI DENGAN SPLIT DATA 80:20 SELESAI")
print("=" * 70)
print(f"Koefisien Terlatih dari 400 Data Latih:")
print(f"  • Intersep (B0) : {B0:.6f}")
print(f"  • Jarak (B1)    : {B1:.6f}")
print(f"  • Transit (B2)  : {B2:.6f}")
print(f"  • Layanan (B3)  : {B3:.6f}")
print("-" * 70)
print(f"1. Data Latih / Train Set ({eval_train['n_samples']} baris - 80%):")
print(f"   Akurasi : {eval_train['accuracy']:.2%}")
print(f"   TP: {eval_train['confusion_matrix']['TP']} | TN: {eval_train['confusion_matrix']['TN']} | FP: {eval_train['confusion_matrix']['FP']} | FN: {eval_train['confusion_matrix']['FN']}")
print("-" * 70)
print(f"2. Data Uji / Test Set ({eval_test['n_samples']} baris - 20%):")
print(f"   Akurasi : {eval_test['accuracy']:.2%}")
print(f"   TP: {eval_test['confusion_matrix']['TP']} | TN: {eval_test['confusion_matrix']['TN']} | FP: {eval_test['confusion_matrix']['FP']} | FN: {eval_test['confusion_matrix']['FN']}")
print("=" * 70)
