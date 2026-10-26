# Klasifikasi Keterlambatan Pengiriman Paket Ekspedisi — Regresi Logistik (Split Data 80:20)

Proyek ini menerapkan algoritma **Regresi Logistik** dengan alur **Split Data 80:20 Sejati (*Train-Test Split*)** untuk mengklasifikasikan status keterlambatan pengiriman paket ekspedisi ke dalam dua kategori biner:
- **Terlambat ($Y = 1$)**
- **Tepat Waktu ($Y = 0$)**

---

## 1. Metodologi Pembagian Data (*Train-Test Split*)
Total dataset adalah **500 baris data** dari `dataset_ekspedisi.csv`:
- **Data Latih / Train Set (80% = 400 Baris Data)**: Digunakan secara nyata untuk proses `model.fit(X_train, y_train)` guna menemukan bobot parameter optimal.
- **Data Uji / Test Set (20% = 100 Baris Data)**: Data independen yang belum pernah dilihat model, digunakan untuk menguji performa generalisasi.
- **Stratifikasi**: Menggunakan `stratify=y` agar proporsi Tepat Waktu dan Terlambat pada Data Latih dan Data Uji tetap seimbang.
- **Random State**: `42` (menjamin hasil pembagian selalu konsisten dan dapat direplikasi).

---

## 2. Parameter Hasil Pelatihan (Data Latih 400 Baris)
Model `LogisticRegression` dilatih menggunakan Scikit-Learn dari 400 data latih dan menghasilkan persamaan logit:

$$z = -1.2917 + (-0.000062 \times X_1) + (-0.018288 \times X_2) + (0.181534 \times X_3)$$

Fungsi Sigmoid:
$$P(Y = 1 \mid X) = \frac{1}{1 + e^{-z}}$$

Aturan Keputusan Standar:
- Jika $P(Y = 1 \mid X) \ge 0.5 \implies$ **Terlambat (1)**
- Jika $P(Y = 1 \mid X) < 0.5 \implies$ **Tepat Waktu (0)**

---

## 3. Hasil Evaluasi Performa Model

| Metrik | Data Latih (400 Baris) | Data Uji (100 Baris) |
| :--- | :---: | :---: |
| **Akurasi** | **74.50%** | **75.00%** |
| **True Negative (TN)** | 298 | 75 |
| **False Negative (FN)** | 102 | 25 |
| **True Positive (TP)** | 0 | 0 |
| **False Positive (FP)** | 0 | 0 |

---

## 4. Cara Menjalankan Program

Buka terminal PowerShell pada direktori proyek:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -u train_model.py
python -m streamlit run app.py
```
