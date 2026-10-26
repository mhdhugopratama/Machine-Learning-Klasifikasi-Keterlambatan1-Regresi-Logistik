from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression.pkl"
EVAL_PATH = MODEL_DIR / "evaluation.json"
DATA_PATH = BASE_DIR / "dataset_ekspedisi.csv"
TRAIN_DATA_PATH = MODEL_DIR / "data_latih_split_400.csv"
TEST_DATA_PATH = MODEL_DIR / "data_uji_split_100.csv"

st.set_page_config(page_title="Klasifikasi Keterlambatan Pengiriman (Split 80:20)", page_icon="📦", layout="wide")
st.title("📦 Klasifikasi Keterlambatan Pengiriman — Regresi Logistik (Split 80:20)")

if not MODEL_PATH.exists() or not EVAL_PATH.exists() or not TEST_DATA_PATH.exists():
    st.error("Model atau hasil split data belum tersedia. Jalankan `python train_model.py` terlebih dahulu di terminal.")
    st.stop()

@st.cache_resource
def load_resources():
    model = joblib.load(str(MODEL_PATH))
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        evaluation = json.load(f)
    data = pd.read_csv(DATA_PATH)
    df_train_proc = pd.read_csv(TRAIN_DATA_PATH)
    df_test_proc = pd.read_csv(TEST_DATA_PATH)
    
    # Proses seluruh 500 data untuk evaluasi komparatif
    layanan_map = evaluation["layanan_encoding"]
    b0 = evaluation["parameters"]["beta_0"]
    b1 = evaluation["parameters"]["beta_1"]
    b2 = evaluation["parameters"]["beta_2"]
    b3 = evaluation["parameters"]["beta_3"]
    
    df_all = data.copy()
    df_all["X1"] = df_all["Jarak_Pengiriman_KM"]
    df_all["X2"] = df_all["Jumlah_Transit"]
    df_all["X3"] = df_all["Jenis_Layanan"].map(layanan_map)
    df_all["Y"] = (df_all["Status_Keterlambatan"] == "Terlambat").astype(int)
    
    z_all = b0 + (b1 * df_all["X1"]) + (b2 * df_all["X2"]) + (b3 * df_all["X3"])
    p_all = 1.0 / (1.0 + np.exp(-z_all))
    pred_all = (p_all >= 0.5).astype(int)
    
    df_all["Skor_z"] = np.round(z_all, 4)
    df_all["Probabilitas_Terlambat"] = np.round(p_all, 4)
    df_all["Prediksi_Model"] = np.where(pred_all == 1, "Terlambat (1)", "Tepat Waktu (0)")
    
    cond_all = [
        (df_all["Y"] == 1) & (pred_all == 1),
        (df_all["Y"] == 0) & (pred_all == 0),
        (df_all["Y"] == 0) & (pred_all == 1),
        (df_all["Y"] == 1) & (pred_all == 0)
    ]
    choices = ["True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"]
    df_all["Hasil_Evaluasi"] = np.select(cond_all, choices, default="-")
    
    kolom = [
        "ID_Pengiriman", "Jarak_Pengiriman_KM", "Jumlah_Transit", "Jenis_Layanan",
        "Status_Keterlambatan", "Skor_z", "Probabilitas_Terlambat", "Prediksi_Model", "Hasil_Evaluasi"
    ]
    df_all_proc = df_all[kolom]
    
    return model, evaluation, data, df_train_proc, df_test_proc, df_all_proc

model, evaluation, df_data, df_train, df_test, df_all = load_resources()

B0 = evaluation["parameters"]["beta_0"]
B1 = evaluation["parameters"]["beta_1"]
B2 = evaluation["parameters"]["beta_2"]
B3 = evaluation["parameters"]["beta_3"]
LAYANAN_MAP = evaluation["layanan_encoding"]
split_info = evaluation["split_info"]

tab_prediksi, tab_evaluasi, tab_data = st.tabs([
    "🔍 Prediksi Pengiriman",
    "📊 Evaluasi Model (Data Latih vs Uji)",
    "📁 Dataset & Split 80:20"
])

# ==============================================================================
# TAB 1: PREDIKSI DENGAN MODEL HASIL PELATIHAN 80% DATA LATIH
# ==============================================================================
with tab_prediksi:
    st.subheader("Simulasi Prediksi Menggunakan Model Terlatih (Hasil Split 80%)")
    st.caption(
        f"Parameter model diperoleh dari fitting nyata pada {split_info['n_train']} data latih: "
        f"β₀ = {B0:.4f}, β₁ = {B1:.6f}, β₂ = {B2:.6f}, β₃ = {B3:.6f}"
    )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        jarak = st.number_input("Jarak Pengiriman (X₁ KM)", min_value=0.0, max_value=5000.0, value=500.0, step=10.0)
    with c2:
        transit = st.number_input("Jumlah Titik Transit (X₂)", min_value=0, max_value=10, value=1, step=1)
    with c3:
        layanan = st.selectbox("Jenis Layanan (X₃)", list(LAYANAN_MAP.keys()), index=2)

    layanan_code = LAYANAN_MAP[layanan]
    z = B0 + (B1 * jarak) + (B2 * transit) + (B3 * layanan_code)
    prob_terlambat = 1.0 / (1.0 + np.exp(-z))
    prob_tepat = 1.0 - prob_terlambat
    prediksi = 1 if prob_terlambat >= 0.5 else 0

    st.divider()
    res1, res2 = st.columns([1, 1])
    with res1:
        st.markdown("### Hasil Klasifikasi")
        if prediksi == 1:
            st.error("🚨 **Hasil Prediksi: Terlambat (Y = 1)**")
        else:
            st.success("✅ **Hasil Prediksi: Tepat Waktu (Y = 0)**")
        st.write(f"• **P(Terlambat)**: `{prob_terlambat:.4f}` ({prob_terlambat * 100:.2f}%)")
        st.write(f"• **P(Tepat Waktu)**: `{prob_tepat:.4f}` ({prob_tepat * 100:.2f}%)")
        st.write("• **Ambang Batas (Threshold)**: `0.50`")

        fig, ax = plt.subplots(figsize=(6, 2))
        bars = ax.barh(['Tepat Waktu', 'Terlambat'], [prob_tepat, prob_terlambat], color=['#1f77b4', '#ff7f0e'], height=0.45)
        ax.set_xlim(0, 1)
        for bar in bars:
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2%}", va='center', fontweight='bold')
        ax.axvline(0.5, color='gray', linestyle='--', label='Threshold 0.5')
        ax.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)

    with res2:
        st.markdown("### Persamaan Logit Hasil Pelatihan (Data Latih 80%)")
        st.latex(r"z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3")
        st.code(
            f"z = {B0:.4f} + ({B1:.6f} * {jarak:.2f}) + ({B2:.6f} * {transit}) + ({B3:.6f} * {layanan_code})\n"
            f"z = {z:.4f}",
            language="text"
        )
        st.latex(r"P(Y=1 \mid X) = \frac{1}{1 + e^{-z}}")
        st.code(f"P = 1 / (1 + e^(-({z:.4f}))) = {prob_terlambat:.4f}", language="text")
        if prob_terlambat >= 0.5:
            st.caption("Keputusan: Karena P ≥ 0.5, paket diklasifikasikan Terlambat.")
        else:
            st.caption("Keputusan: Karena P < 0.5, paket diklasifikasikan Tepat Waktu.")

# ==============================================================================
# TAB 2: EVALUASI MODEL (DATA LATIH VS DATA UJI)
# ==============================================================================
with tab_evaluasi:
    st.subheader("Evaluasi Performa Model Regresi Logistik")
    
    pilihan_eval = st.radio(
        "Pilih Data untuk Dievaluasi:",
        [
            "🧪 Data Uji / Test Set (20% = 100 Data)",
            "📚 Data Latih / Train Set (80% = 400 Data)",
            "📦 Seluruh Dataset (500 Data)"
        ],
        horizontal=True
    )

    if "Uji" in pilihan_eval:
        eval_data = evaluation["evaluasi_data_uji"]
        df_tampil = df_test
        label_set = "Data Uji (100 Baris)"
    elif "Latih" in pilihan_eval:
        eval_data = evaluation["evaluasi_data_latih"]
        df_tampil = df_train
        label_set = "Data Latih (400 Baris)"
    else:
        eval_data = evaluation["evaluasi_seluruh_data"]
        df_tampil = df_all
        label_set = "Seluruh Data (500 Baris)"

    cm = eval_data["confusion_matrix"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasi", f"{eval_data['accuracy']:.2%}")
    m2.metric("Presisi", f"{eval_data['precision']:.2%}")
    m3.metric("Recall", f"{eval_data['recall']:.2%}")
    m4.metric("F1-Score", f"{eval_data['f1_score']:.2%}")

    st.markdown("---")
    c_cm, c_info = st.columns([1, 1.2])
    with c_cm:
        st.markdown(f"**Confusion Matrix ({label_set}):**")
        cm_df = pd.DataFrame([
            {"Aktual": "Aktual Terlambat (1)", "Prediksi Terlambat (1)": f"TP = {cm['TP']}", "Prediksi Tepat Waktu (0)": f"FN = {cm['FN']}"},
            {"Aktual": "Aktual Tepat Waktu (0)", "Prediksi Terlambat (1)": f"FP = {cm['FP']}", "Prediksi Tepat Waktu (0)": f"TN = {cm['TN']}"}
        ])
        st.dataframe(cm_df, use_container_width=True, hide_index=True)
    with c_info:
        st.markdown("**Analisis Evaluasi:**")
        st.info(
            f"Pada **{label_set}**, akurasi mencapai **{eval_data['accuracy']:.2%}**. "
            f"Probabilitas keterlambatan tertinggi berada pada **{eval_data['prob_max']:.2%}**, sehingga pada threshold 0.50 model memprediksi kelas mayoritas (Tepat Waktu)."
        )

    st.markdown("---")
    st.subheader(f"📋 Tabel Lengkap Seluruh Data Evaluasi: {label_set}")
    st.caption(f"Menampilkan seluruh {len(df_tampil)} baris data lengkap dengan skor z, probabilitas, hasil prediksi, dan sel matriks evaluasi:")
    st.dataframe(df_tampil, use_container_width=True, hide_index=False)
    
    st.download_button(
        label=f"📥 Unduh Tabel {label_set} (CSV)",
        data=df_tampil.to_csv(index=False).encode("utf-8"),
        file_name=f"evaluasi_{label_set.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ==============================================================================
# TAB 3: DATASET & INFORMASI SPLIT 80:20
# ==============================================================================
with tab_data:
    st.subheader("📁 Eksplorasi Dataset & Detail Pembagian (Split 80:20)")
    
    st.markdown(
        f"""
        **Metodologi Pembagian Data (*Train-Test Split*):**
        - **Metode**: `train_test_split` dari pustaka Scikit-Learn
        - **Proporsi**: 80% Data Latih ({split_info['n_train']} baris) : 20% Data Uji ({split_info['n_test']} baris)
        - **Stratifikasi**: Menggunakan `stratify=y` untuk menjaga rasio seimbang antara Tepat Waktu dan Terlambat
        - **Random State**: `42` (menjamin hasil konsisten dan dapat direplikasi)
        """
    )
    
    sub1, sub2, sub3 = st.tabs([
        "🧪 Data Uji / Test Set (20% = 100 Baris)",
        "📚 Data Latih / Train Set (80% = 400 Baris)",
        "📦 Seluruh Dataset Awal (500 Baris)"
    ])
    
    with sub1:
        st.markdown("### Seluruh Data Uji (100 Baris Data)")
        st.caption("Data ini tidak pernah dilihat oleh model selama proses pelatihan (digunakan murni untuk pengujian performa generalisasi):")
        
        u1, u2, u3 = st.columns(3)
        u1.metric("Total Sampel Uji", f"{len(df_test)} baris")
        u2.metric("Tepat Waktu", f"{(df_test['Status_Keterlambatan'] == 'Tepat Waktu').sum()} paket")
        u3.metric("Terlambat", f"{(df_test['Status_Keterlambatan'] == 'Terlambat').sum()} paket")
        
        st.dataframe(df_test, use_container_width=True, hide_index=False)
        st.download_button(
            "📥 Unduh Data Uji (100 Data CSV)",
            data=df_test.to_csv(index=False).encode("utf-8"),
            file_name="data_uji_test_split_100.csv",
            mime="text/csv"
        )
        
    with sub2:
        st.markdown("### Seluruh Data Latih (400 Baris Data)")
        st.caption("Data ini digunakan secara nyata untuk proses `model.fit(X_train, y_train)` guna memperoleh nilai bobot koefisien:")
        
        l1, l2, l3 = st.columns(3)
        l1.metric("Total Sampel Latih", f"{len(df_train)} baris")
        l2.metric("Tepat Waktu", f"{(df_train['Status_Keterlambatan'] == 'Tepat Waktu').sum()} paket")
        l3.metric("Terlambat", f"{(df_train['Status_Keterlambatan'] == 'Terlambat').sum()} paket")
        
        st.dataframe(df_train, use_container_width=True, hide_index=False)
        st.download_button(
            "📥 Unduh Data Latih (400 Data CSV)",
            data=df_train.to_csv(index=False).encode("utf-8"),
            file_name="data_latih_train_split_400.csv",
            mime="text/csv"
        )
        
    with sub3:
        st.markdown("### Seluruh Dataset Awal (500 Baris Data)")
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            st.write("**Distribusi Status Keterlambatan:**")
            st.dataframe(df_data["Status_Keterlambatan"].value_counts(), use_container_width=True)
        with c_d2:
            st.write("**Distribusi Jenis Layanan:**")
            st.dataframe(df_data["Jenis_Layanan"].value_counts(), use_container_width=True)

        st.dataframe(df_data, use_container_width=True, hide_index=False)
        st.download_button(
            "📥 Unduh Dataset Asli (500 Data CSV)",
            data=df_data.to_csv(index=False).encode("utf-8"),
            file_name="dataset_ekspedisi.csv",
            mime="text/csv"
        )
