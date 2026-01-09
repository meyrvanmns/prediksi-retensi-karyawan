import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan pendukung
model = joblib.load("employee_retention_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(
    page_title="Sistem Prediksi Retensi Karyawan",
    layout="centered"
)

# ======================
# JUDUL APLIKASI
# ======================
st.title("📊 Sistem Prediksi Retensi Karyawan")
st.write(
    "Aplikasi ini digunakan untuk memprediksi risiko karyawan "
    "mengundurkan diri berdasarkan data historis dan Machine Learning."
)

st.divider()

# ======================
# PRESET DATA
# ======================
st.subheader("🧪 Contoh Data Otomatis (Simulasi)")

preset = st.selectbox(
    "Pilih contoh skenario karyawan:",
    ("Input Manual", "Risiko Tinggi", "Risiko Sedang", "Risiko Rendah")
)

# Default input
input_data = {feature: 0.0 for feature in features}

# ----------------------
# PRESET RISIKO TINGGI
# ----------------------
if preset == "Risiko Tinggi":
    contoh_data = {
        "Age": 25,
        "MonthlyIncome": 3000000,
        "JobSatisfaction": 1,
        "WorkLifeBalance": 1,
        "EnvironmentSatisfaction": 1,
        "YearsAtCompany": 1,
        "TotalWorkingYears": 2,
        "DistanceFromHome": 25,
        "OverTime": 1,
        "Gender": 1,
        "MaritalStatus": 2,
        "Department": 2
    }
    input_data.update(contoh_data)

# ----------------------
# PRESET RISIKO SEDANG
# ----------------------
elif preset == "Risiko Sedang":
    contoh_data = {
        "Age": 32,
        "MonthlyIncome": 6500000,
        "JobSatisfaction": 3,
        "WorkLifeBalance": 3,
        "EnvironmentSatisfaction": 3,
        "YearsAtCompany": 4,
        "TotalWorkingYears": 8,
        "DistanceFromHome": 12,
        "OverTime": 1,
        "Gender": 1,
        "MaritalStatus": 1,
        "Department": 1
    }
    input_data.update(contoh_data)

# ----------------------
# PRESET RISIKO RENDAH
# ----------------------
elif preset == "Risiko Rendah":
    contoh_data = {
        "Age": 40,
        "MonthlyIncome": 12000000,
        "JobSatisfaction": 4,
        "WorkLifeBalance": 4,
        "EnvironmentSatisfaction": 4,
        "YearsAtCompany": 10,
        "TotalWorkingYears": 15,
        "DistanceFromHome": 5,
        "OverTime": 0,
        "Gender": 1,
        "MaritalStatus": 1,
        "Department": 1
    }
    input_data.update(contoh_data)

st.divider()

# ======================
# FORM INPUT DATA
# ======================
st.subheader("🔹 Input Data Karyawan")

for feature in features:
    input_data[feature] = st.number_input(
        label=feature,
        value=float(input_data.get(feature, 0.0)),
        format="%.2f"
    )

# ======================
# PREDIKSI
# ======================
if st.button("Prediksi Risiko Resign"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    hasil_prediksi = model.predict(input_scaled)[0]
    skor_risiko = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("🔍 Hasil Prediksi")

    if hasil_prediksi == 1:
        st.error("⚠️ Risiko Tinggi: Karyawan Berpotensi Mengundurkan Diri")
    else:
        st.success("✅ Risiko Rendah: Karyawan Cenderung Bertahan")

    st.metric(
        label="Skor Risiko Resign",
        value=f"{skor_risiko * 100:.2f}%"
    )

    st.divider()
    st.subheader("📌 Rekomendasi untuk HR")

    if skor_risiko > 0.7:
        st.write("""
        **Tindakan yang Disarankan:**
        - Evaluasi beban kerja dan kompensasi
        - Lakukan diskusi personal (one-on-one)
        - Tingkatkan keseimbangan kerja dan kehidupan pribadi
        """)
    elif skor_risiko > 0.4:
        st.write("""
        **Tindakan yang Disarankan:**
        - Pantau kepuasan kerja karyawan
        - Berikan peluang pengembangan karier
        """)
    else:
        st.write("""
        **Tindakan yang Disarankan:**
        - Pertahankan strategi keterlibatan karyawan yang ada
        - Lanjutkan monitoring secara berkala
        """)
