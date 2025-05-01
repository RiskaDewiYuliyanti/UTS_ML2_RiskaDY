import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model & scaler
model = tf.lite.Interpreter(model_path="bodyfat_model.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

scaler = joblib.load('scaler.pkl')

# Konfigurasi halaman
st.set_page_config(page_title="🧠 Body Fat Predictor", layout="centered")

# Tambah font Poppins & header styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        font-size: 2.5em;
        font-weight: 600;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .sub-header {
        font-size: 1.1em;
        text-align: center;
        color: #555;
        margin-bottom: 2em;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🧠 Body Fat Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Masukkan data tubuhmu untuk memprediksi persentase lemak tubuh 💪</div>', unsafe_allow_html=True)

# Input nama
nama = st.text_input("👤 Siapa namamu?", max_chars=30)

if nama:
    st.success(f"Hai, {nama}! Ayo kita mulai mengisi datamu 😊")
    st.markdown("---")

    # Layout input
    col1, col2 = st.columns(2)

    with col1:
        density = st.number_input("🧪 Density", min_value=1.0, max_value=2.0, value=1.05, step=0.01)
        age = st.slider("🎂 Age", 18, 80, 30)
        weight = st.number_input("⚖️ Weight (kg)", 40.0, 150.0, 70.0, step=0.5)
        height = st.number_input("📏 Height (cm)", 140.0, 210.0, 170.0, step=0.5)
        neck = st.slider("👔 Neck (cm)", 20, 60, 38)
        chest = st.slider("🫁 Chest (cm)", 60, 150, 100)
        abdomen = st.slider("🧍 Abdomen (cm)", 60, 150, 90)

    with col2:
        hip = st.number_input("🍑 Hip (cm)", 60.0, 150.0, 95.0)
        thigh = st.slider("🦵 Thigh (cm)", 30, 90, 55)
        knee = st.number_input("🦿 Knee (cm)", 30.0, 70.0, 40.0)
        ankle = st.slider("🦶 Ankle (cm)", 15, 40, 22)
        biceps = st.number_input("💪 Biceps (cm)", 20.0, 60.0, 35.0)
        forearm = st.slider("🦾 Forearm (cm)", 15, 50, 28)
        wrist = st.number_input("🖐️ Wrist (cm)", 10.0, 30.0, 18.0)

    # Buat array dari input SESUAI URUTAN SAAT TRAINING (tanpa BodyFat)
    input_features = [
        density,  # Density
        age,      # Age
        weight,   # Weight
        height,   # Height
        neck,     # Neck
        chest,    # Chest
        abdomen,  # Abdomen
        hip,      # Hip
        thigh,    # Thigh
        knee,     # Knee
        ankle,    # Ankle
        biceps,   # Biceps
        forearm,  # Forearm
        wrist     # Wrist
    ]

    scaled_input = scaler.transform([input_features])
    model.set_tensor(input_details[0]['index'], scaled_input.astype(np.float32))
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])[0][0]

    # Tombol prediksi
    if st.button("🔍 Prediksi Body Fat"):
        st.subheader(f"✨ {nama}, estimasi body fat kamu adalah: **{prediction:.2f}%**")

        if prediction > 25:
            st.warning("⚠️ Body fat kamu agak tinggi. Pertimbangkan olahraga rutin dan pola makan seimbang.")
        elif prediction < 10:
            st.info("🤔 Kamu cukup lean! Tapi pastikan tetap dalam kondisi sehat.")
        else:
            st.balloons()
            st.success("🎯 Body fat kamu termasuk sehat! Great job 💪")

    st.markdown("---")
    st.caption("© 2025 Body Fat Predictor by [Riska D Y]")
else:
    st.info("👆 Masukkan namamu dulu untuk memulai prediksi ya!")
