import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="bodyfat_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Title
st.markdown("<h1 style='text-align: center;'>Prediksi Body Fat Percentage</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Umur", min_value=0, value=25)
        weight = st.number_input("Berat Badan (kg)", min_value=0.0, format="%.2f")
        height = st.number_input("Tinggi Badan (cm)", min_value=0.0, format="%.2f")
        neck = st.number_input("Lingkar Leher (cm)", min_value=0.0, format="%.2f")
        chest = st.number_input("Lingkar Dada (cm)", min_value=0.0, format="%.2f")
        abdomen = st.number_input("Lingkar Perut (cm)", min_value=0.0, format="%.2f")

    with col2:
        hip = st.number_input("Lingkar Pinggul (cm)", min_value=0.0, format="%.2f")
        thigh = st.number_input("Lingkar Paha (cm)", min_value=0.0, format="%.2f")
        knee = st.number_input("Lingkar Lutut (cm)", min_value=0.0, format="%.2f")
        ankle = st.number_input("Lingkar Pergelangan Kaki (cm)", min_value=0.0, format="%.2f")
        biceps = st.number_input("Lingkar Bisep (cm)", min_value=0.0, format="%.2f")
        forearm = st.number_input("Lingkar Lengan Bawah (cm)", min_value=0.0, format="%.2f")
        wrist = st.number_input("Lingkar Pergelangan Tangan (cm)", min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("Prediksi")

if submitted:
    try:
        # Gabungkan semua input ke array
        input_data = np.array([[age, weight, height, neck, chest, abdomen,
                                hip, thigh, knee, ankle, biceps, forearm, wrist]])

        # Scaling
        scaled_input = scaler.transform(input_data).astype(np.float32)

        # Set input ke model
        interpreter.set_tensor(input_details[0]['index'], scaled_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Ambil prediksi
        prediction = float(output[0][0])
        st.success(f"**Persentase Lemak Tubuh yang Diprediksi: {prediction:.2f}%**")

        # Penentuan kategori
        if prediction < 6:
            category = "Essential Fat"
            desc = "Lemak minimal yang dibutuhkan untuk fungsi tubuh normal."
        elif prediction < 14:
            category = "Athletes"
            desc = "Kadar lemak yang umum dimiliki atlet; sehat dan fit."
        elif prediction < 18:
            category = "Fitness"
            desc = "Tingkat lemak sehat untuk individu aktif secara fisik."
        elif prediction < 25:
            category = "Average"
            desc = "Kadar lemak rata-rata populasi umum. Masih dalam batas sehat."
        else:
            category = "Obese"
            desc = "Lemak tubuh berlebih. Perlu perhatian dan pola hidup sehat."

        # Tampilkan kategori dan edukasi
        st.markdown(f"### Kategori Lemak Tubuh: **{category}**")
        st.info(desc)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
