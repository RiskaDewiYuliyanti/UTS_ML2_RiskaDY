# Input nama
nama = st.text_input("👤 Siapa namamu?", max_chars=30)

if nama:
    st.success(f"Hai, {nama}! Ayo kita mulai mengisi datamu 😊")
    st.markdown("---")

    # Layout input
    col1, col2 = st.columns(2)

    with col1:
        density = st.number_input("🧪 Density", min_value=1.0, max_value=2.0, value=1.05, step=0.01)
        age = st.number_input("🎂 Age", min_value=18, max_value=80, value=30, step=1)
        height = st.number_input("📏 Height (cm)", min_value=140.0, max_value=210.0, value=170.0, step=0.5)  # Corrected Height input
        weight = st.number_input("⚖️ Weight (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.5)  # Corrected Weight input
        neck = st.number_input("👔 Neck (cm)", min_value=20, max_value=60, value=38)
        chest = st.number_input("🫁 Chest (cm)", min_value=60, max_value=150, value=100)
        abdomen = st.number_input("🧍 Abdomen (cm)", min_value=60, max_value=150, value=90)

    with col2:
        hip = st.number_input("🍑 Hip (cm)", min_value=60.0, max_value=150.0, value=95.0)
        thigh = st.number_input("🦵 Thigh (cm)", min_value=30, max_value=90, value=55)
        knee = st.number_input("🦿 Knee (cm)", min_value=30.0, max_value=70.0, value=40.0)
        ankle = st.number_input("🦶 Ankle (cm)", min_value=15, max_value=40, value=22)
        biceps = st.number_input("💪 Biceps (cm)", min_value=20.0, max_value=60.0, value=35.0)
        forearm = st.number_input("🦾 Forearm (cm)", min_value=15, max_value=50, value=28)
        wrist = st.number_input("🖐️ Wrist (cm)", min_value=10.0, max_value=30.0, value=18.0)

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

    # Scale the input data
    scaled_input = scaler.transform([input_features])
    
    # Debugging: Print shape of scaled_input
    st.write(f"Scaled input shape: {scaled_input.shape}")

    # Pass to the model
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
