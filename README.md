# UTS_ML2_RiskaDY
# 🧠 Body Fat Predictor

Aplikasi prediksi persentase **lemak tubuh** berdasarkan data antropometri.  
Dibuat dengan **Streamlit**, **TensorFlow Lite**, dan **Scikit-learn**, aplikasi ini memanfaatkan model machine learning untuk memberikan estimasi body fat seseorang secara cepat dan interaktif.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-ff4b4b?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## ✨ Fitur

- 🔍 Prediksi kadar lemak tubuh (Body Fat %) dari 14 fitur tubuh
- 🎨 UI modern dan ramah pengguna (menggunakan font Poppins & emoji)
- 📊 Input fleksibel: kombinasi slider dan number input
- 🙋 Personalisasi dengan input nama pengguna
- 🧠 Model machine learning berbasis TensorFlow Lite
- 🎈 Animasi interaktif (balon, alert, dll)

---

## 🚀 Cara Menjalankan

1. **Clone repositori ini**

```bash
git clone https://github.com/username/body-fat-predictor.git
cd body-fat-predictor

Install dependensi

bash
Copy
Edit
pip install -r requirements.txt
Jalankan aplikasi

bash
Copy
Edit
streamlit run app.py
Aplikasi akan terbuka otomatis di browser: http://localhost:8501

📂 Struktur File
bash
Copy
Edit
body-fat-predictor/
├── app.py                  # Main Streamlit app
├── bodyfat_model.tflite    # Trained model in TFLite format
├── scaler.pkl              # Preprocessing scaler (StandardScaler)
├── requirements.txt        # List of required packages
└── README.md               # Project documentation

📋 Requirements
Python >= 3.10
TensorFlow
Streamlit
NumPy
Joblib

Semua sudah tercantum di requirements.txt.

🤖 Tentang Model
Model dilatih menggunakan dataset body fat yang umum digunakan dalam studi kesehatan.
Fitur input meliputi:

Density, Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist

Ukuran tubuh: Density, Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist

Model menggunakan regresi dan telah dikonversi ke TensorFlow Lite untuk efisiensi runtime.

👩‍💻 Pengembang
Riska D Y
📧 [riskadewiyuliyanti@gmail.coml]
🌐 [https://www.linkedin.com/in/riskady/]


⚠️ Disclaimer
Aplikasi ini bertujuan untuk edukasi dan estimasi, bukan diagnosis medis.
