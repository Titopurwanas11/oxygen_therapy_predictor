"""
Model Pipeline Documentation Utility for OxyPredict.
Outlines the data pre-processing and model execution sequence.
"""

def get_pipeline_steps() -> list:
    """Return list of dictionary objects describing each step in the pipeline."""
    return [
        {
            "step": "1",
            "name": "Patient Data Input",
            "desc": "Penerimaan 44 data klinis pasien, baik melalui form input manual (single) maupun upload file dataset (batch).",
            "icon": "📋"
        },
        {
            "step": "2",
            "name": "Missing Value Imputation",
            "desc": "Pengisian otomatis nilai kosong menggunakan nilai median (untuk numerik) atau modus (untuk kategorikal) berdasarkan dataset pelatihan.",
            "icon": "🩹"
        },
        {
            "step": "3",
            "name": "One Hot Encoding",
            "desc": "Mengubah variabel kategori (seperti jenis kelamin, riwayat asma, atau asuransi) menjadi representasi angka biner yang dipahami model.",
            "icon": "🔢"
        },
        {
            "step": "4",
            "name": "Feature Transformation",
            "desc": "Standarisasi skala data dan penyesuaian order kolom agar sesuai dengan urutan fitur saat pelatihan model.",
            "icon": "🔄"
        },
        {
            "step": "5",
            "name": "Random Forest Classifier",
            "desc": "Prediksi dievaluasi oleh 800 decision trees secara paralel. Setiap tree memberikan suara ('vote') untuk menentukan kelas.",
            "icon": "🌲"
        },
        {
            "step": "6",
            "name": "Probability & Decision Output",
            "desc": "Mengalkulasi persentase probabilitas kebutuhan terapi oksigen dan memicu keputusan biner YES (>=50%) atau NO (<50%).",
            "icon": "🔮"
        },
        {
            "step": "7",
            "name": "SHAP Explainability",
            "desc": "Menghitung kontribusi nilai SHAP individual menggunakan TreeExplainer untuk menganalisis pengaruh positif dan negatif dari setiap fitur.",
            "icon": "🧠"
        },
        {
            "step": "8",
            "name": "AI Clinical Summary",
            "desc": "Menghasilkan narasi penjelasan klinis dinamis secara otomatis berdasarkan kombinasi prediksi dan fitur kontributor utama SHAP.",
            "icon": "📝"
        },
        {
            "step": "9",
            "name": "Clinical Recommendation",
            "desc": "Mengevaluasi rule-engine klinis berbasis aturan untuk menghasilkan prioritas tindakan (Emergency, High, Medium, Low) dan instruksi pemantauan.",
            "icon": "🩺"
        }
    ]
