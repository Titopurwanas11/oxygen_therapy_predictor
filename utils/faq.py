"""
FAQ Educational Content for OxyPredict.
Contains common questions and answers about the CDSS operation.
"""

def get_faqs_list() -> list:
    """Return a list of FAQ dictionaries."""
    return [
        {
            "question": "Apa arti dari Prediction Probability?",
            "answer": (
                "Prediction Probability adalah nilai probabilitas (0% hingga 100%) yang menunjukkan keyakinan matematis "
                "dari model Random Forest terhadap kelas target. Probabilitas di atas 50% memicu prediksi 'YES' (membutuhkan oksigen), "
                "sedangkan di bawah 50% memicu prediksi 'NO'. Semakin jauh dari 50%, semakin kuat sinyal klinis yang ditemukan model."
            )
        },
        {
            "question": "Mengapa hasil Prediction bisa berbeda dengan tingkat Confidence?",
            "answer": (
                "Prediction menunjukkan keputusan biner akhir (membutuhkan terapi oksigen atau tidak), sedangkan "
                "Confidence mengukur seberapa dekat probabilitas dengan batas keputusan (50%). Misalnya, prediksi 'YES' dengan "
                "probabilitas 52% akan menghasilkan kategori 'Low Confidence' karena nilainya sangat dekat dengan batas netral 50%. "
                "Sebaliknya, probabilitas 96% akan diklasifikasikan sebagai 'Very High Confidence' karena model memiliki bukti kuat."
            )
        },
        {
            "question": "Mengapa nilai kontribusi SHAP bisa bernilai positif dan negatif?",
            "answer": (
                "Model Random Forest bekerja dengan menjumlahkan kontribusi dari berbagai fitur. Nilai SHAP positif "
                "menunjukkan bahwa fitur tersebut mendorong prediksi ke arah 'YES' (menaikkan probabilitas kebutuhan oksigen). "
                "Nilai SHAP negatif bekerja sebaliknya, mendorong prediksi ke arah 'NO' (menurunkan risiko). Ini mencerminkan "
                "keseimbangan antara faktor risiko (seperti sesak napas) dan faktor protektif (seperti saturasi oksigen normal)."
            )
        },
        {
            "question": "Apakah sistem CDSS OxyPredict ini menggantikan peran dokter?",
            "answer": (
                "Sama sekali tidak. OxyPredict dirancang sebagai Clinical Decision Support System (CDSS) untuk membantu "
                "prioritisasi asesmen pasien dan memberikan opini kedua berbasis bukti data historis. Keputusan akhir, diagnosis, "
                "dan rencana perawatan pasien sepenuhnya tetap menjadi tanggung jawab dokter atau tenaga medis yang berwenang."
            )
        },
        {
            "question": "Bagaimana jika data klinis pasien yang diinput tidak lengkap?",
            "answer": (
                "Pada Single Prediction, sistem mewajibkan pengisian seluruh input formulir (44 fitur) untuk menghindari "
                "prediksi yang tidak akurat. Untuk Batch Prediction, sistem melakukan validasi kolom. Jika ada fitur penting yang "
                "hilang, validasi akan gagal untuk melindungi integritas hasil analisis klinis."
            )
        },
        {
            "question": "Mengapa hasil prediksi dapat berubah drastis meskipun hanya satu input yang diubah?",
            "answer": (
                "Model machine learning menangkap hubungan non-linear yang kompleks antar variabel. Beberapa fitur vital seperti "
                "Saturasi Oksigen (SaO2) atau adanya Cyanosis memiliki bobot kontribusi (SHAP) yang sangat tinggi. Perubahan kecil "
                "pada fitur-fitur kritis ini dapat secara signifikan menggeser keseimbangan keputusan model."
            )
        }
    ]
