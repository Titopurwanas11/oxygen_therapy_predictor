"""
Clinical Guideline Utility for OxyPredict.
Provides educational content explaining prediction parameters and SHAP values.
"""

def get_prediction_guidelines() -> dict:
    """Get explanation details for reading model prediction parameters."""
    return {
        "yes_interpretation": {
            "title": "Prediction = YES",
            "desc": "Model memperkirakan pasien anak membutuhkan terapi oksigen segera. Ini didasarkan pada tanda-tanda klinis gangguan pernapasan atau desaturasi oksigen yang terdeteksi dalam data pasien.",
            "icon": "🔴"
        },
        "no_interpretation": {
            "title": "Prediction = NO",
            "desc": "Model memperkirakan pasien anak belum membutuhkan terapi oksigen saat ini. Pasien dikelompokkan ke dalam kategori observasi rutin atau perawatan suportif tanpa oksigen tambahan.",
            "icon": "🟢"
        },
        "probability_explanation": {
            "title": "Prediction Probability (%)",
            "desc": "Nilai persentase absolut yang dihasilkan oleh model Random Forest. Semakin mendekati 100%, semakin besar keyakinan model bahwa pasien membutuhkan terapi oksigen. Sebaliknya, semakin mendekati 0%, semakin yakin model bahwa oksigen tidak diperlukan.",
            "icon": "📊"
        },
        "confidence_levels": [
            {"level": "Very High", "range": ">= 95%", "desc": "Sangat yakin. Hasil prediksi didukung oleh data klinis yang sangat konsisten.", "icon": "🟢"},
            {"level": "High", "range": "90 - 95%", "desc": "Keyakinan tinggi. Prediksi sangat dapat diandalkan.", "icon": "🟢"},
            {"level": "Moderate", "range": "80 - 90%", "desc": "Keyakinan sedang. Disarankan untuk memantau tanda vital secara berkala.", "icon": "🟡"},
            {"level": "Low / Very Low", "range": "< 80%", "desc": "Keyakinan rendah. Keputusan klinis harus lebih mengandalkan pemeriksaan manual.", "icon": "🔴"}
        ],
        "risk_levels": [
            {"level": "Very High / High Risk", "range": "Probabilitas >= 70%", "desc": "Pasien memiliki risiko klinis tinggi mengalami gagal napas atau desaturasi. Diperlukan tindakan triage cepat.", "icon": "🔴"},
            {"level": "Moderate Risk", "range": "Probabilitas 50 - 70%", "desc": "Risiko sedang. Pasien harus dipantau secara ketat dalam 30-60 menit berikutnya.", "icon": "🟡"},
            {"level": "Low-Moderate / Low Risk", "range": "Probabilitas < 50%", "desc": "Risiko rendah. Pemulihan alami dengan pemantauan klinis standar biasanya mencukupi.", "icon": "🟢"}
        ]
    }


def get_shap_guidelines() -> dict:
    """Get explanation details for reading SHAP interpretability graphs."""
    return {
        "definition": (
            "SHAP (SHapley Additive exPlanations) adalah metode berbasis teori permainan kooperatif "
            "yang digunakan untuk menjelaskan kontribusi individual setiap fitur klinis terhadap hasil prediksi model."
        ),
        "visual_rules": [
            {
                "color": "Merah (SHAP Positif)",
                "desc": "Fitur klinis tersebut berkontribusi meningkatkan kemungkinan atau risiko pasien membutuhkan terapi oksigen. Contohnya: SaO2 yang sangat rendah, atau adanya gejala wheezing/nasal flaring.",
                "hex": "#ef4444"
            },
            {
                "color": "Biru (SHAP Negatif)",
                "desc": "Fitur klinis tersebut berkontribusi menurunkan kemungkinan atau risiko pasien membutuhkan terapi oksigen (atau mengarah pada keputusan 'No'). Contohnya: SaO2 normal atau tidak ada tanda distress pernapasan.",
                "hex": "#3b82f6"
            },
            {
                "color": "Panjang Batang (Magnitude)",
                "desc": "Semakin panjang batang horizontal pada grafik SHAP, semakin besar bobot atau pengaruh fitur klinis tersebut dalam mengubah keputusan model untuk pasien ini.",
                "hex": "#475569"
            }
        ],
        "example_case": (
            "<b>Contoh Interpretasi Pasien:</b><br/>"
            "Jika grafik SHAP menunjukkan fitur <code>Oxygen saturation (SaO2) at admission = 88%</code> berwarna merah "
            "dengan batang yang panjang, ini berarti desaturasi oksigen pasien sebesar 88% adalah faktor utama "
            "yang mendorong model memprediksi bahwa pasien tersebut membutuhkan terapi oksigen."
        )
    }
