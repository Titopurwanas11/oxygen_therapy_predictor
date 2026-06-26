"""
SHAP Explanation utilities for the Oxygen Therapy Predictor.
Contains metadata, descriptions, and check functions for the SHAP visualizations.
"""

import os
from utils.config import ASSETS_DIR

def get_shap_plots_metadata() -> list:
    """
    Get metadata and descriptions for the SHAP plots.
    
    Returns:
        List of dictionaries containing plot title, filename, descriptions, and styling.
    """
    return [
        {
            "title": "📊 SHAP Bar Plot — Feature Importance",
            "filename": "shap_bar_plot.png",
            "description": """
            **SHAP Bar Plot** menampilkan rata-rata nilai absolut SHAP untuk setiap fitur,
            yang menunjukkan **tingkat kepentingan (importance)** masing-masing fitur dalam model.
            Fitur dengan bar lebih panjang memiliki pengaruh lebih besar terhadap keputusan prediksi model.

            **Cara membaca:**
            - Sumbu X menunjukkan rata-rata |SHAP value|
            - Sumbu Y menampilkan nama fitur, diurutkan dari yang paling penting
            - Fitur di posisi atas adalah fitur yang paling berpengaruh dalam model
            """,
            "color_start": "#1a73e8",
            "color_end": "#1557b0",
            "bg": "linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)",
            "border": "#bfdbfe",
        },
        {
            "title": "🎯 SHAP Summary Plot — Feature Impact",
            "filename": "shap_summary_plot.png",
            "description": """
            **SHAP Summary Plot (Beeswarm)** menampilkan distribusi SHAP values untuk setiap fitur
            di seluruh dataset. Plot ini menunjukkan **arah dan besaran pengaruh** setiap fitur.

            **Cara membaca:**
            - Setiap titik mewakili satu data pasien
            - Warna menunjukkan nilai fitur: <span style="color: #dc2626; font-weight: 700;">merah = tinggi</span>,
              <span style="color: #2563eb; font-weight: 700;">biru = rendah</span>
            - Posisi horizontal menunjukkan dampak terhadap prediksi
            - Titik di kanan garis nol → mendorong prediksi ke "Butuh Terapi Oksigen"
            - Titik di kiri garis nol → mendorong prediksi ke "Tidak Butuh Terapi Oksigen"
            """,
            "color_start": "#059669",
            "color_end": "#047857",
            "bg": "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
            "border": "#bbf7d0",
        },
        {
            "title": "🔬 SHAP Waterfall Plot — Individual Explanation",
            "filename": "shap_waterfall.png",
            "description": """
            **SHAP Waterfall Plot** menjelaskan prediksi untuk **satu pasien tertentu** secara detail.
            Plot ini menampilkan bagaimana setiap fitur mendorong prediksi dari nilai dasar (base value)
            menuju nilai prediksi akhir.

            **Cara membaca:**
            - Base value (E[f(x)]) adalah rata-rata prediksi model
            - Bar <span style="color: #dc2626; font-weight: 700;">merah</span> menunjukkan fitur
              yang **meningkatkan** probabilitas terapi oksigen
            - Bar <span style="color: #2563eb; font-weight: 700;">biru</span> menunjukkan fitur
              yang **menurunkan** probabilitas terapi oksigen
            - f(x) di bagian atas adalah nilai prediksi akhir
            """,
            "color_start": "#7c3aed",
            "color_end": "#6d28d9",
            "bg": "linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%)",
            "border": "#d8b4fe",
        },
    ]

def get_plot_path(filename: str) -> str:
    """Get the absolute path to a SHAP plot image."""
    return os.path.join(ASSETS_DIR, filename)

def check_plot_exists(filename: str) -> bool:
    """Check if the SHAP plot image exists in the assets directory."""
    return os.path.exists(get_plot_path(filename))
