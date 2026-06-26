"""
SHAP Explanation Page — Model interpretability visualizations.
"""

import streamlit as st
from utils.config import setup_page
from utils.shap_utils import get_shap_plots_metadata, get_plot_path, check_plot_exists

st.set_page_config(page_title="SHAP Explanation — OxyPredict", page_icon="🫁", layout="wide")
setup_page("SHAP Explanation — OxyPredict")

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0a2e52 0%, #1a4a7a 50%, #2563eb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(10, 46, 82, 0.18);
">
    <h1 style="
        margin: 0;
        color: white;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.3px;
    ">🔍 SHAP Explanation</h1>
    <p style="
        margin: 0.3rem 0 0 0;
        color: #93c5fd;
        font-size: 0.95rem;
    ">Interpretasi model menggunakan SHapley Additive exPlanations (SHAP)</p>
</div>
""", unsafe_allow_html=True)

# ─── What is SHAP ───────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #93c5fd;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
">
    <h4 style="margin: 0 0 0.5rem 0; color: #1e3a5f; font-weight: 700; font-size: 1rem;">
        💡 Apa itu SHAP?
    </h4>
    <p style="margin: 0; color: #475569; font-size: 0.85rem; line-height: 1.7;">
        <strong>SHAP (SHapley Additive exPlanations)</strong> adalah metode interpretasi model berbasis teori
        permainan kooperatif (Shapley values). SHAP menjelaskan prediksi sebuah model dengan mengukur
        kontribusi setiap fitur terhadap output prediksi. Nilai SHAP positif mendorong prediksi ke arah
        kelas positif (membutuhkan terapi oksigen), sedangkan nilai negatif mendorong ke arah kelas negatif.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── SHAP Visualizations ────────────────────────────────────────────────────

# Get SHAP plots metadata
shap_plots = get_shap_plots_metadata()

for plot in shap_plots:
    st.markdown(f"""
    <div style="
        background: {plot['bg']};
        border: 1px solid {plot['border']};
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <h3 style="
            margin: 0 0 0.8rem 0;
            background: linear-gradient(135deg, {plot['color_start']}, {plot['color_end']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.15rem;
            font-weight: 700;
        ">{plot['title']}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Check if image exists
    filename = plot["filename"]

    if check_plot_exists(filename):
        st.image(
            get_plot_path(filename),
            use_container_width=True,
            caption=plot["title"].split("—")[1].strip() if "—" in plot["title"] else "",
        )
    else:
        st.markdown(f"""
        <div style="
            background: #f8fafc;
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 3rem 2rem;
            text-align: center;
            margin-bottom: 0.5rem;
        ">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🖼️</div>
            <p style="margin: 0; color: #64748b; font-weight: 600; font-size: 0.9rem;">
                Gambar belum tersedia
            </p>
            <p style="margin: 0.3rem 0 0 0; color: #94a3b8; font-size: 0.78rem;">
                Simpan file <code>{filename}</code> ke folder <code>assets/</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Description
    st.markdown(f"""
    <div style="
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-bottom: 1.5rem;
    ">
        <p style="margin: 0; color: #334155; font-size: 0.85rem; line-height: 1.7;">
            {plot['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── SHAP Interpretation Guide ──────────────────────────────────────────────
st.markdown("")
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0a2e52, #1a4a7a);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    box-shadow: 0 4px 20px rgba(10, 46, 82, 0.2);
">
    <h3 style="color: white; font-weight: 700; margin: 0 0 1rem 0; font-size: 1.1rem;">
        📚 Panduan Interpretasi SHAP
    </h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div style="
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 1rem;
        ">
            <p style="color: #93c5fd; font-weight: 600; font-size: 0.85rem; margin: 0 0 0.3rem 0;">
                ➕ SHAP Value Positif
            </p>
            <p style="color: #cbd5e1; font-size: 0.78rem; margin: 0; line-height: 1.5;">
                Mendorong prediksi ke arah "Membutuhkan Terapi Oksigen". Semakin besar nilainya,
                semakin kuat pengaruhnya.
            </p>
        </div>
        <div style="
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 1rem;
        ">
            <p style="color: #93c5fd; font-weight: 600; font-size: 0.85rem; margin: 0 0 0.3rem 0;">
                ➖ SHAP Value Negatif
            </p>
            <p style="color: #cbd5e1; font-size: 0.78rem; margin: 0; line-height: 1.5;">
                Mendorong prediksi ke arah "Tidak Membutuhkan Terapi Oksigen". Semakin kecil nilainya,
                semakin kuat pengaruhnya.
            </p>
        </div>
        <div style="
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 1rem;
        ">
            <p style="color: #93c5fd; font-weight: 600; font-size: 0.85rem; margin: 0 0 0.3rem 0;">
                🎯 Base Value
            </p>
            <p style="color: #cbd5e1; font-size: 0.78rem; margin: 0; line-height: 1.5;">
                Rata-rata prediksi model di seluruh dataset. Merupakan titik awal sebelum
                kontribusi fitur diperhitungkan.
            </p>
        </div>
        <div style="
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 1rem;
        ">
            <p style="color: #93c5fd; font-weight: 600; font-size: 0.85rem; margin: 0 0 0.3rem 0;">
                📊 Feature Importance
            </p>
            <p style="color: #cbd5e1; font-size: 0.78rem; margin: 0; line-height: 1.5;">
                Dihitung dari rata-rata |SHAP value| per fitur. Menunjukkan seberapa besar pengaruh
                rata-rata fitur terhadap prediksi.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
