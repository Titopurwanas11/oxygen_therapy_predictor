"""
OxyPredict - Sistem Prediksi Kebutuhan Terapi Oksigen
Main application entry point.

Prediksi Kebutuhan Terapi Oksigen pada Pasien Anak
dengan ISPA dan Pneumonia Menggunakan Random Forest dan SHAP.

Run with: streamlit run app.py
"""

import streamlit as st
from utils.config import setup_page

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="OxyPredict — Prediksi Terapi Oksigen",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_page("OxyPredict — Prediksi Terapi Oksigen")

# =============================================================================
# Home Page Content
# =============================================================================
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0;">
    <div style="
        display: inline-block;
        font-size: 4rem;
        margin-bottom: 0.5rem;
        animation: pulse 2s ease-in-out infinite;
    ">🫁</div>
    <h1 style="
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0a2e52 0%, #1a73e8 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    ">OxyPredict</h1>
    <p style="
        margin: 0.5rem auto 0 auto;
        max-width: 640px;
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.6;
    ">
        Prediksi Kebutuhan Terapi Oksigen pada Pasien Anak<br>
        dengan <strong>ISPA</strong> dan <strong>Pneumonia</strong>
        Menggunakan <strong>Random Forest</strong> dan <strong>SHAP</strong>
    </p>
</div>

<style>
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.08); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("")

# Hero cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    ">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🩺</div>
        <h3 style="margin: 0 0 0.5rem 0; color: #1e3a5f; font-weight: 700; font-size: 1.1rem;">
            Prediksi Pasien
        </h3>
        <p style="margin: 0; color: #64748b; font-size: 0.85rem; line-height: 1.5;">
            Masukkan data klinis pasien untuk mendapatkan prediksi kebutuhan terapi oksigen secara real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    ">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
        <h3 style="margin: 0 0 0.5rem 0; color: #14532d; font-weight: 700; font-size: 1.1rem;">
            Batch Prediction
        </h3>
        <p style="margin: 0; color: #64748b; font-size: 0.85rem; line-height: 1.5;">
            Upload file CSV atau XLSX untuk memprediksi kebutuhan terapi oksigen secara massal.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #fcd34d;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    ">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔍</div>
        <h3 style="margin: 0 0 0.5rem 0; color: #78350f; font-weight: 700; font-size: 1.1rem;">
            Interpretasi SHAP
        </h3>
        <p style="margin: 0; color: #64748b; font-size: 0.85rem; line-height: 1.5;">
            Pahami alasan di balik prediksi model melalui visualisasi SHAP yang interaktif.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown("")

# Quick info bar
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0a2e52 0%, #1a4a7a 100%);
    border-radius: 14px;
    padding: 1.3rem 2rem;
    display: flex;
    justify-content: space-around;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    box-shadow: 0 4px 20px rgba(10, 46, 82, 0.2);
">
    <div style="text-align: center;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #60a5fa;">801</div>
        <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">
            Data Pasien
        </div>
    </div>
    <div style="width: 1px; height: 35px; background: rgba(148, 163, 184, 0.3);"></div>
    <div style="text-align: center;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #34d399;">44</div>
        <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">
            Fitur Klinis
        </div>
    </div>
    <div style="width: 1px; height: 35px; background: rgba(148, 163, 184, 0.3);"></div>
    <div style="text-align: center;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #fbbf24;">89.44%</div>
        <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">
            Accuracy
        </div>
    </div>
    <div style="width: 1px; height: 35px; background: rgba(148, 163, 184, 0.3);"></div>
    <div style="text-align: center;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #f472b6;">Random Forest</div>
        <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">
            Model
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# Footer
st.markdown("""
<div style="
    text-align: center;
    padding: 1.5rem 0 1rem 0;
    color: #94a3b8;
    font-size: 0.75rem;
">
    <p style="margin: 0;">
        Dibuat untuk keperluan <strong>Skripsi</strong> — Sistem Pendukung Keputusan Klinis
    </p>
    <p style="margin: 0.2rem 0 0 0;">
        Menggunakan Streamlit • scikit-learn • SHAP
    </p>
</div>
""", unsafe_allow_html=True)
