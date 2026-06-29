"""
OxyPredict - Clinical Decision Support System (CDSS)
Main application entry point.
"""

import streamlit as st
from utils.config import (
    setup_page,
    render_page_header,
    render_section_divider,
    render_footer,
)

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Dashboard — OxyPredict",
    page_icon="assets/favicon-64x64.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_page("Dashboard — OxyPredict")

# Helper function to render HTML safely
def st_html(html_str):
    cleaned_lines = [line.lstrip() for line in html_str.splitlines()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)

# ─── SECTION 1: COMPACT CDSS HEADER ──────────────────────────────────────────
st_html(render_page_header(
    "",
    "Selamat Datang di OxyPredict",
    "Sistem Pendukung Keputusan Klinis (Clinical Decision Support System / CDSS) untuk Terapi Oksigen Pasien Anak."
))

# ─── SECTION 2: WELCOME CARD ───────────────────────────────────────────────
st_html("""
<div class="cdss-card" style="border-left: 5px solid #0F4C75;">
    <h3 style="margin-top: 0; color: #0F4C75; font-size: 20px; font-weight: 700;">Sistem Pendukung Keputusan Klinis</h3>
    <p style="color: #1E293B; font-size: 15px; line-height: 1.7; margin-bottom: 1rem;">
        OxyPredict dirancang untuk membantu dokter, perawat, dan tenaga kesehatan dalam memperkirakan kebutuhan terapi oksigen pada pasien anak dengan Infeksi Saluran Pernapasan Akut (ISPA) dan Pneumonia berdasarkan data klinis yang diinput.
    </p>
    <p style="color: #64748B; font-size: 14px; font-weight: 500; margin: 0;">
        👉 Gunakan menu di sebelah kiri untuk memulai <strong>Prediksi Pasien</strong>, melakukan <strong>Prediksi Massal</strong>, atau melihat <strong>Monitoring Prediksi</strong>.
    </p>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 3: SYSTEM PERFORMANCE & INFO ────────────────────────────────────
st_html("""
<h3 class="section-title-custom">Kinerja Model & Informasi Sistem</h3>
""")

col1, col2 = st.columns(2)

with col1:
    st_html("""
    <div class="cdss-card" style="height: 100%;">
        <h4 style="font-size: 18px; color: #0F172A; margin: 0 0 1.2rem 0; font-weight: 700;">Statistik Kinerja Model Prediksi</h4>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.75rem 0; color: #64748B; font-weight: 500;">Metode Algoritma</td>
                <td style="padding: 0.75rem 0; color: #0f172a; font-weight: 700; text-align: right;">Random Forest (800 Estimators)</td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.75rem 0; color: #64748B; font-weight: 500;">Akurasi Model (Accuracy)</td>
                <td style="padding: 0.75rem 0; color: #22C55E; font-weight: 700; text-align: right;">89.44%</td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.75rem 0; color: #64748B; font-weight: 500;">Skor F1 (Macro F1-Score)</td>
                <td style="padding: 0.75rem 0; color: #0f172a; font-weight: 700; text-align: right;">86.20%</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem 0; color: #64748B; font-weight: 500;">Area Di Bawah Kurva ROC (ROC AUC)</td>
                <td style="padding: 0.75rem 0; color: #0f172a; font-weight: 700; text-align: right;">90.93%</td>
            </tr>
        </table>
    </div>
    """)

with col2:
    st_html("""
    <div class="cdss-card" style="height: 100%;">
        <h4 style="font-size: 18px; color: #0F172A; margin: 0 0 1.2rem 0; font-weight: 700;">Informasi Basis Data Pengembangan</h4>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.75rem 0; color: #64748B; font-weight: 500;">Sumber Basis Data</td>
                <td style="padding: 0.75rem 0; color: #0f172a; font-weight: 700; text-align: right;">BD IRA (bd_raw.csv)</td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.75rem 0; color: #64748B; font-weight: 500;">Jumlah Data Sampel (Dataset)</td>
                <td style="padding: 0.75rem 0; color: #0f172a; font-weight: 700; text-align: right;">801 Pasien Anak</td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 0.75rem 0; color: #64748B; font-weight: 500;">Jumlah Parameter Awal</td>
                <td style="padding: 0.75rem 0; color: #0f172a; font-weight: 700; text-align: right;">91 Kolom Data</td>
            </tr>
            <tr>
                <td style="padding: 0.75rem 0; color: #64748B; font-weight: 500;">Parameter Klinis Terseleksi</td>
                <td style="padding: 0.75rem 0; color: #0f172a; font-weight: 700; text-align: right;">44 Fitur Utama</td>
            </tr>
        </table>
    </div>
    """)

st_html(render_section_divider())

# ─── SECTION 4: MEDICAL DISCLAIMER ──────────────────────────────────────────
st_html("""
<div class="cdss-card" style="border-left: 5px solid #EF4444; background-color: #FEF2F2;">
    <h4 style="margin-top: 0; color: #B91C1C; font-size: 16px; font-weight: 700;">
        Peringatan Medis (Medical Disclaimer)
    </h4>
    <p style="color: #991B1B; font-size: 14px; line-height: 1.6; margin: 0; font-weight: 500;">
        OxyPredict merupakan Sistem Pendukung Keputusan Klinis (Clinical Decision Support System/CDSS) yang dirancang untuk membantu tenaga kesehatan dalam memperkirakan kebutuhan terapi oksigen berdasarkan data klinis pasien.
        <br/><br/>
        Hasil prediksi bersifat sebagai informasi pendukung dan <strong>tidak menggantikan penilaian klinis, diagnosis, maupun keputusan akhir dokter atau tenaga kesehatan yang berwenang</strong>.
    </p>
</div>
""")

st.markdown("")

# ─── Footer ──────────────────────────────────────────────────────────────────
st_html(render_footer())
