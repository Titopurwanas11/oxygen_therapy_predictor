"""
About Model Page — Model Evaluation and Methodology Dashboard.
"""

import os
import streamlit as st
import pandas as pd
from utils.config import (
    MODEL_NAME,
    MODEL_N_ESTIMATORS,
    MODEL_ACCURACY,
    MODEL_F1_MACRO,
    MODEL_ROC_AUC,
    DATASET_NAME,
    DATASET_TOTAL_SAMPLES,
    DATASET_SELECTED_FEATURES,
    ASSETS_DIR,
    setup_page,
)

st.set_page_config(page_title="About Model — OxyPredict", page_icon="🫁", layout="wide")
setup_page("About Model — OxyPredict")

# Helper function to strip all leading whitespace from every line of HTML.
# This prevents the Streamlit markdown engine from parsing nested HTML/SVG tags
# as indented code blocks (which triggers when lines start with 4 or more spaces).
def st_html(html_str):
    cleaned_lines = [line.lstrip() for line in html_str.splitlines()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)

# ─── Custom CSS Styling ───────────────────────────────────────────────────────
st_html("""
<style>
    /* Styling Dashboard */
    .dashboard-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header styling */
    .header-card {
        background: linear-gradient(135deg, #0a2e52 0%, #153e75 50%, #1e40af 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(10, 46, 82, 0.15);
        color: white;
    }
    .header-title {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        margin: 0.5rem 0 0 0;
        color: #93c5fd;
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Section titles */
    .section-header {
        color: #0a2e52;
        font-weight: 700;
        font-size: 1.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
        padding-left: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Overview Metric Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    @media (max-width: 992px) {
        .metric-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    @media (max-width: 600px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
    
    .metric-card-custom {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03), 0 2px 4px -1px rgba(0,0,0,0.01);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        transition: all 0.25s ease;
    }
    .metric-card-custom:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        border-color: #3b82f6;
    }
    .metric-card-icon {
        font-size: 1.8rem;
        width: 46px;
        height: 46px;
        background: #eff6ff;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #1d4ed8;
        flex-shrink: 0;
    }
    .metric-card-info {
        display: flex;
        flex-direction: column;
    }
    .metric-card-label {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.15rem;
    }

    /* Content Cards */
    .panel-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.01);
        height: 100%;
        display: flex;
        flex-direction: column;
        margin-bottom: 1.5rem;
    }
    .panel-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0a2e52;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.5rem;
    }

    /* Confusion Matrix fallback styling */
    .cm-grid-container {
        display: grid;
        grid-template-columns: 80px 1fr 1fr;
        gap: 8px;
        text-align: center;
        font-size: 0.8rem;
        margin: 1rem 0;
    }
    .cm-header-label {
        font-weight: 700;
        color: #475569;
        padding: 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .cm-cell-label {
        font-weight: 700;
        color: #475569;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: right;
        padding-right: 0.5rem;
    }
    .cm-box-tn {
        background-color: #f0fdf4;
        border: 2px solid #bbf7d0;
        border-radius: 10px;
        padding: 1.25rem 0.5rem;
        font-weight: 800;
        color: #15803d;
        font-size: 1.2rem;
        transition: all 0.2s ease;
    }
    .cm-box-tp {
        background-color: #f0fdf4;
        border: 2px solid #bbf7d0;
        border-radius: 10px;
        padding: 1.25rem 0.5rem;
        font-weight: 800;
        color: #15803d;
        font-size: 1.2rem;
        transition: all 0.2s ease;
    }
    .cm-box-fp {
        background-color: #fff1f2;
        border: 2px solid #fecaca;
        border-radius: 10px;
        padding: 1.25rem 0.5rem;
        font-weight: 800;
        color: #b91c1c;
        font-size: 1.2rem;
        transition: all 0.2s ease;
    }
    .cm-box-fn {
        background-color: #fff1f2;
        border: 2px solid #fecaca;
        border-radius: 10px;
        padding: 1.25rem 0.5rem;
        font-weight: 800;
        color: #b91c1c;
        font-size: 1.2rem;
        transition: all 0.2s ease;
    }
    .cm-box-tn:hover, .cm-box-tp:hover, .cm-box-fp:hover, .cm-box-fn:hover {
        transform: scale(1.03);
    }
    .cm-cell-subtitle {
        font-size: 0.65rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 0.25rem;
    }

    /* Explanation lists */
    .explanation-list {
        margin-top: 0.75rem;
        padding-left: 1.2rem;
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.6;
    }
    .explanation-list li {
        margin-bottom: 0.5rem;
    }

    /* ROC AUC Card */
    .roc-auc-card {
        background: linear-gradient(135deg, #0a2e52 0%, #1e3a8a 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 20px rgba(10, 46, 82, 0.15);
        margin-bottom: 1.25rem;
    }
    .roc-auc-title {
        font-size: 0.9rem;
        color: #93c5fd;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .roc-auc-value {
        font-size: 3rem;
        font-weight: 800;
        color: #60a5fa;
        margin: 0.5rem 0;
        text-shadow: 0 2px 10px rgba(96,165,250,0.3);
    }
    .roc-auc-desc {
        font-size: 0.78rem;
        color: #cbd5e1;
        margin: 0;
        line-height: 1.5;
    }

    /* Table styling */
    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.01);
    }
    .report-table th {
        background-color: #0a2e52;
        color: #ffffff;
        text-align: left;
        padding: 0.8rem 1rem;
        font-weight: 600;
    }
    .report-table td {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #e2e8f0;
        color: #334155;
    }
    .report-table tr:last-child td {
        border-bottom: none;
    }
    .report-table tr:hover {
        background-color: #f8fafc;
    }
    .report-table .class-name {
        font-weight: 600;
        color: #0f172a;
    }
    .report-table .metric-val {
        font-family: 'Courier New', Courier, monospace;
        font-weight: 600;
    }
    .report-table .high-score {
        color: #16a34a;
        font-weight: 700;
    }
    .report-table .mid-score {
        color: #2563eb;
        font-weight: 700;
    }
    .report-table .summary-row {
        background-color: #f1f5f9;
        font-weight: 600;
    }
    .report-table .summary-row td {
        color: #0f172a;
    }

    /* Academic card */
    .academic-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.05);
        margin-top: 1.5rem;
    }
    .academic-title {
        margin: 0 0 0.5rem 0;
        color: #0a2e52;
        font-weight: 700;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .academic-body {
        margin: 0;
        color: #1e3a8a;
        font-size: 0.9rem;
        line-height: 1.7;
        font-style: italic;
    }
    .academic-meta {
        margin-top: 0.75rem;
        font-size: 0.8rem;
        color: #475569;
        font-weight: 600;
        text-align: right;
    }
</style>
""")

# ─── Header ──────────────────────────────────────────────────────────────────
st_html("""
<div class="header-card">
    <h1 class="header-title">ℹ️ Evaluasi & Metrik Model</h1>
    <p class="header-subtitle">Dokumentasi hasil evaluasi model Random Forest dan interpretasi akademik untuk presentasi skripsi</p>
</div>
""")

# ─── Section 1: Model Overview ───────────────────────────────────────────────
st_html(f"""
<h3 class="section-header">
    ⚙️ Model Overview
</h3>
""")

st_html(f"""
<div class="metric-grid">
    <div class="metric-card-custom">
        <div class="metric-card-icon">🤖</div>
        <div class="metric-card-info">
            <div class="metric-card-label">Algorithm</div>
            <div class="metric-card-value">{MODEL_NAME}</div>
        </div>
    </div>
    <div class="metric-card-custom">
        <div class="metric-card-icon">🌲</div>
        <div class="metric-card-info">
            <div class="metric-card-label">Estimators</div>
            <div class="metric-card-value">{MODEL_N_ESTIMATORS} Trees</div>
        </div>
    </div>
    <div class="metric-card-custom">
        <div class="metric-card-icon">👥</div>
        <div class="metric-card-info">
            <div class="metric-card-label">Dataset Samples</div>
            <div class="metric-card-value">{DATASET_TOTAL_SAMPLES} Patients</div>
        </div>
    </div>
    <div class="metric-card-custom">
        <div class="metric-card-icon">📊</div>
        <div class="metric-card-info">
            <div class="metric-card-label">Selected Features</div>
            <div class="metric-card-value">{DATASET_SELECTED_FEATURES} Features</div>
        </div>
    </div>
    <div class="metric-card-custom">
        <div class="metric-card-icon">🫁</div>
        <div class="metric-card-info">
            <div class="metric-card-label">Target Variable</div>
            <div class="metric-card-value">Oxygen Therapy</div>
        </div>
    </div>
</div>
""")

# ─── Section 2 & 3: Confusion Matrix & ROC Curve ──────────────────────────────
st_html("""
<h3 class="section-header">
    📈 Evaluasi Klasifikasi & Kemampuan Diskriminasi
</h3>
""")

col_left, col_right = st.columns(2)

with col_left:
    st_html('<div class="panel-card">')
    st_html('<div class="panel-card-title">🎯 Confusion Matrix</div>')
    
    cm_img_path = os.path.join(ASSETS_DIR, "confusion_matrix.png")
    if os.path.exists(cm_img_path):
        st.image(cm_img_path, use_container_width=True, caption="Confusion Matrix Model pada Test Set")
    else:
        st_html("""
            <div style="background-color: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; text-align: center;">
                <p style="font-size: 0.78rem; color: #64748b; margin: 0 0 0.8rem 0; font-style: italic;">
                    ℹ️ File <code>assets/confusion_matrix.png</code> tidak ditemukan. Menampilkan visualisasi data uji (N = 161).
                </p>
                <div class="cm-grid-container">
                    <div></div>
                    <div class="cm-header-label">Predicted NO</div>
                    <div class="cm-header-label">Predicted YES</div>
                    
                    <div class="cm-cell-label">Actual NO</div>
                    <div class="cm-box-tn">
                        35
                        <div class="cm-cell-subtitle">True Negative (TN)</div>
                    </div>
                    <div class="cm-box-fp">
                        5
                        <div class="cm-cell-subtitle">False Positive (FP)</div>
                    </div>
                    
                    <div class="cm-cell-label">Actual YES</div>
                    <div class="cm-box-fn">
                        12
                        <div class="cm-cell-subtitle">False Negative (FN)</div>
                    </div>
                    <div class="cm-box-tp">
                        109
                        <div class="cm-cell-subtitle">True Positive (TP)</div>
                    </div>
                </div>
            </div>
            """)
        
    st_html("""
        <div style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
            <p style="margin-top: 0; font-weight: 600; color: #0a2e52;">Penjelasan Metrik:</p>
            <ul class="explanation-list">
                <li><strong>True Positive (TP) = 109:</strong> Pasien yang secara medis membutuhkan oksigen dan diprediksi dengan benar oleh model.</li>
                <li><strong>True Negative (TN) = 35:</strong> Pasien stabil yang diprediksi tidak membutuhkan oksigen dengan benar.</li>
                <li><strong>False Positive (FP) = 5:</strong> Pasien stabil tetapi diprediksi butuh terapi oksigen (kesalahan tipe I / alarm palsu).</li>
                <li><strong>False Negative (FN) = 12:</strong> Pasien butuh oksigen tetapi diprediksi stabil (kesalahan tipe II / diabaikan). Kondisi ini memiliki risiko klinis tertinggi.</li>
            </ul>
        </div>
        """)
    st_html('</div>')

with col_right:
    st_html('<div class="panel-card">')
    st_html('<div class="panel-card-title">📈 ROC Curve & AUC Score</div>')
    
    roc_img_path = os.path.join(ASSETS_DIR, "roc_curve.png")
    if os.path.exists(roc_img_path):
        st.image(roc_img_path, use_container_width=True, caption="ROC-AUC Curve Model pada Test Set")
    else:
        st_html(f"""
            <div style="display: flex; flex-direction: column; gap: 0.75rem; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <div class="roc-auc-card" style="width: 100%; box-sizing: border-box;">
                    <div class="roc-auc-title">ROC-AUC SCORE</div>
                    <div class="roc-auc-value">{MODEL_ROC_AUC}%</div>
                    <div class="roc-auc-desc">
                        ℹ️ File <code>assets/roc_curve.png</code> tidak ditemukan. Menampilkan kurva estimasi akademis.
                    </div>
                </div>
                
                <div style="background-color: #ffffff; border: 1px solid #f1f5f9; border-radius: 12px; padding: 0.5rem; width: 100%; display: flex; justify-content: center;">
                    <svg viewBox="0 0 200 200" width="100%" height="auto" style="max-width: 160px; display: block;">
                        <!-- Grid Lines -->
                        <line x1="30" y1="20" x2="30" y2="170" stroke="#cbd5e1" stroke-width="1" />
                        <line x1="30" y1="170" x2="180" y2="170" stroke="#cbd5e1" stroke-width="1" />
                        <line x1="30" y1="20" x2="180" y2="20" stroke="#f1f5f9" stroke-dasharray="2,2" stroke-width="0.75" />
                        <line x1="180" y1="20" x2="180" y2="170" stroke="#f1f5f9" stroke-dasharray="2,2" stroke-width="0.75" />
                        
                        <!-- Diagonal Baseline -->
                        <line x1="30" y1="170" x2="180" y2="20" stroke="#94a3b8" stroke-dasharray="3,3" stroke-width="1" />
                        
                        <!-- Shaded ROC area -->
                        <path d="M 30 170 C 40 100, 60 30, 180 20 L 180 170 Z" fill="rgba(59, 130, 246, 0.08)" />
                        
                        <!-- ROC Curve -->
                        <path d="M 30 170 C 40 100, 60 30, 180 20" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" />
                        
                        <!-- Highlight dot -->
                        <circle cx="55" cy="38" r="3" fill="#1d4ed8" />
                        
                        <!-- Axes Titles -->
                        <text x="105" y="185" fill="#64748b" font-size="6" text-anchor="middle" font-family="sans-serif">False Positive Rate (1 - Spec)</text>
                        <text x="10" y="95" fill="#64748b" font-size="6" text-anchor="middle" font-family="sans-serif" transform="rotate(-90,10,95)">True Positive Rate (Sens)</text>
                        
                        <!-- Values -->
                        <text x="25" y="173" fill="#94a3b8" font-size="6" text-anchor="end" font-family="sans-serif">0</text>
                        <text x="180" y="178" fill="#94a3b8" font-size="6" text-anchor="middle" font-family="sans-serif">1</text>
                        <text x="25" y="23" fill="#94a3b8" font-size="6" text-anchor="end" font-family="sans-serif">1</text>
                        
                        <text x="110" y="100" fill="#2563eb" font-size="7" text-anchor="middle" font-family="sans-serif" font-weight="bold">AUC = 90.93%</text>
                    </svg>
                </div>
            </div>
            """)

    st_html("""
        <div style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
            <p style="margin-top: 0; font-weight: 600; color: #0a2e52;">Interpretasi Kurva ROC & AUC:</p>
            <p style="margin: 0.3rem 0; font-size: 0.82rem;">
                Kurva <strong>ROC (Receiver Operating Characteristic)</strong> menggambarkan kinerja model di seluruh ambang klasifikasi. Nilai <strong>AUC (Area Under the Curve)</strong> sebesar <strong>90.93%</strong> menunjukkan model memiliki kemampuan diskriminasi yang sangat baik dalam membedakan pasien yang membutuhkan terapi oksigen dari yang tidak.
            </p>
        </div>
        """)
    st_html('</div>')

# ─── Section 4: Classification Report ────────────────────────────────────────
st_html("""
<h3 class="section-header">
    📊 Classification Report
</h3>
""")

st_html(f"""
<div class="panel-card">
    <div class="panel-card-title">Laporan Klasifikasi Terperinci</div>
    <table class="report-table">
        <thead>
            <tr>
                <th>Kelas / Metrik</th>
                <th style="text-align: center;">Precision</th>
                <th style="text-align: center;">Recall</th>
                <th style="text-align: center;">F1-Score</th>
                <th style="text-align: center;">Support</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="class-name">❌ No Oxygen Therapy</td>
                <td style="text-align: center;" class="metric-val">74.47%</td>
                <td style="text-align: center;" class="metric-val">87.50%</td>
                <td style="text-align: center;" class="metric-val">80.46%</td>
                <td style="text-align: center; color: #64748b; font-family: monospace;">40</td>
            </tr>
            <tr>
                <td class="class-name">✅ Need Oxygen Therapy</td>
                <td style="text-align: center;" class="metric-val high-score">95.61%</td>
                <td style="text-align: center;" class="metric-val high-score">90.08%</td>
                <td style="text-align: center;" class="metric-val high-score">92.77%</td>
                <td style="text-align: center; color: #64748b; font-family: monospace;">121</td>
            </tr>
            <tr class="summary-row">
                <td class="class-name">📊 Accuracy (Global)</td>
                <td colspan="3" style="text-align: center; font-weight: 700; color: #0a2e52;">{MODEL_ACCURACY}%</td>
                <td style="text-align: center; color: #475569; font-family: monospace;">161</td>
            </tr>
            <tr class="summary-row">
                <td class="class-name">⚖️ Macro Average</td>
                <td style="text-align: center;" class="metric-val">85.04%</td>
                <td style="text-align: center;" class="metric-val">88.79%</td>
                <td style="text-align: center;" class="metric-val">{MODEL_F1_MACRO}%</td>
                <td style="text-align: center; color: #475569; font-family: monospace;">161</td>
            </tr>
            <tr class="summary-row">
                <td class="class-name">💼 Weighted Average</td>
                <td style="text-align: center;" class="metric-val">90.34%</td>
                <td style="text-align: center;" class="metric-val">89.44%</td>
                <td style="text-align: center;" class="metric-val">89.71%</td>
                <td style="text-align: center; color: #475569; font-family: monospace;">161</td>
            </tr>
        </tbody>
    </table>
</div>
""")

# ─── Section 5: Academic Interpretation ──────────────────────────────────────
st_html("""
<h3 class="section-header">
    🎓 Academic Interpretation
</h3>
""")

st_html("""
<div class="academic-card">
    <div class="academic-title">
        🎓 Analisis & Kesimpulan Akademis
    </div>
    <div class="academic-body">
        "The model achieved strong predictive performance with ROC-AUC above 0.90, indicating excellent discrimination capability between patients who require oxygen therapy and those who do not."
    </div>
    <div style="margin-top: 1rem; font-size: 0.85rem; color: #334155; line-height: 1.6;">
        Berdasarkan metrik evaluasi di atas, model klasifikasi berbasis <strong>Random Forest</strong> menunjukkan performa yang kokoh dan dapat diandalkan secara akademis.
        Tingginya nilai <strong>ROC-AUC (90.93%)</strong> menunjukkan model memiliki kemampuan diskriminasi yang sangat baik dalam memilah pasien yang stabil vs. kritis.
        Selain itu, metrik <strong>Recall kelas Need Oxygen sebesar 90.08%</strong> membuktikan sensitivitas model yang tinggi terhadap deteksi kondisi kritis pasien anak, meminimalkan peluang tidak terpantaunya pasien yang membutuhkan bantuan medis segera.
    </div>
    <div class="academic-meta">
        — Panel Evaluasi OxyPredict • Bidang Keahlian CDSS & Machine Learning
    </div>
</div>
""")

st_html("<br><br>")

# ─── Footer ──────────────────────────────────────────────────────────────────
st_html("""
<div style="
    text-align: center;
    padding: 1.5rem 0;
    color: #94a3b8;
    font-size: 0.75rem;
    border-top: 1px solid #e2e8f0;
">
    <p style="margin: 0;">
        <strong>OxyPredict</strong> — CDSS Model Evaluation Dashboard
    </p>
    <p style="margin: 0.2rem 0 0 0;">
        Dibuat menggunakan Streamlit • scikit-learn • SHAP
    </p>
</div>
""")
