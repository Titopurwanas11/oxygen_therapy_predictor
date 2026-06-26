"""
Clinical Guideline & Model Interpretation Page for OxyPredict.
Serves as an educational hub and documentation center for clinicians using the CDSS.
"""

import streamlit as st
import pandas as pd

from utils.config import setup_page
from utils.clinical_reference import get_clinical_reference_df
from utils.guideline import get_prediction_guidelines, get_shap_guidelines
from utils.faq import get_faqs_list
from utils.model_pipeline import get_pipeline_steps

st.set_page_config(page_title="clinical Guideline — OxyPredict", page_icon="📚", layout="wide")
setup_page("Clinical Guideline — OxyPredict")

# Helper function to render HTML safely
def st_html(html_str):
    cleaned_lines = [line.lstrip() for line in html_str.splitlines()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)

# Custom Glass Card CSS Styles
st_html("""
<style>
    .cdss-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.25rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .cdss-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(10, 46, 82, 0.04);
    }
    .section-title-custom {
        color: #0a2e52;
        font-weight: 700;
        font-size: 1.3rem;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2563eb;
        padding-left: 0.6rem;
    }
    .badge-rule {
        display: inline-block;
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        font-size: 0.7rem;
        font-weight: 800;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
</style>
""")

# ─── STICKY NAVIGATION BAR ──────────────────────────────────────────────────
st_html("""
<div style="
    position: -webkit-sticky;
    position: sticky;
    top: 0;
    background-color: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border: 1px solid #e2e8f0;
    padding: 0.6rem 1rem;
    z-index: 999;
    display: flex;
    justify-content: space-around;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
">
    <a href="#about-cdss" style="text-decoration: none; color: #0a2e52; font-weight: 700; font-size: 0.8rem;">📖 About</a>
    <a href="#clinical-workflow" style="text-decoration: none; color: #0a2e52; font-weight: 700; font-size: 0.8rem;">⏱️ Workflow</a>
    <a href="#how-to-read-prediction" style="text-decoration: none; color: #0a2e52; font-weight: 700; font-size: 0.8rem;">🔮 Prediction</a>
    <a href="#how-to-read-shap" style="text-decoration: none; color: #0a2e52; font-weight: 700; font-size: 0.8rem;">🧠 SHAP</a>
    <a href="#feature-reference" style="text-decoration: none; color: #0a2e52; font-weight: 700; font-size: 0.8rem;">📋 Features</a>
    <a href="#model-pipeline" style="text-decoration: none; color: #0a2e52; font-weight: 700; font-size: 0.8rem;">🔄 Pipeline</a>
    <a href="#model-performance" style="text-decoration: none; color: #0a2e52; font-weight: 700; font-size: 0.8rem;">📈 Performance</a>
    <a href="#faq" style="text-decoration: none; color: #0a2e52; font-weight: 700; font-size: 0.8rem;">❓ FAQ</a>
</div>
""")

# ─── SECTION 1: HERO HEADER ──────────────────────────────────────────────────
st_html("""
<div style="
    background: linear-gradient(135deg, #0a2e52 0%, #1a4a7a 50%, #2563eb 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(10, 46, 82, 0.18);
">
    <h1 style="margin: 0; color: white; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.3px; display: flex; align-items: center; gap: 0.6rem;">
        📚 🩺 Clinical Guideline & Model Interpretation
    </h1>
    <p style="margin: 0.4rem 0 0 0; color: #93c5fd; font-size: 0.95rem;">
        Clinical knowledge, model interpretation, and decision support guidance for healthcare professionals.
    </p>
</div>
""")

# ─── SECTION 2: WHAT IS OXYPREDICT ───────────────────────────────────────────
st_html("<div id='about-cdss'></div>")
st_html("<h3 class=\"section-title-custom\">📖 What is OxyPredict?</h3>")
st_html("""
<div class="cdss-card" style="border-left: 6px solid #2563eb;">
    <h4 style="margin: 0 0 0.8rem 0; color: #0a2e52; font-weight: 800; font-size: 1.1rem;">Clinical Decision Support System (CDSS) Overview</h4>
    <p style="margin: 0; color: #334155; font-size: 0.9rem; line-height: 1.7; text-align: justify;">
        <strong>OxyPredict</strong> merupakan Clinical Decision Support System berbasis Machine Learning yang membantu memprediksi kebutuhan terapi oksigen pada pasien anak dengan ISPA dan Pneumonia. Sistem ini dikembangkan untuk mendeteksi tanda desaturasi dan gangguan pernapasan secara dini. 
        <br/><br/>
        Model penduga dikembangkan menggunakan algoritma <strong>Random Forest</strong> yang dilatih pada cohort data klinis dari <strong>801 pasien anak</strong> dengan menguji <strong>44 fitur klinis</strong> (mencakup data demografis, riwayat kesehatan, tanda-tanda vital, serta hasil tes lab darah).
        <br/><br/>
        <strong style="color: #0a2e52;">PENTING:</strong> Sistem ini dikembangkan sebagai alat bantu penunjang keputusan klinis (opini medis kedua). Rekomendasi ataupun interpretasi model tidak pernah ditujukan untuk menggantikan diagnosis manual, dan <strong>keputusan akhir perawatan pasien sepenuhnya tetap berada pada dokter penanggung jawab</strong>.
    </p>
</div>
""")

# ─── SECTION 3: WHEN SHOULD THIS SYSTEM BE USED ──────────────────────────────
st_html("<div id='clinical-workflow'></div>")
st_html("<h3 class=\"section-title-custom\">⏱️ Clinical Workflow Integration</h3>")
st_html("<p style='color: #64748b; font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1rem;'>Alur pemanfaatan sistem CDSS OxyPredict dari penerimaan pasien hingga keputusan klinis:</p>")

st_html("""
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; gap: 0.8rem; margin-bottom: 1.5rem;">
    <div style="flex: 1; min-width: 110px; text-align: center; background: white; border: 1px solid #e2e8f0; padding: 0.6rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.01);">
        <span style="font-size: 1.2rem;">👤</span>
        <div style="font-size: 0.75rem; font-weight: 800; color: #0a2e52; margin-top: 0.3rem;">Patient Arrives</div>
    </div>
    <div style="font-size: 1.2rem; color: #cbd5e1;">➔</div>
    <div style="flex: 1; min-width: 110px; text-align: center; background: white; border: 1px solid #e2e8f0; padding: 0.6rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.01);">
        <span style="font-size: 1.2rem;">🩺</span>
        <div style="font-size: 0.75rem; font-weight: 800; color: #0a2e52; margin-top: 0.3rem;">Clinical Assessment</div>
    </div>
    <div style="font-size: 1.2rem; color: #cbd5e1;">➔</div>
    <div style="flex: 1; min-width: 110px; text-align: center; background: white; border: 1px solid #e2e8f0; padding: 0.6rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.01);">
        <span style="font-size: 1.2rem;">💻</span>
        <div style="font-size: 0.75rem; font-weight: 800; color: #0a2e52; margin-top: 0.3rem;">Input Information</div>
    </div>
    <div style="font-size: 1.2rem; color: #cbd5e1;">➔</div>
    <div style="flex: 1; min-width: 110px; text-align: center; background: white; border: 1px solid #2563eb; padding: 0.6rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(37,99,235,0.05); border-left: 4px solid #2563eb;">
        <span style="font-size: 1.2rem;">🔮</span>
        <div style="font-size: 0.75rem; font-weight: 800; color: #2563eb; margin-top: 0.3rem;">AI Prediction</div>
    </div>
    <div style="font-size: 1.2rem; color: #cbd5e1;">➔</div>
    <div style="flex: 1; min-width: 110px; text-align: center; background: white; border: 1px solid #e2e8f0; padding: 0.6rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.01);">
        <span style="font-size: 1.2rem;">🧠</span>
        <div style="font-size: 0.75rem; font-weight: 800; color: #0a2e52; margin-top: 0.3rem;">SHAP Explanation</div>
    </div>
    <div style="font-size: 1.2rem; color: #cbd5e1;">➔</div>
    <div style="flex: 1; min-width: 110px; text-align: center; background: white; border: 1px solid #e2e8f0; padding: 0.6rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.01);">
        <span style="font-size: 1.2rem;">📝</span>
        <div style="font-size: 0.75rem; font-weight: 800; color: #0a2e52; margin-top: 0.3rem;">Clinical Summary</div>
    </div>
    <div style="font-size: 1.2rem; color: #cbd5e1;">➔</div>
    <div style="flex: 1; min-width: 110px; text-align: center; background: white; border: 1px solid #e2e8f0; padding: 0.6rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.01);">
        <span style="font-size: 1.2rem;">🩹</span>
        <div style="font-size: 0.75rem; font-weight: 800; color: #0a2e52; margin-top: 0.3rem;">Recommendation</div>
    </div>
    <div style="font-size: 1.2rem; color: #cbd5e1;">➔</div>
    <div style="flex: 1; min-width: 110px; text-align: center; background: #eff6ff; border: 1px solid #bfdbfe; padding: 0.6rem; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.02); border-left: 4px solid #3b82f6;">
        <span style="font-size: 1.2rem;">👨‍⚕️</span>
        <div style="font-size: 0.75rem; font-weight: 800; color: #1e40af; margin-top: 0.3rem;">Physician Decision</div>
    </div>
</div>
""")

# ─── SECTION 4: HOW TO READ PREDICTION ───────────────────────────────────────
st_html("<div id='how-to-read-prediction'></div>")
st_html("<h3 class=\"section-title-custom\">🔮 How to Read Predictions</h3>")

pred_guide = get_prediction_guidelines()

col_pred1, col_pred2 = st.columns(2)
with col_pred1:
    st_html(f"""
    <div class="cdss-card" style="border-top: 4px solid #ef4444; height: 180px;">
        <div style="font-size: 1.1rem; font-weight: 800; color: #ef4444; margin-bottom: 0.5rem;">
            {pred_guide['yes_interpretation']['icon']} {pred_guide['yes_interpretation']['title']}
        </div>
        <p style="margin: 0; color: #475569; font-size: 0.85rem; line-height: 1.6;">
            {pred_guide['yes_interpretation']['desc']}
        </p>
    </div>
    """)
with col_pred2:
    st_html(f"""
    <div class="cdss-card" style="border-top: 4px solid #10b981; height: 180px;">
        <div style="font-size: 1.1rem; font-weight: 800; color: #10b981; margin-bottom: 0.5rem;">
            {pred_guide['no_interpretation']['icon']} {pred_guide['no_interpretation']['title']}
        </div>
        <p style="margin: 0; color: #475569; font-size: 0.85rem; line-height: 1.6;">
            {pred_guide['no_interpretation']['desc']}
        </p>
    </div>
    """)

st_html(f"""
<div class="cdss-card" style="border-left: 4px solid #3b82f6;">
    <div style="font-size: 1rem; font-weight: 800; color: #0a2e52; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
        <span>{pred_guide['probability_explanation']['icon']}</span> {pred_guide['probability_explanation']['title']}
    </div>
    <p style="margin: 0; color: #475569; font-size: 0.85rem; line-height: 1.6;">
        {pred_guide['probability_explanation']['desc']}
    </p>
</div>
""")

col_metric1, col_metric2 = st.columns(2)
with col_metric1:
    st_html("<div class=\"cdss-card\" style=\"height: 100%;\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Confidence Levels Lookup</h4>")
    for cl in pred_guide["confidence_levels"]:
        st_html(f"""
        <div style="display: flex; align-items: flex-start; gap: 0.6rem; font-size: 0.82rem; margin-bottom: 0.6rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.4rem;">
            <span style="font-size: 1rem; flex-shrink: 0;">{cl['icon']}</span>
            <div>
                <span style="font-weight: 700; color: #0f172a;">{cl['level']} ({cl['range']})</span>
                <span style="color: #64748b; display: block; margin-top: 0.1rem;">{cl['desc']}</span>
            </div>
        </div>
        """)
    st_html("</div>")

with col_metric2:
    st_html("<div class=\"cdss-card\" style=\"height: 100%;\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Risk Levels Lookup</h4>")
    for rl in pred_guide["risk_levels"]:
        st_html(f"""
        <div style="display: flex; align-items: flex-start; gap: 0.6rem; font-size: 0.82rem; margin-bottom: 0.6rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.4rem;">
            <span style="font-size: 1rem; flex-shrink: 0;">{rl['icon']}</span>
            <div>
                <span style="font-weight: 700; color: #0f172a;">{rl['level']} ({rl['range']})</span>
                <span style="color: #64748b; display: block; margin-top: 0.1rem;">{rl['desc']}</span>
            </div>
        </div>
        """)
    st_html("</div>")

# ─── SECTION 5: HOW TO READ SHAP ─────────────────────────────────────────────
st_html("<div id='how-to-read-shap'></div>")
st_html("<h3 class=\"section-title-custom\">🧠 How to Read SHAP Explanations</h3>")

shap_guide = get_shap_guidelines()

st_html(f"""
<div class="cdss-card">
    <p style="margin: 0 0 1.2rem 0; color: #334155; font-size: 0.88rem; line-height: 1.7; text-align: justify;">
        {shap_guide['definition']}
    </p>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.2rem;">
""")

for vr in shap_guide["visual_rules"]:
    st_html(f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid {vr['hex']}; border-radius: 8px; padding: 0.8rem 1rem;">
            <div style="font-weight: 800; color: #0f172a; font-size: 0.85rem; margin-bottom: 0.3rem;">{vr['color']}</div>
            <p style="margin: 0; color: #475569; font-size: 0.8rem; line-height: 1.5;">{vr['desc']}</p>
        </div>
    """)

st_html(f"""
    </div>
    
    <div style="
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #1e40af;
        line-height: 1.6;
    ">
        {shap_guide['example_case']}
    </div>
</div>
""")

# ─── SECTION 6: CLINICAL FEATURE REFERENCE (SEARCH & TABLE) ──────────────────
st_html("<div id='feature-reference'></div>")
st_html("<h3 class=\"section-title-custom\">📋 Clinical Feature Reference</h3>")
st_html("<p style='color: #64748b; font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1.2rem;'>Cari penjelasan klinis, rentang normal, dan signifikansi medis untuk seluruh 44 fitur masukan model:</p>")

# Load Feature Reference Data
df_ref = get_clinical_reference_df()

# Search Box
search_query = st.text_input(
    "🔍 Search Clinical Feature",
    "",
    placeholder="Ketik nama fitur klinis (misal: 'sao2', 'wheezing', 'age')..."
)

if search_query:
    df_filtered = df_ref[
        df_ref["Feature"].str.contains(search_query, case=False) |
        df_ref["Clinical Meaning"].str.contains(search_query, case=False) |
        df_ref["Clinical Importance"].str.contains(search_query, case=False)
    ]
else:
    df_filtered = df_ref

st.dataframe(df_filtered, use_container_width=True, hide_index=True)

# ─── SECTION 7: MODEL PIPELINE ───────────────────────────────────────────────
st_html("<div id='model-pipeline'></div>")
st_html("<h3 class=\"section-title-custom\">🔄 Model Processing Pipeline</h3>")
st_html("<p style='color: #64748b; font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1.2rem;'>Tahapan pengolahan data pasien hingga menghasilkan diagnosis dan rekomendasi klinis:</p>")

pipeline_steps = get_pipeline_steps()

for idx, s in enumerate(pipeline_steps):
    st_html(f"""
    <div class="cdss-card" style="border-left: 4px solid #2563eb; display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem; background-color: #ffffff;">
        <span style="font-size: 1rem; background: #eff6ff; color: #2563eb; min-width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">
            {s['step']}
        </span>
        <div style="flex-grow: 1;">
            <div style="font-weight: 800; color: #0a2e52; font-size: 0.9rem; display: flex; align-items: center; gap: 0.4rem;">
                <span>{s['icon']}</span> {s['name']}
            </div>
            <div style="font-size: 0.8rem; color: #475569; margin-top: 0.15rem; line-height: 1.4;">{s['desc']}</div>
        </div>
    </div>
    """)
    if idx < len(pipeline_steps) - 1:
        st_html("<div style='text-align: center; color: #cbd5e1; font-size: 1.1rem; margin-top: -0.3rem; margin-bottom: 0.3rem;'>↓</div>")

st.markdown("")

# ─── SECTION 8: MODEL PERFORMANCE ────────────────────────────────────────────
st_html("<div id='model-performance'></div>")
st_html("<h3 class=\"section-title-custom\">📈 Research Model Performance Metrics</h3>")

p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
p_col1.metric("Classification Accuracy", "89.44%")
p_col2.metric("F1 Macro Score", "86.20%")
p_col3.metric("ROC-AUC Score", "90.93%")
p_col4.metric("Dataset Size", "801 Patients")
p_col5.metric("Model Input Features", "44 Variables")

# ─── SECTION 9: MODEL LIMITATIONS ────────────────────────────────────────────
st_html("<h3 class=\"section-title-custom\">⚠️ Model Limitations</h3>")
st_html("""
<div class="cdss-card" style="
    background-color: #fffbeb;
    border: 1px solid #fef08a;
    border-left: 5px solid #eab308;
">
    <h4 style="margin: 0 0 0.6rem 0; color: #713f12; font-weight: 800; font-size: 1rem;">Clinical & Technical Limitations</h4>
    <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.85rem; color: #713f12; line-height: 1.6;">
        <li><strong>Dataset Homogeneity:</strong> Model ini dilatih menggunakan satu dataset retrospektif (801 sampel pasien anak). Hasil mungkin bervariasi jika diuji di fasilitas kesehatan dengan populasi demografis yang berbeda.</li>
        <li><strong>No External Validation:</strong> Model belum divalidasi secara klinis eksternal (multi-center validation). Hasil harus ditafsirkan sebagai perkiraan keputusan pendukung saja.</li>
        <li><strong>Decision Support Only:</strong> Prediksi model tidak ditujukan sebagai diagnosis primer yang berdiri sendiri, melainkan sebagai alat bantu skrining tambahan di UGD / ruang rawat anak.</li>
        <li><strong>Missing Values:</strong> Perkiraan terbaik dilakukan ketika seluruh fitur diinput secara lengkap dan akurat.</li>
    </ul>
</div>
""")

# ─── SECTION 10: FAQ ─────────────────────────────────────────────────────────
st_html("<div id='faq'></div>")
st_html("<h3 class=\"section-title-custom\">❓ Frequently Asked Questions (FAQ)</h3>")

faqs = get_faqs_list()

for faq in faqs:
    with st.expander(f"💬 {faq['question']}", expanded=False):
        st.markdown(f"<p style='font-size: 0.85rem; color: #334155; line-height: 1.6; text-align: justify;'>{faq['answer']}</p>", unsafe_allow_html=True)

# ─── SECTION 11: CLINICAL DISCLAIMER ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st_html("""
<div class="cdss-card" style="
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #64748b;
    padding: 1.5rem;
">
    <h4 style="margin: 0 0 0.6rem 0; color: #334155; font-weight: 800; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.5px;">🏥 CDSS Clinical Disclaimer</h4>
    <p style="margin: 0; color: #475569; font-size: 0.8rem; line-height: 1.6; text-align: justify;">
        This application is intended solely as a Clinical Decision Support System (CDSS). Predictions, explanations, and recommendations generated by the system are designed to assist healthcare professionals in clinical assessment. They must not replace physician judgment, institutional guidelines, or comprehensive patient evaluation. Final clinical decisions remain the responsibility of qualified medical personnel.
    </p>
</div>
""")

# ─── SECTION 12: QUICK NAVIGATION ───────────────────────────────────────────
st_html("<h3 class=\"section-title-custom\">🔗 Quick Navigation</h3>")

col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

with col_nav1:
    st.page_link("pages/2_Single_Prediction.py", label="🩺 Go to Single Prediction", use_container_width=True)
with col_nav2:
    st.page_link("pages/3_Batch_Prediction.py", label="📊 Go to Batch Prediction", use_container_width=True)
with col_nav3:
    st.page_link("pages/7_AI_Monitoring_Dashboard.py", label="📈 View AI Monitoring", use_container_width=True)
with col_nav4:
    st.page_link("pages/1_Dashboard.py", label="🫁 View Core Dashboard", use_container_width=True)
