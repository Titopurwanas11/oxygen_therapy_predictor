"""
Single Patient Prediction Page — Enter patient data manually and get a prediction.
Redesigned as a modern hospital Clinical Decision Support System (CDSS).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io
import base64
from utils.config import (
    FEATURE_GROUPS,
    NUMERICAL_FEATURES,
    NUMERICAL_RANGES,
    BINARY_OPTIONS,
    MULTI_CATEGORICAL_FEATURES,
    ALL_FEATURES,
    setup_page,
    render_page_header,
    render_section_divider,
    render_footer,
)
from utils.prediction import predict_single, get_confidence_level, get_risk_level
from utils.pdf_report import generate_pdf_report
from utils.shap_utils import (
    compute_shap_for_patient,
    generate_shap_clinical_interpretation,
    get_shap_feature_icon,
    generate_ai_clinical_summary,
)
from utils.session_analytics import (
    init_analytics_state,
    track_single_prediction,
    track_pdf_report_generated,
)
from utils.recommendation import generate_recommendation

# Try imports with graceful fallbacks
try:
    import plotly.graph_objects as go
    use_plotly = True
except ImportError:
    use_plotly = False

st.set_page_config(page_title="Prediksi Pasien — OxyPredict", page_icon="assets/favicon-64x64.png", layout="wide")
setup_page("Prediksi Pasien — OxyPredict")
init_analytics_state()

# Helper function to render HTML safely without markdown layout side-effects
def st_html(html_str):
    cleaned_lines = [line.lstrip() for line in html_str.splitlines()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)


def get_pdf_download_link(pdf_bytes: bytes, filename: str) -> str:
    encoded = base64.b64encode(pdf_bytes).decode()
    return (
        f'<a href="data:application/pdf;base64,{encoded}" download="{filename}" '
        f'style="display:inline-block;padding:0.8rem 1rem;border-radius:0.75rem;'
        f'background-color:#2563EB;color:#ffffff;font-weight:700;text-decoration:none;'
        f'box-shadow:0 10px 20px rgba(37,99,235,0.12);">Unduh Laporan Klinis (PDF)</a>'
    )

# ─── Custom Layout Styling ───────────────────────────────────────────────────
st_html("""
<style>
    /* Dynamic check list styling */
    .clinical-indicator-list {
        list-style-type: none;
        padding-left: 0;
        font-size: 0.88rem;
        line-height: 1.8;
    }
</style>
""")

# ─── Header ──────────────────────────────────────────────────────────────────
st_html(render_page_header(
    "",
    "Prediksi Pasien",
    "Sistem Pendukung Keputusan Klinis untuk Prediksi Kebutuhan Terapi Oksigen pada Pasien."
))

# ─── Session State Initialization ────────────────────────────────────────────
# Initialize session state for all inputs with their defaults if not present
for group_name, features in FEATURE_GROUPS.items():
    for feat in features:
        key = f"input_{feat}"
        if key not in st.session_state:
            if feat in NUMERICAL_FEATURES:
                min_val, max_val, default_val, step = NUMERICAL_RANGES[feat]
                st.session_state[key] = default_val
            elif feat in MULTI_CATEGORICAL_FEATURES:
                options = MULTI_CATEGORICAL_FEATURES[feat]
                st.session_state[key] = options[-1] # Default to 'Yes' or equivalent
            elif feat in BINARY_OPTIONS:
                options = BINARY_OPTIONS[feat]
                st.session_state[key] = options[0] # Default to 'No' or 'Female'

if "interacted_fields" not in st.session_state:
    st.session_state.interacted_fields = set()
if "single_predicted" not in st.session_state:
    st.session_state.single_predicted = False
if "single_prediction_result" not in st.session_state:
    st.session_state.single_prediction_result = None

def register_interaction(feat):
    st.session_state.interacted_fields.add(feat)
    st.session_state.single_predicted = False
    st.session_state.single_prediction_result = None

# ─── Placeholder Declarations ────────────────────────────────────────────────
# Placeholders are rendered first, and updated dynamically at the end of the script
header_placeholder = st.empty()
alerts_placeholder = st.empty()

# ─── Patient Input Form ──────────────────────────────────────────────────────
st_html("<h3 class=\"section-title-custom\">Formulir Data Klinis Pasien</h3>")

patient_data = {}
tab_names = list(FEATURE_GROUPS.keys())
tabs = st.tabs(tab_names)

for tab, (group_name, features) in zip(tabs, FEATURE_GROUPS.items()):
    with tab:
        st.markdown("<br>", unsafe_allow_html=True)
        n_cols = min(3, len(features))
        cols = st.columns(n_cols)

        for i, feat in enumerate(features):
            col = cols[i % n_cols]
            key = f"input_{feat}"

            with col:
                if feat in NUMERICAL_FEATURES:
                    min_val, max_val, _, step = NUMERICAL_RANGES[feat]
                    if isinstance(step, float):
                        patient_data[feat] = st.number_input(
                            feat,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            step=step,
                            format="%.1f",
                            key=key,
                            on_change=register_interaction,
                            args=(feat,)
                        )
                    else:
                        patient_data[feat] = st.number_input(
                            feat,
                            min_value=int(min_val),
                            max_value=int(max_val),
                            step=step,
                            key=key,
                            on_change=register_interaction,
                            args=(feat,)
                        )
                elif feat in MULTI_CATEGORICAL_FEATURES:
                    options = MULTI_CATEGORICAL_FEATURES[feat]
                    patient_data[feat] = st.selectbox(
                        feat,
                        options=options,
                        key=key,
                        on_change=register_interaction,
                        args=(feat,)
                    )
                elif feat in BINARY_OPTIONS:
                    options = BINARY_OPTIONS[feat]
                    patient_data[feat] = st.selectbox(
                        feat,
                        options=options,
                        key=key,
                        on_change=register_interaction,
                        args=(feat,)
                    )

st.markdown("")

# ─── Calculate Reactive Header & Tracker Values ──────────────────────────────
# Fetch session state values for reactive card calculations
age_months = st.session_state.get("input_Age (months)", 24)
gender = st.session_state.get("input_Gender", "Female")
weight = st.session_state.get("input_Weight (Kg)", 10.0)
height = st.session_state.get("input_Height (cm)", 75.0)
sao2 = st.session_state.get("input_Oxygen saturation (SaO2) at admission", 96.0)
temp = st.session_state.get("input_Axillary temperature (°C)", 37.0)
rr = st.session_state.get("input_Respiratory rate", 30)
wheezing = st.session_state.get("input_Wheezing", "No")
nasal_flaring = st.session_state.get("input_Nasal flaring", "No")

# Calculate BMI
if height > 0:
    bmi = weight / ((height / 100.0) ** 2)
else:
    bmi = 0.0

# Calculate BMI Category & Badge
if bmi < 18.5:
    bmi_category = "Underweight"
    bmi_color = "#dc2626" # Red
    bmi_bg = "rgba(220, 38, 38, 0.1)"
elif 18.5 <= bmi < 25:
    bmi_category = "Normal"
    bmi_color = "#16a34a" # Green
    bmi_bg = "rgba(22, 163, 74, 0.1)"
else:
    bmi_category = "Overweight"
    bmi_color = "#F59E0B" # Orange
    bmi_bg = "rgba(245, 158, 11, 0.08)"

# Calculate Clinical Alerts
is_tachypnea = False
if age_months < 2:
    is_tachypnea = rr >= 60
elif 2 <= age_months < 12:
    is_tachypnea = rr >= 50
elif 12 <= age_months < 60:
    is_tachypnea = rr >= 40
else:
    is_tachypnea = rr >= 30

alerts = []
if sao2 < 90:
    alerts.append({
        "type": "danger",
        "title": "Hipoksemia Berat",
        "desc": f"Saturasi oksigen (SaO2) sangat rendah pada <strong>{sao2:.1f}%</strong>. Pemberian oksigen segera sangat diperlukan."
    })
if is_tachypnea:
    alerts.append({
        "type": "warning",
        "title": "Takipnea Terdeteksi",
        "desc": f"Laju pernapasan meningkat pada <strong>{rr} bpm</strong> untuk pasien usia {age_months} bulan. Menunjukkan distres pernapasan."
    })
if temp > 38.5:
    alerts.append({
        "type": "warning",
        "title": "Demam Tinggi",
        "desc": f"Suhu aksila pasien meningkat pada <strong>{temp:.1f}°C</strong>, menunjukkan demam signifikan."
    })
if wheezing == "Yes":
    alerts.append({
        "type": "info",
        "title": "Wheezing Terdengar",
        "desc": "Suara napas tambahan terdeteksi. Mengindikasikan obstruksi saluran napas atau bronkospasme."
    })
if nasal_flaring == "Yes":
    alerts.append({
        "type": "warning",
        "title": "Tanda Distres Pernapasan",
        "desc": "Napas cuping hidung terdeteksi, menunjukkan peningkatan usaha bernapas dan penggunaan otot bantu pernapasan."
    })

# ─── Render Reactive Header, Tracker & Alerts placeholders ──────────────────
with header_placeholder.container():
    st_html(f"""
    <div class="cdss-card">
        <h4 style="margin: 0 0 1rem 0; color: #0F172A; font-weight: 700; font-size: 18px;">
            Ikhtisar Pasien
        </h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 1rem;">
            <div style="background-color: #f8fafc; border-radius: 10px; padding: 0.6rem 0.8rem; border: 1px solid #f1f5f9;">
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Usia</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-top: 0.15rem;">{age_months} bulan</div>
            </div>
            <div style="background-color: #f8fafc; border-radius: 10px; padding: 0.6rem 0.8rem; border: 1px solid #f1f5f9;">
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Jenis Kelamin</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-top: 0.15rem;">{gender}</div>
            </div>
            <div style="background-color: #f8fafc; border-radius: 10px; padding: 0.6rem 0.8rem; border: 1px solid #f1f5f9;">
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Berat Badan</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-top: 0.15rem;">{weight:.1f} Kg</div>
            </div>
            <div style="background-color: #f8fafc; border-radius: 10px; padding: 0.6rem 0.8rem; border: 1px solid #f1f5f9;">
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Tinggi Badan</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-top: 0.15rem;">{height:.1f} cm</div>
            </div>
            <div style="background-color: #f8fafc; border-radius: 10px; padding: 0.6rem 0.8rem; border: 1px solid #f1f5f9; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">IMT</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-top: 0.15rem;">{bmi:.2f}</div>
                </div>
                <div style="
                    display: inline-block;
                    align-self: flex-start;
                    background-color: {bmi_bg};
                    color: {bmi_color};
                    border: 1px solid {bmi_color}30;
                    font-size: 0.65rem;
                    font-weight: 800;
                    padding: 0.15rem 0.4rem;
                    border-radius: 6px;
                    margin-top: 0.3rem;
                    text-transform: uppercase;
                ">
                    {bmi_category}
                </div>
            </div>
        </div>
    </div>
    """)

with alerts_placeholder.container():
    if alerts:
        st_html("<div style='margin-bottom: 1.5rem;'>")
        st_html("<h4 style='color: #0F172A; font-size: 18px; font-weight: 700; margin-bottom: 0.8rem;'>Peringatan Klinis Real-time</h4>")
        for alert in alerts:
            if alert["type"] == "danger":
                bg = "#FEF2F2"
                border = "#EF4444"
                color = "#B91C1C"
            elif alert["type"] == "warning":
                bg = "#FFFBEB"
                border = "#F59E0B"
                color = "#B45309"
            else: # info
                bg = "#EFF6FF"
                border = "#3282B8"
                color = "#1D4ED8"
                
            st_html(f"""
            <div style="
                background-color: {bg};
                border-left: 4px solid {border};
                border-radius: 8px;
                padding: 0.8rem 1rem;
                margin-bottom: 0.5rem;
                box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            ">
                <span style="font-size: 0.85rem; font-weight: 700; color: {border}; display: block; margin-bottom: 0.2rem;">{alert['title']}</span>
                <span style="font-size: 0.8rem; color: {color}; line-height: 1.5;">{alert['desc']}</span>
            </div>
            """)
        st_html("</div>")

# ─── Predict Button ─────────────────────────────────────────────────────────
col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
with col_btn_c:
    predict_clicked = st.button(
        "🔮  Jalankan Prediksi Klinis",
        use_container_width=True,
        type="primary"
    )

# ─── Prediction Result Rendering ─────────────────────────────────────────────
if predict_clicked or st.session_state.get("single_predicted", False):
    if predict_clicked or "single_prediction_result" not in st.session_state:
        # 1. Validate inputs
        from utils.config import ALL_FEATURES, NUMERICAL_RANGES, CATEGORICAL_OPTIONS, show_warning_card, show_error_card
        
        is_valid, val_err = True, ""
        for feat in ALL_FEATURES:
            if feat not in patient_data:
                is_valid, val_err = False, f"Fitur '{feat}' belum terisi."
                break
            val = patient_data[feat]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                is_valid, val_err = False, f"Fitur '{feat}' tidak boleh kosong (NaN)."
                break
            if feat in NUMERICAL_RANGES:
                min_val, max_val, _, _ = NUMERICAL_RANGES[feat]
                if val < min_val or val > max_val:
                    is_valid, val_err = False, f"Nilai '{feat}' harus berada pada rentang {min_val} hingga {max_val}."
                    break
            elif feat in CATEGORICAL_OPTIONS:
                opts = CATEGORICAL_OPTIONS[feat]
                if val not in opts:
                    is_valid, val_err = False, f"Nilai '{feat}' tidak valid. Pilihan yang diperbolehkan: {', '.join(opts)}."
                    break
                    
        if not is_valid:
            show_warning_card("Data Input Pasien Tidak Valid", val_err)
            st.stop()

        with st.spinner("Memproses Prediksi Klinis..."):
            try:
                from utils.prediction import ModelLoadError, predict_single
                label, prob_yes = predict_single(patient_data)
                track_single_prediction()
                prob_yes_pct = prob_yes * 100
                
                # Confidence Interpretation calculation
                if label == "Yes":
                    prob_pct = prob_yes * 100
                else:
                    prob_pct = (1 - prob_yes) * 100
                    
                if prob_pct > 85:
                    confidence_level = "High Confidence"
                    confidence_label = "High"
                elif 70 <= prob_pct <= 85:
                    confidence_level = "Moderate Confidence"
                    confidence_label = "Moderate"
                else:
                    confidence_level = "Low Confidence"
                    confidence_label = "Low"

                # Clinical Risk Badge calculation
                if prob_yes_pct >= 85:
                    risk_level = "Risiko Tinggi"
                    risk_badge = "🔴 Risiko Tinggi"
                    risk_color = "#EF4444"
                    risk_color_rgb = (239, 68, 68)
                    risk_bg = "#FEF2F2"
                    risk_border = "#FCA5A5"
                elif 60 <= prob_yes_pct < 85:
                    risk_level = "Risiko Sedang"
                    risk_badge = "🟠 Risiko Sedang"
                    risk_color = "#F59E0B"
                    risk_color_rgb = (245, 158, 11)
                    risk_bg = "#FFFBEB"
                    risk_border = "#FCD34D"
                else:
                    risk_level = "Risiko Rendah"
                    risk_badge = "🟢 Risiko Rendah"
                    risk_color = "#22C55E"
                    risk_color_rgb = (34, 197, 94)
                    risk_bg = "#ECFDF5"
                    risk_border = "#A7F3D0"

                # Save to prediction history file
                try:
                    from utils.monitoring import record_prediction
                    conf_pct = prob_yes * 100 if label == "Yes" else (1.0 - prob_yes) * 100
                    record_prediction(
                        patient_data=patient_data,
                        prediction=label,
                        confidence=conf_pct,
                        risk_level=risk_level,
                        type="Single"
                    )
                except Exception:
                    pass

                # Compute SHAP and AI Clinical Summary narrative upfront
                shap_ok = False
                shap_values_list = []
                top_10 = []
                positive_factors = []
                negative_factors = []
                narrative_text = "Clinical summary not available due to SHAP computation error."
                
                try:
                    shap_result = compute_shap_for_patient(patient_data)
                    shap_values_list = shap_result["shap_values"]
                    shap_base = shap_result["base_value"]
                    shap_prob = shap_result["predicted_prob"]
                    shap_ok = True
                    
                    top_10 = shap_values_list[:10]
                    positive_factors = [s for s in shap_values_list if s["shap_value"] > 0.005]
                    negative_factors = [s for s in shap_values_list if s["shap_value"] < -0.005]

                    pred_val = 1 if label == "Yes" else 0
                    narrative_text = generate_ai_clinical_summary(
                        prediction=pred_val,
                        probability=shap_prob,
                        shap_values=shap_values_list,
                        feature_names=ALL_FEATURES,
                        feature_values=patient_data
                    )
                except Exception as shap_err:
                    from utils.config import logger
                    logger.error("Could not compute SHAP values: %s", str(shap_err), exc_info=True)

                # Generate Clinical Recommendation
                rec_dict = generate_recommendation(
                    prediction=label,
                    probability=prob_yes,
                    risk_level=risk_level,
                    confidence_level=confidence_label,
                    patient_data=patient_data,
                    top_shap_features=top_10
                )

                # Generate professional PDF report using ReportLab
                pdf_report_bytes = None
                try:
                    pdf_report_bytes = generate_pdf_report(
                        patient_data=patient_data,
                        prediction=label,
                        probability=prob_yes,
                        confidence=confidence_label,
                        risk_level=risk_level,
                        clinical_summary=narrative_text,
                        top_shap_features=top_10,
                        shap_values=shap_values_list,
                        feature_names=ALL_FEATURES,
                        recommendation=rec_dict
                    )
                except Exception as pdf_err:
                    from utils.config import logger
                    logger.error("Single PDF report generation failed: %s", str(pdf_err), exc_info=True)
                    pdf_report_bytes = None

                # Store in session state
                st.session_state.single_prediction_result = {
                    "label": label,
                    "prob_yes": prob_yes,
                    "prob_yes_pct": prob_yes_pct,
                    "prob_pct": prob_pct,
                    "confidence_level": confidence_level,
                    "confidence_label": confidence_label,
                    "risk_level": risk_level,
                    "risk_badge": risk_badge,
                    "risk_color": risk_color,
                    "risk_color_rgb": risk_color_rgb,
                    "risk_bg": risk_bg,
                    "risk_border": risk_border,
                    "shap_ok": shap_ok,
                    "shap_values_list": shap_values_list,
                    "top_10": top_10,
                    "positive_factors": positive_factors,
                    "negative_factors": negative_factors,
                    "narrative_text": narrative_text,
                    "rec_dict": rec_dict,
                    "pdf_report_bytes": pdf_report_bytes,
                }
                st.session_state.single_predicted = True

            except ModelLoadError:
                from utils.config import show_error_card
                show_error_card(
                    "Model Prediksi Tidak Dapat Dimuat",
                    "Kemungkinan penyebab:<br>• Berkas model tidak ditemukan.<br>• Model tidak kompatibel.<br>• Versi model tidak sesuai.<br><br>Silakan hubungi administrator sistem."
                )
                st.stop()
            except Exception as e:
                from utils.config import logger, show_error_card
                logger.error("Single prediction failed: %s", str(e), exc_info=True)
                show_error_card(
                    "Prediksi Gagal Diproses",
                    "Sistem tidak dapat melakukan prediksi. Periksa kembali data pasien kemudian coba kembali."
                )
                st.stop()

        try:
            res = st.session_state.single_prediction_result
            label = res["label"]
            prob_yes = res["prob_yes"]
            prob_yes_pct = res["prob_yes_pct"]
            prob_pct = res["prob_pct"]
            confidence_level = res["confidence_level"]
            confidence_label = res["confidence_label"]
            risk_level = res["risk_level"]
            risk_badge = res["risk_badge"]
            risk_color = res["risk_color"]
            risk_color_rgb = res["risk_color_rgb"]
            risk_bg = res["risk_bg"]
            risk_border = res["risk_border"]
            shap_ok = res["shap_ok"]
            shap_values_list = res["shap_values_list"]
            top_10 = res["top_10"]
            positive_factors = res["positive_factors"]
            negative_factors = res["negative_factors"]
            narrative_text = res["narrative_text"]
            rec_dict = res["rec_dict"]
            pdf_report_bytes = res["pdf_report_bytes"]

            st.markdown("<br>", unsafe_allow_html=True)
            st_html("<h3 class=\"section-title-custom\">Hasil Prediksi & Analisis Klinis</h3>")
            
            col_res_left, col_res_right = st.columns([3, 2])

            with col_res_left:
                # Diagnostic Summary Card
                st_html(f"""
                <div class="cdss-card">
                    <div style="font-size: 13px; color: #64748B; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">ANALISIS DIAGNOSTIK KLINIS</div>
                    <div style="font-size: 18px; font-weight: 700; color: #0F172A; margin: 0.5rem 0 0.8rem 0;">Ringkasan Prediksi</div>
                    <div style="border-top: 1px solid #E8EEF5; padding-top: 0.8rem; display: flex; flex-direction: column; gap: 0.6rem; font-size: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #64748B;">Hasil Prediksi:</span>
                            <span class="cdss-badge cdss-badge-{'danger' if label == 'Yes' else 'success'}">
                                {'Butuh Terapi Oksigen' if label == 'Yes' else 'Tidak Butuh Terapi Oksigen'}
                            </span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #64748B;">Probabilitas:</span>
                            <span style="font-weight: 600; color: #0F172A;">{prob_yes_pct:.1f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #64748B;">Tingkat Keyakinan:</span>
                            <span class="cdss-badge cdss-badge-{'danger' if confidence_label == 'Low' else ('warning' if confidence_label == 'Moderate' else 'success')}">
                                {confidence_level} ({prob_pct:.1f}%)
                            </span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #64748B;">Tingkat Risiko Klinis:</span>
                            <span class="cdss-badge cdss-badge-{'danger' if risk_level == 'Risiko Tinggi' else ('warning' if risk_level == 'Risiko Sedang' else 'success')}">
                                {risk_level}
                            </span>
                        </div>
                    </div>
                </div>
                """)

                # Key Clinical Factors Checklist Card
                indicator_html = ""
                if sao2 < 92:
                    indicator_html += f"<li style='color: #EF4444; margin-bottom: 0.4rem;'><strong>Saturasi Oksigen Rendah:</strong> SaO2 di bawah 92% ({sao2:.1f}%)</li>"
                else:
                    indicator_html += f"<li style='color: #1E293B; margin-bottom: 0.4rem;'>✓ Saturasi oksigen normal ({sao2:.1f}%)</li>"
                
                if is_tachypnea:
                    indicator_html += f"<li style='color: #F59E0B; margin-bottom: 0.4rem;'><strong>Laju Pernapasan Meningkat:</strong> Takipnea terdeteksi ({rr} bpm)</li>"
                else:
                    indicator_html += f"<li style='color: #1E293B; margin-bottom: 0.4rem;'>✓ Laju pernapasan normal ({rr} bpm)</li>"
                    
                if temp > 37.5:
                    indicator_html += f"<li style='color: #F59E0B; margin-bottom: 0.4rem;'><strong>Demam:</strong> Suhu {temp:.1f}°C</li>"
                else:
                    indicator_html += f"<li style='color: #1E293B; margin-bottom: 0.4rem;'>✓ Suhu tubuh normal ({temp:.1f}°C)</li>"
                    
                if wheezing == "Yes":
                    indicator_html += "<li style='color: #3282B8; margin-bottom: 0.4rem;'><strong>Wheezing Terdengar:</strong> Suara napas bronkial tambahan</li>"
                else:
                    indicator_html += "<li style='color: #1E293B; margin-bottom: 0.4rem;'>✓ Tidak ada wheezing</li>"
                    
                if nasal_flaring == "Yes":
                    indicator_html += "<li style='color: #EF4444; margin-bottom: 0.4rem;'><strong>Napas Cuping Hidung:</strong> Tanda distres pernapasan</li>"
                else:
                    indicator_html += "<li style='color: #1E293B; margin-bottom: 0.4rem;'>✓ Tidak ada napas cuping hidung</li>"

                st_html(f"""
                <div class="cdss-card">
                    <div style="font-size: 13px; color: #64748B; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">TEMUAN KLINIS PASIEN</div>
                    <div style="font-size: 18px; font-weight: 700; color: #0F172A; margin: 0.4rem 0 0.8rem 0;">Indikator Klinis Utama</div>
                    <ul class="clinical-indicator-list" style="margin: 0; padding-left: 1.2rem; color: #1E293B; font-size: 15px; line-height: 1.6;">
                        {indicator_html}
                    </ul>
                </div>
                """)

                # Download Link for PDF Report (uses HTML link to avoid page rerun refresh)
                if pdf_report_bytes:
                    report_filename = f"OxyPredict_Laporan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf_link = get_pdf_download_link(pdf_report_bytes, report_filename)
                    st.markdown(pdf_link, unsafe_allow_html=True)
                else:
                    st.warning("Laporan PDF tidak dapat dibuat saat ini.")
                    
            with col_res_right:
                # Gauge Chart rendering
                st_html("""
                <div class="cdss-card" style="
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 250px;
                ">
                    <div style="font-size: 13px; color: #64748B; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; width: 100%; text-align: left; margin-bottom: 1rem;">PROBABILITAS KEBUTUHAN OKSIGEN</div>
                """)

                if use_plotly:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_yes_pct,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#0F4C75"},
                            'bar': {'color': risk_color},
                            'bgcolor': "white",
                            'borderwidth': 1.5,
                            'bordercolor': "#D6E4F0",
                            'steps': [
                                {'range': [0, 60], 'color': 'rgba(34, 197, 94, 0.08)'},
                                {'range': [60, 85], 'color': 'rgba(245, 158, 11, 0.08)'},
                                {'range': [85, 100], 'color': 'rgba(239, 68, 68, 0.08)'}
                            ],
                            'threshold': {
                                'line': {'color': risk_color, 'width': 3},
                                'thickness': 0.75,
                                'value': prob_yes_pct
                            }
                        }
                    ))
                    fig.update_layout(
                        height=160,
                        margin=dict(t=10, b=10, l=15, r=15),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#0F172A", 'family': "Inter"}
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    # SVG Gauge Fallback
                    st_html(f"""
                        <svg viewBox="0 0 120 70" width="100%" height="auto" style="max-width: 170px; display: block; margin: 0 auto;">
                            <!-- Background Arc -->
                            <path d="M 20 60 A 40 40 0 0 1 100 60" fill="none" stroke="#e2e8f0" stroke-width="7" stroke-linecap="round" />
                            
                            <!-- Colored Progress Arc -->
                            <path d="M 20 60 A 40 40 0 0 1 100 60" fill="none" stroke="{risk_color}" stroke-width="7" stroke-linecap="round" 
                                  stroke-dasharray="125.66" stroke-dashoffset="{125.66 * (1 - prob_yes_pct/100)}" />
                                  
                            <!-- Probability Text -->
                            <text x="60" y="52" fill="#0F172A" font-size="14" text-anchor="middle" font-family="sans-serif" font-weight="800">{prob_yes_pct:.1f}%</text>
                            <text x="60" y="64" fill="#64748b" font-size="7" text-anchor="middle" font-family="sans-serif" font-weight="600">{risk_level.upper()}</text>
                        </svg>
                    """)
                st_html("</div>")

                # Recommendation panel
                if label == "Yes":
                    rec_bg = "#FEF2F2"
                    rec_border = "#EF4444"
                    rec_color = "#B91C1C"
                    rec_text = "Pasien perlu menjalani penilaian pernapasan lebih lanjut dan pemantauan saturasi oksigen."
                else:
                    rec_bg = "#ECFDF5"
                    rec_border = "#22C55E"
                    rec_color = "#15803D"
                    rec_text = "Pasien tidak memerlukan terapi oksigen saat ini. Lanjutkan pemantauan rutin dan penilaian ulang jika kondisi berubah."

                st_html(f"""
                <div class="cdss-card" style="border-left: 5px solid {rec_border};">
                    <h5 style="margin: 0 0 0.4rem 0; color: #0F172A; font-weight: 700; font-size: 16px;">Pertimbangan Klinis Awal</h5>
                    <p style="margin: 0; color: {rec_color}; font-size: 15px; line-height: 1.6; font-weight: 500;">
                        {rec_text}
                    </p>
                </div>
                """)

            st.markdown("---")

            # ── Section 1: Header ─────────────────────────────────────────
            st_html("""
            <div class="cdss-card" style="border-left: 6px solid #0F4C75; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.6rem;">
                    <h2 style="margin: 0; color: #0F172A; font-size: 22px; font-weight: 700;">Penjelasan Prediksi Berbasis SHAP</h2>
                </div>
                <p style="margin: 0; color: #64748B; font-size: 15px; line-height: 1.6;">
                    Penjelasan ini dihasilkan menggunakan <strong style="color: #0F172A;">SHAP</strong>
                    (SHapley Additive exPlanations) untuk meningkatkan transparansi hasil prediksi model.
                </p>
            </div>
            """)

            if shap_ok:
                top_10 = shap_values_list[:10]
                positive_factors = [s for s in shap_values_list if s["shap_value"] > 0.005]
                negative_factors = [s for s in shap_values_list if s["shap_value"] < -0.005]

                # ── Section 3: Top 10 Features Table ──────────────────────
                st_html("<h3 class=\"section-title-custom\">10 Fitur dengan Pengaruh Terbesar</h3>")
                st_html("<p style='color: #64748b; font-size: 0.82rem; margin-top: -0.5rem; margin-bottom: 1rem;'>Fitur diurutkan berdasarkan nilai SHAP absolut — menunjukkan pengaruh terkuat terhadap prediksi ini:</p>")

                table_rows = ""
                for i, sv in enumerate(top_10):
                    shap_val = sv["shap_value"]
                    if shap_val > 0:
                        impact_label = "↑ Meningkatkan Prediksi"
                        impact_color = "#EF4444"
                        impact_bg = "#FEF2F2"
                        sign = "+"
                    else:
                        impact_label = "↓ Menurunkan Prediksi"
                        impact_color = "#3282B8"
                        impact_bg = "#EFF6FF"
                        sign = ""
                    row_bg = "#FFFFFF" if i % 2 == 0 else "#F8FAFC"
                    table_rows += f"""
                    <tr style="background-color: {row_bg};">
                        <td style="padding: 0.7rem 1rem; font-weight: 600; color: #1E293B; font-size: 0.82rem; border-bottom: 1px solid #EAF2F8;">
                            {sv['feature']}
                        </td>
                        <td style="padding: 0.7rem 1rem; color: #1E293B; font-size: 0.82rem; border-bottom: 1px solid #EAF2F8; text-align: center;">
                            <strong>{sv['patient_value']}</strong>
                        </td>
                        <td style="padding: 0.7rem 1rem; font-family: 'Courier New', monospace; font-weight: 700; color: {impact_color}; font-size: 0.82rem; border-bottom: 1px solid #EAF2F8; text-align: center;">
                            {sign}{shap_val:.4f}
                        </td>
                        <td style="padding: 0.7rem 1rem; border-bottom: 1px solid #f1f5f9; text-align: center;">
                            <span style="background-color: {impact_bg}; color: {impact_color}; font-size: 0.7rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 6px;">
                                {impact_label}
                            </span>
                        </td>
                    </tr>
                    """

                st_html(f"""
                <div style="
                    background: #ffffff;
                    border: 1px solid #D6E4F0;
                    border-radius: 16px;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(15,76,117,0.04);
                    margin-bottom: 1.5rem;
                ">
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="background-color: #0F4C75;">
                                <th style="padding: 0.85rem 1.2rem; text-align: left; color: #FFFFFF; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; border-bottom: 2px solid #3282B8;">Fitur</th>
                                <th style="padding: 0.85rem 1.2rem; text-align: center; color: #FFFFFF; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; border-bottom: 2px solid #3282B8;">Nilai Pasien</th>
                                <th style="padding: 0.85rem 1.2rem; text-align: center; color: #FFFFFF; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; border-bottom: 2px solid #3282B8;">Nilai SHAP</th>
                                <th style="padding: 0.85rem 1.2rem; text-align: center; color: #FFFFFF; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; border-bottom: 2px solid #3282B8;">Dampak</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
                """)

                # ── Section 4: Positive / Negative Split ──────────────────
                st_html("<h3 class=\"section-title-custom\">Arah Dampak Fitur</h3>")

                col_pos, col_neg = st.columns(2)

                with col_pos:
                    pos_items = ""
                    for pf in positive_factors[:6]:
                        pos_items += f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #FEF2F2;">
                            <span style="font-size: 0.8rem; color: #EF4444; font-weight: 500;">{pf['feature']}</span>
                            <span style="font-family: monospace; font-weight: 700; color: #EF4444; font-size: 0.8rem;">+{pf['shap_value']:.4f}</span>
                        </div>
                        """
                    if not positive_factors:
                        pos_items = "<p style='color: #94a3b8; font-size: 0.8rem; font-style: italic;'>Tidak ada faktor positif signifikan</p>"
                    st_html(f"""
                    <div class="cdss-card" style="border-top: 4px solid #EF4444; height: 100%;">
                        <div style="font-size: 16px; font-weight: 700; color: #EF4444; margin-bottom: 0.8rem;">
                            Fitur yang Meningkatkan Prediksi
                        </div>
                        {pos_items}
                    </div>
                    """)

                with col_neg:
                    neg_items = ""
                    for nf in negative_factors[:6]:
                        neg_items += f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #EFF6FF;">
                            <span style="font-size: 0.8rem; color: #3282B8; font-weight: 500;">{nf['feature']}</span>
                            <span style="font-family: monospace; font-weight: 700; color: #3282B8; font-size: 0.8rem;">{nf['shap_value']:.4f}</span>
                        </div>
                        """
                    if not negative_factors:
                        neg_items = "<p style='color: #94a3b8; font-size: 0.8rem; font-style: italic;'>Tidak ada faktor negatif signifikan</p>"
                    st_html(f"""
                    <div class="cdss-card" style="border-top: 4px solid #3282B8; height: 100%;">
                        <div style="font-size: 16px; font-weight: 700; color: #3282B8; margin-bottom: 0.8rem;">
                            Fitur yang Menurunkan Prediksi
                        </div>
                        {neg_items}
                    </div>
                    """)

                # ── Section 5: Interactive Plotly SHAP Bar Chart ──────────
                st_html("<h3 class=\"section-title-custom\">Distribusi Nilai SHAP (Interaktif)</h3>")

                if use_plotly:
                    chart_data = list(reversed(top_10))
                    chart_features = [d["feature"] for d in chart_data]
                    chart_shap = [d["shap_value"] for d in chart_data]
                    chart_colors = ["#EF4444" if v > 0 else "#3282B8" for v in chart_shap]

                    fig_shap = go.Figure(go.Bar(
                        x=chart_shap,
                        y=chart_features,
                        orientation='h',
                        marker=dict(
                            color=chart_colors,
                            line=dict(width=0),
                        ),
                        hovertemplate="<b>%{y}</b><br>Nilai SHAP: %{x:.4f}<extra></extra>"
                    ))
                    fig_shap.update_layout(
                        height=380,
                        margin=dict(t=10, b=30, l=10, r=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="#1E293B", size=12),
                        xaxis=dict(
                            title="Nilai SHAP (dampak pada prediksi)",
                            titlefont=dict(size=11, color="#64748b"),
                            gridcolor="#EAF2F8",
                            zerolinecolor="#D6E4F0",
                            zerolinewidth=2,
                        ),
                        yaxis=dict(
                            tickfont=dict(size=11),
                            automargin=True,
                        ),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True, config={
                        'displayModeBar': True,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                        'displaylogo': False,
                    })
                else:
                    st.info("Instal Plotly (`pip install plotly`) untuk grafik SHAP interaktif.")

                # ── Section 5.5: AI Clinical Summary (Explainable AI) ─────
                st_html(f"""
                <div class="cdss-card" style="border-left: 6px solid #3282B8; margin-top: 1.5rem; margin-bottom: 2rem;">
                    <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1rem;">
                        <div>
                            <h4 style="margin: 0; color: #0F172A; font-weight: 700; font-size: 18px; letter-spacing: -0.3px; line-height: 1.2;">
                                Ringkasan Klinis AI
                            </h4>
                            <div style="font-size: 13px; color: #64748B; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.1rem;">
                                Dihasilkan dari Explainable AI (SHAP)
                            </div>
                        </div>
                    </div>

                    <p style="
                        margin: 0; 
                        color: #334155; 
                        font-size: 16px; 
                        line-height: 1.75; 
                        font-weight: 400;
                        text-align: justify;
                    ">
                        {narrative_text}
                    </p>
                </div>
                """)

                # ── Section 5.8: Clinical Recommendation Card ──────────────
                priority_val = rec_dict["priority"]
                if priority_val in ["Darurat", "Emergency", "Tinggi"]:
                    border_col = "#B45309"
                    badge_class = "cdss-badge-danger"
                    priority_text = "Tinggi"
                elif priority_val == "Sedang":
                    border_col = "#B45309"
                    badge_class = "cdss-badge-warning"
                    priority_text = "Sedang"
                else:
                    border_col = "#15803D"
                    badge_class = "cdss-badge-success"
                    priority_text = "Rendah"

                actions_html = "".join([f"<li style='margin-bottom: 0.3rem;'>{act}</li>" for act in rec_dict["clinical_action"]])
                monitoring_html = "".join([f"<li style='margin-bottom: 0.3rem;'>{mon}</li>" for mon in rec_dict["monitoring"]])
                notes_html = " ".join(rec_dict["notes"])

                st_html(f"""
                <div class="cdss-card" style="border-left: 6px solid {border_col}; margin-top: 1.5rem; margin-bottom: 2rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.2rem;">
                        <div style="display: flex; align-items: center; gap: 0.6rem;">
                            <div>
                                <h4 style="margin: 0; color: #0F172A; font-weight: 700; font-size: 18px; letter-spacing: -0.3px; line-height: 1.2;">
                                    Rekomendasi Klinis
                                </h4>
                                <div style="font-size: 13px; color: #64748B; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.1rem;">
                                    Rekomendasi CDSS Berbasis Aturan Klinis
                                </div>
                            </div>
                        </div>
                        <span class="cdss-badge {badge_class}">
                            Prioritas {priority_text}
                        </span>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.2rem;">
                        <div>
                            <h5 style="margin: 0 0 0.5rem 0; color: #0F172A; font-size: 16px; font-weight: 600; border-bottom: 1px solid #D6E4F0; padding-bottom: 0.3rem;">Tindakan Klinis</h5>
                            <ul style="margin: 0; padding-left: 1.2rem; color: #1E293B; font-size: 16px; line-height: 1.6;">
                                {actions_html}
                            </ul>
                        </div>
                        <div>
                            <h5 style="margin: 0 0 0.5rem 0; color: #0F172A; font-size: 16px; font-weight: 600; border-bottom: 1px solid #D6E4F0; padding-bottom: 0.3rem;">Panduan Pemantauan</h5>
                            <ul style="margin: 0; padding-left: 1.2rem; color: #1E293B; font-size: 16px; line-height: 1.6;">
                                {monitoring_html}
                            </ul>
                        </div>
                    </div>

                    <div style="
                        background-color: #FFFBEB;
                        border: 1px solid #FCD34D;
                        border-left: 4px solid #F59E0B;
                        border-radius: 8px;
                        padding: 0.8rem 1rem;
                        font-size: 13px;
                        color: #B45309;
                        line-height: 1.5;
                    ">
                        <strong>Perhatian & Pedoman:</strong><br/>
                        {notes_html}
                    </div>
                </div>
                """)

                # ── Section 6: Clinical Interpretation ────────────────────
                interpretation = generate_shap_clinical_interpretation(shap_result, label)
                st_html(f"""
                <div class="cdss-card" style="
                    background-color: #EFF6FF;
                    border: 1px solid #BFDBFE;
                    border-left: 5px solid #3282B8;
                    margin-bottom: 2rem;
                ">
                    <h4 style="margin: 0 0 0.6rem 0; color: #1D4ED8; font-weight: 700; font-size: 18px; display: flex; align-items: center; gap: 0.4rem;">
                        Interpretasi Klinis
                    </h4>
                    <p style="margin: 0; color: #1D4ED8; font-size: 16px; line-height: 1.7;">
                        {interpretation}
                    </p>
                </div>
                """)

                # ── Section 7: Top 5 Feature Importance Cards ─────────────
                st_html("<h3 class=\"section-title-custom\">5 Fitur Paling Berpengaruh</h3>")

                top_5_shap = shap_values_list[:5]
                cols_top5 = st.columns(5)
                for idx, sv in enumerate(top_5_shap):
                    shap_val = sv["shap_value"]
                    if shap_val > 0:
                        card_bg = "#fdf2f2"
                        card_border = "#fecaca"
                        val_color = "#DC2626"
                        sign = "+"
                    else:
                        card_bg = "#eff6ff"
                        card_border = "#bfdbfe"
                        val_color = "#2563EB"
                        sign = ""
                    with cols_top5[idx]:
                        st_html(f"""
                        <div style="
                            background-color: {card_bg};
                            border: 1px solid {card_border};
                            border-radius: 16px;
                            padding: 1rem;
                            min-height: 150px;
                            display: flex;
                            flex-direction: column;
                            justify-content: space-between;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
                            transition: all 0.25s ease;
                        ">
                            <div>
                                <div style="font-size: 13px; font-weight: 700; color: #0F172A; line-height: 1.3; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">
                                    {sv['feature']}
                                </div>
                            </div>
                            <div style="margin-top: 0.6rem;">
                                <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Contribution</div>
                                <div style="font-size: 1.1rem; font-weight: 800; color: {val_color}; margin-top: 0.1rem;">
                                    {sign}{shap_val:.4f}
                                </div>
                            </div>
                        </div>
                        """)

                st.markdown("")

                # ── Section 8: Explainability Summary ─────────────────────
                st_html("<h3 class=\"section-title-custom\">Ringkasan Eksplanabilitas Model</h3>")

                n_positive = len(positive_factors)
                n_negative = len(negative_factors)
                most_influential = shap_values_list[0]["feature"] if shap_values_list else "N/A"

                sm1, sm2, sm3, sm4 = st.columns(4)
                with sm1:
                    st.metric("Tingkat Keyakinan Prediksi", f"{prob_pct:.1f}%")
                with sm2:
                    st.metric("Faktor Pendorong Positif", f"{n_positive}")
                with sm3:
                    st.metric("Faktor Pendorong Negatif", f"{n_negative}")
                with sm4:
                    st.metric("Fitur Paling Berpengaruh", most_influential[:20])

                st.markdown("")

                # ── Section 9: Academic SHAP Note ─────────────────────────
                st_html("""
                <div class="cdss-card">
                    <p style="margin: 0; color: #64748B; font-size: 14px; line-height: 1.7;">
                        <strong style="color: #0F172A;">Tentang Nilai SHAP:</strong>
                        Nilai SHAP (SHapley Additive exPlanations) mengukur seberapa besar kontribusi setiap fitur terhadap prediksi individu.
                        Nilai SHAP positif <strong style="color: #EF4444;">meningkatkan</strong> probabilitas Terapi Oksigen,
                        sementara nilai negatif <strong style="color: #3282B8;">menguranginya</strong>.
                        Jumlah semua nilai SHAP ditambah nilai dasar sama dengan hasil prediksi model untuk pasien ini.
                    </p>
                </div>
                """)
            else:
                from utils.config import show_warning_card
                show_warning_card(
                    "Penjelasan Tidak Tersedia",
                    "Penjelasan SHAP tidak dapat dihitung untuk prediksi ini. Prediksi tetap valid."
                )

            # CDSS Disclaimer block
            st_html("""
            <div class="cdss-card" style="display: flex; align-items: flex-start; gap: 0.8rem; margin-top: 1.5rem;">
                <p style="margin: 0; color: #64748B; font-size: 14px; line-height: 1.6;">
                    <strong>Peringatan:</strong> Rekomendasi ini dihasilkan oleh model prediksi dan hanya boleh digunakan sebagai alat bantu keputusan klinis (CDSS). Keputusan medis akhir tetap menjadi tanggung jawab tenaga kesehatan profesional yang berkualifikasi.
                </p>
            </div>
            """)

        except ModelLoadError:
            show_error_card(
                "Model Prediksi Tidak Dapat Dimuat",
                "Kemungkinan penyebab:<br>• Berkas model tidak ditemukan.<br>• Model tidak kompatibel.<br>• Versi model tidak sesuai.<br><br>Silakan hubungi administrator sistem."
            )
        except Exception as e:
            from utils.config import logger
            logger.error("Single prediction failed: %s", str(e), exc_info=True)
            show_error_card(
                "Prediksi Gagal Diproses",
                "Sistem tidak dapat melakukan prediksi. Periksa kembali data pasien kemudian coba kembali."
            )

# ─── Footer ──────────────────────────────────────────────────────────────────
st_html(render_footer())
