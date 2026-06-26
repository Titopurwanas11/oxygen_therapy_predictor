"""
AI Monitoring Dashboard Page for OxyPredict.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime

from utils.config import setup_page
from utils.monitoring import get_monitoring_data, calculate_confidence_distribution
from utils.system_health import get_system_health_status
from utils.session_analytics import get_session_runtime_str, init_analytics_state
from utils.charts import (
    create_prediction_ratio_donut,
    create_risk_distribution_chart,
    create_probability_histogram,
    create_avg_prob_per_risk_chart,
    create_confidence_histogram,
    create_avg_confidence_gauge
)

st.set_page_config(page_title="AI Monitoring Dashboard — OxyPredict", page_icon="📈", layout="wide")
setup_page("AI Monitoring Dashboard — OxyPredict")
init_analytics_state()

# Helper function to render HTML safely
def st_html(html_str):
    cleaned_lines = [line.lstrip() for line in html_str.splitlines()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)

# CSS Styling
st_html("""
<style>
    .cdss-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.02);
        margin-bottom: 1.25rem;
    }
    .section-title-custom {
        color: #0a2e52;
        font-weight: 700;
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #2563eb;
        padding-left: 0.6rem;
    }
</style>
""")

# ─── Hero Header ─────────────────────────────────────────────────────────────
st_html("""
<div style="
    background: linear-gradient(135deg, #0a2e52 0%, #153e75 50%, #2563eb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(10, 46, 82, 0.18);
">
    <h1 style="margin: 0; color: white; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.3px;">📈 AI Monitoring Dashboard</h1>
    <p style="margin: 0.3rem 0 0 0; color: #93c5fd; font-size: 0.95rem;">
        Monitor model performance, prediction statistics, system health, and usage telemetry in real-time.
    </p>
</div>
""")

# Fetch System Health & Session runtime
health_status = get_system_health_status()
uptime_str = get_session_runtime_str()

# ─── System Health & Status ──────────────────────────────────────────────────
st_html("<h3 class=\"section-title-custom\">🟢 System Status & Health</h3>")

h_col1, h_col2, h_col3, h_col4, h_col5, h_col6 = st.columns(6)

with h_col1:
    st.metric("System Status", "🟢 Online")
with h_col2:
    st.metric("Session Uptime", uptime_str)
with h_col3:
    status_icon = "🟢 Healthy" if health_status.get("Model Loaded") == "Healthy" else "🔴 Error"
    st.metric("Model Status", status_icon)
with h_col4:
    status_icon = "🟢 Healthy" if health_status.get("SHAP Loaded") == "Healthy" else "🔴 Error"
    st.metric("SHAP Service", status_icon)
with h_col5:
    status_icon = "🟢 Healthy" if health_status.get("Prediction Service") == "Healthy" else "🔴 Error"
    st.metric("Prediction Engine", status_icon)
with h_col6:
    status_icon = "🟢 Healthy" if health_status.get("PDF Generator") == "Healthy" else "🔴 Error"
    st.metric("PDF Compiler", status_icon)

# ─── Model Static Performance metrics ──────────────────────────────────────────
st_html("<h3 class=\"section-title-custom\">⚙️ Model Static Performance Metrics</h3>")
col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
with col_perf1:
    st.metric("Accuracy", "91.8%")
with col_perf2:
    st.metric("F1 Macro Score", "92.4%")
with col_perf3:
    st.metric("ROC-AUC Score", "94.2%")
with col_perf4:
    st.metric("Predictive Features", "44 Columns")

# ─── Data Source & Prediction Analytics ──────────────────────────────────────
df, is_dummy = get_monitoring_data()

st_html("<h3 class=\"section-title-custom\">📊 Patient Analytics Dashboard</h3>")

if is_dummy:
    st_html("""
    <div style="
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 5px solid #d97706;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
    ">
        <span style="font-size: 0.9rem; font-weight: 700; color: #92400e;">⚠️ Sample Monitoring Data:</span>
        <span style="font-size: 0.85rem; color: #b45309;">No active batch results available in session. Showing simulated hospital monitoring data of 120 patients. Run a Batch Prediction to analyze live population results.</span>
    </div>
    """)
else:
    st_html("""
    <div style="
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
    ">
        <span style="font-size: 0.9rem; font-weight: 700; color: #14532d;">🟢 Live Production Data:</span>
        <span style="font-size: 0.85rem; color: #166534;">Analyzing active batch prediction results from the uploaded patient dataset.</span>
    </div>
    """)

# Compute Average Confidence
confidences = np.where(df["Prediction"] == "Yes", df["Probability"], 100.0 - df["Probability"])
avg_confidence = float(confidences.mean())

# Graphics Layout Grid
v_col1, v_col2 = st.columns(2)
with v_col1:
    st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Oxygen Therapy Need Ratio</h4>")
    st.plotly_chart(create_prediction_ratio_donut(df), use_container_width=True)
    st_html("</div>")
with v_col2:
    st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Risk Level Distribution</h4>")
    st.plotly_chart(create_risk_distribution_chart(df), use_container_width=True)
    st_html("</div>")

v_col3, v_col4 = st.columns(2)
with v_col3:
    st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Prediction Probability Distribution Histogram</h4>")
    st.plotly_chart(create_probability_histogram(df), use_container_width=True)
    st_html("</div>")
with v_col4:
    st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Average Probability per Risk Level</h4>")
    st.plotly_chart(create_avg_prob_per_risk_chart(df), use_container_width=True)
    st_html("</div>")

v_col5, v_col6 = st.columns(2)
with v_col5:
    st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Confidence Level Distribution</h4>")
    st.plotly_chart(create_confidence_histogram(df), use_container_width=True)
    st_html("</div>")
with v_col6:
    st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Average Model Confidence</h4>")
    st.plotly_chart(create_avg_confidence_gauge(avg_confidence), use_container_width=True)
    st_html("</div>")

# ─── Confidence Table ────────────────────────────────────────────────────────
st_html("<h3 class=\"section-title-custom\">📋 Model Confidence Level Analysis</h3>")
dist_df = calculate_confidence_distribution(df)
st.dataframe(dist_df, use_container_width=True, hide_index=True)

# ─── Telemetry Usage & Activity Log ──────────────────────────────────────────
st_html("<h3 class=\"section-title-custom\">📈 Telemetry and Usage Logs</h3>")

u_col1, u_col2 = st.columns(2)

with u_col1:
    st_html("""
    <div class="cdss-card" style="height: 100%;">
        <h4 style="font-size: 0.95rem; color: #0a2e52; margin: 0 0 1.2rem 0; font-weight: 700;">System Usage Statistics (Current Session)</h4>
        <div style="display: flex; flex-direction: column; gap: 0.8rem; font-size: 0.9rem;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.4rem;">
                <span style="color: #475569;">Single Patient Predictions:</span>
                <span style="font-weight: 700; color: #0f172a;">{single_cnt}</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.4rem;">
                <span style="color: #475569;">Batch Patient Predictions:</span>
                <span style="font-weight: 700; color: #0f172a;">{batch_cnt}</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.4rem;">
                <span style="color: #475569;">PDF Reports Generated:</span>
                <span style="font-weight: 700; color: #0f172a;">{pdf_cnt}</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.4rem;">
                <span style="color: #475569;">Excel Exports Downloaded:</span>
                <span style="font-weight: 700; color: #0f172a;">{excel_cnt}</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.4rem;">
                <span style="color: #475569;">CSV Datasets Downloaded:</span>
                <span style="font-weight: 700; color: #0f172a;">{csv_cnt}</span>
            </div>
        </div>
    </div>
    """.format(
        single_cnt=st.session_state.get("single_predictions_count", 0),
        batch_cnt=st.session_state.get("batch_predictions_count", 0),
        pdf_cnt=st.session_state.get("pdf_reports_count", 0),
        excel_cnt=st.session_state.get("excel_downloads_count", 0),
        csv_cnt=st.session_state.get("csv_downloads_count", 0)
    ))

with u_col2:
    st_html("<div class=\"cdss-card\" style=\"height: 100%;\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1.2rem 0; font-weight: 700;\">Recent System Activity Log</h4>")
    logs = st.session_state.get("activity_log", [])
    if not logs:
        st_html("<p style='font-style: italic; color: #94a3b8; font-size: 0.85rem;'>No activities logged in this session yet.</p>")
    else:
        log_rows = ""
        for log in logs[:6]:
            status_color = "#16a34a" if log["status"] in ["Success", "Completed"] else "#dc2626"
            log_rows += f"""
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; border-bottom: 1px solid #f1f5f9; padding: 0.4rem 0;">
                <span style="color: #64748b; width: 80px;">[{log['time']}]</span>
                <span style="color: #0f172a; font-weight: 600; flex-grow: 1;">{log['action']}</span>
                <span style="color: {status_color}; font-weight: 700;">{log['status']}</span>
            </div>
            """
        st_html(log_rows)
    st_html("</div>")

# ─── System Limitations & CDSS Disclaimer ───────────────────────────────────
st_html("<h3 class=\"section-title-custom\">⚠️ System Limitations & Academic Notes</h3>")
st_html("""
<div style="
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    font-size: 0.8rem;
    line-height: 1.7;
    color: #475569;
">
    <strong style="color: #0a2e52; font-size: 0.85rem; display: block; margin-bottom: 0.5rem;">📚 Academic Research Notes:</strong>
    <ul>
        <li><strong>Dataset Constraints:</strong> The underlying classifier model is trained on a retrospectively gathered clinical dataset of 801 pediatric patients experiencing respiratory symptoms.</li>
        <li><strong>Real-time Telemetry:</strong> Session counters are tracked in memory (`st.session_state`) and reset if the Streamlit server restarts.</li>
        <li><strong>Explainable AI:</strong> Explainability via SHAP calculations is computed offline or on demand using single patient diagnostics.</li>
        <li><strong>Clinical Disclaimer:</strong> OxyPredict is intended to act as a Clinical Decision Support System (CDSS). Final validation and treatment selection must be performed by a registered physician.</li>
    </ul>
</div>
""")

# ─── Quick Actions ───────────────────────────────────────────────────────────
st_html("<h3 class=\"section-title-custom\">🔗 Quick Actions</h3>")

qa_col1, qa_col2, qa_col3 = st.columns(3)
with qa_col1:
    st.page_link("pages/2_Single_Prediction.py", label="👉 Go to Single Patient Diagnostic", icon="🩺")
with qa_col2:
    st.page_link("pages/3_Batch_Prediction.py", label="👉 Go to CDSS Batch Analysis", icon="📊")
with qa_col3:
    st.page_link("pages/1_Dashboard.py", label="👉 View Core Dashboard", icon="🫁")
