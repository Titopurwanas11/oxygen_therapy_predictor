"""
OxyPredict - Clinical Decision Support System (CDSS)
Main application entry point.
"""

import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.config import (
    setup_page,
    render_page_header,
    render_section_divider,
    render_footer,
    render_empty_state,
)
from utils.monitoring import get_prediction_history, calculate_confidence_distribution

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

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st_html("""
<style>
    .metric-grid-custom {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    @media (max-width: 992px) {
        .metric-grid-custom {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 576px) {
        .metric-grid-custom {
            grid-template-columns: 1fr;
        }
    }
    
    .usage-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .usage-table tr {
        border-bottom: 1px solid #f1f5f9;
    }
    .usage-table tr:last-child {
        border-bottom: none;
    }
    .usage-table td {
        padding: 0.6rem 0;
    }
    .usage-table td.label {
        color: #64748b;
        font-weight: 500;
    }
    .usage-table td.value {
        color: #0f172a;
        font-weight: 700;
        text-align: right;
    }
</style>
""")

# ─── Retrieve Prediction History ─────────────────────────────────────────────
df_history = get_prediction_history()

# ─── SECTION 1: COMPACT CDSS HEADER ──────────────────────────────────────────
st_html(render_page_header(
    "📊",
    "Clinical Activity Dashboard",
    "Real-time CDSS operations, patient risk distributions, diagnostic logs, and model telemetry."
))

# ─── CHECK EMPTY STATE ───────────────────────────────────────────────────────
if df_history.empty:
    st_html(render_empty_state(
        "📊",
        "No prediction history available",
        "Run a prediction to start monitoring model usage."
    ))
else:
    # Prepare statistics
    total_predictions = len(df_history)
    need_oxygen = int((df_history["Prediction"] == "Yes").sum())
    no_oxygen = int((df_history["Prediction"] == "No").sum())
    
    avg_confidence = float(df_history["Confidence"].mean()) if total_predictions > 0 else 0.0
    
    # ─── SECTION 2: METRIC CARDS ─────────────────────────────────────────────
    st_html("""
    <h3 class="section-title-custom">📈 Real-time Summary Metrics</h3>
    """)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Predictions", f"{total_predictions}")
    m2.metric("Need Oxygen", f"{need_oxygen}")
    m3.metric("No Oxygen", f"{no_oxygen}")
    m4.metric("Average Confidence", f"{avg_confidence:.1f}%")
    
    st_html(render_section_divider())
    
    # ─── SECTION 7: CLINICAL SUMMARY ─────────────────────────────────────────
    st_html("""
    <h3 class="section-title-custom">🩺 Clinical Summary</h3>
    """)
    
    # Calculate major risk category
    risk_counts = df_history["Risk Level"].value_counts()
    most_common_risk = risk_counts.index[0] if not risk_counts.empty else "Low Risk"
    need_oxygen_pct = (need_oxygen / total_predictions) * 100 if total_predictions > 0 else 0.0
    
    st_html(f"""
    <div class="cdss-card" style="border-left: 5px solid #3282B8;">
        <p style="margin: 0; color: #1E293B; font-size: 16px; font-weight: 500; line-height: 1.7;">
            <strong style="color: #0F4C75;">Clinical Summary:</strong> In the last {total_predictions} predictions, 
            {need_oxygen_pct:.0f}% of patients were classified as requiring oxygen therapy. 
            The average prediction confidence was {avg_confidence:.1f}%. 
            Most patients were categorized as <strong>{most_common_risk}</strong>.
        </p>
    </div>
    """)
    
    st_html(render_section_divider())
    
    # ─── SECTIONS 3, 4, 5: CHARTS GRID ───────────────────────────────────────
    st_html("""
    <h3 class="section-title-custom">📊 Prediction Analytics</h3>
    """)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 18px; color: #0F172A; margin: 0 0 1rem 0; font-weight: 700;\">Prediction Distribution</h4>")
        
        # Donut Chart
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Need Oxygen", "No Oxygen"],
            values=[need_oxygen, no_oxygen],
            hole=.4,
            marker_colors=["#0F4C75", "#3282B8"],
            textinfo="percent+label",
            showlegend=False
        )])
        fig_donut.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            height=220,
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        st_html("</div>")
        
    with col_chart2:
        st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 18px; color: #0F172A; margin: 0 0 1rem 0; font-weight: 700;\">Risk Level Distribution</h4>")
        
        # Risk levels distribution chart
        r_counts = []
        r_labels = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
        r_colors = ["#22C55E", "#F59E0B", "#EA580C", "#EF4444"]
        
        for r_lbl in r_labels:
            r_counts.append(int((df_history["Risk Level"].str.contains(r_lbl, case=False, na=False)).sum()))
            
        fig_risk = go.Figure(data=[go.Bar(
            x=r_labels,
            y=r_counts,
            marker_color=r_colors,
            text=r_counts,
            textposition="auto"
        )])
        fig_risk.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=220,
            xaxis_title=None,
            yaxis_title="Patient Count"
        )
        st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})
        st_html("</div>")
        
    st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 18px; color: #0F172A; margin: 0 0 1rem 0; font-weight: 700;\">Model Confidence Distribution</h4>")
    
    # Confidence distribution bar chart
    c_bins = ["60–70%", "70–80%", "80–90%", "90–100%"]
    c_counts = [0, 0, 0, 0]
    
    for _, row in df_history.iterrows():
        c_val = row["Confidence"]
        if c_val >= 90.0:
            c_counts[3] += 1
        elif c_val >= 80.0:
            c_counts[2] += 1
        elif c_val >= 70.0:
            c_counts[1] += 1
        else:
            c_counts[0] += 1
            
    fig_conf = go.Figure(data=[go.Bar(
        x=c_bins,
        y=c_counts,
        marker_color="#3282B8",
        text=c_counts,
        textposition="auto"
    )])
    fig_conf.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        xaxis_title="Confidence Ranges",
        yaxis_title="Patient Count"
    )
    st.plotly_chart(fig_conf, use_container_width=True, config={'displayModeBar': False})
    st_html("</div>")
    
    st_html(render_section_divider())
    
    # ─── SECTION 8: PREDICTION TREND ─────────────────────────────────────────
    st_html("""
    <h3 class="section-title-custom">📈 Prediction Trends</h3>
    """)
    
    # Parse timestamp to date
    try:
        df_history["Date"] = pd.to_datetime(df_history["Timestamp"]).dt.date
        trend_df = df_history.groupby("Date").size().reset_index(name="Predictions")
        
        if len(trend_df) >= 2:
            fig_trend = px.line(
                trend_df,
                x="Date",
                y="Predictions",
                labels={"Date": "Tanggal", "Predictions": "Jumlah Prediksi"},
                markers=True
            )
            fig_trend.update_traces(line_color="#3282B8", marker=dict(size=8, color="#0F4C75"))
            fig_trend.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=240,
                xaxis_title="Date",
                yaxis_title="Total Predictions"
            )
            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("ℹ️ Not enough historical data to display prediction trends.")
    except Exception:
        st.info("ℹ️ Not enough historical data to display prediction trends.")
        
    st_html(render_section_divider())
    
    # ─── SECTION 6 & 9: RECENT TABLE & SYSTEM STATS ──────────────────────────
    col_table, col_stats = st.columns([2, 1])
    
    with col_table:
        st_html("""
        <div class="cdss-card" style="height: 100%;">
            <h4 style="font-size: 18px; color: #0F172A; margin: 0 0 1rem 0; font-weight: 700;">📋 Recent Predictions (Last 10)</h4>
        """)
        
        # Get last 10 entries ordered by timestamp descending
        recent_df = df_history.tail(10).iloc[::-1].copy()
        
        # format columns
        recent_table_df = pd.DataFrame({
            "Date & Time": recent_df["Timestamp"],
            "Patient Age (m)": recent_df["Age"].astype(int),
            "Prediction": recent_df["Prediction"],
            "Confidence": recent_df["Confidence"].map(lambda x: f"{x:.1f}%"),
            "Risk Level": recent_df["Risk Level"]
        })
        
        st.dataframe(recent_table_df, use_container_width=True, hide_index=True)
        st_html("</div>")
        
    with col_stats:
        st_html("""
        <div class="cdss-card" style="height: 100%;">
            <h4 style="font-size: 18px; color: #0F172A; margin: 0 0 1rem 0; font-weight: 700;">⚙️ System Usage Statistics</h4>
            <table class="usage-table">
        """)
        
        # Calculate usage values
        single_cnt = int((df_history["Type"] == "Single").sum())
        batch_cnt = int((df_history["Type"] == "Batch").sum())
        last_time = df_history["Timestamp"].iloc[-1] if not df_history.empty else "-"
        
        st_html(f"""
                <tr>
                    <td class="label">Single Predictions</td>
                    <td class="value">{single_cnt}</td>
                </tr>
                <tr>
                    <td class="label">Batch Predictions</td>
                    <td class="value">{batch_cnt}</td>
                </tr>
                <tr>
                    <td class="label">Last Prediction Time</td>
                    <td class="value" style="font-size:0.75rem;">{last_time}</td>
                </tr>
            </table>
        </div>
        """)

st.markdown("")

# ─── Footer ──────────────────────────────────────────────────────────────────
st_html(render_footer())
