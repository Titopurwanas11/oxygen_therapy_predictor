"""
Monitoring Prediksi — Halaman pemantauan aktivitas klinis dan log prediksi OxyPredict.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.config import (
    setup_page,
    render_page_header,
    render_section_divider,
    render_footer,
    render_empty_state,
)
from utils.monitoring import get_prediction_history

st.set_page_config(
    page_title="Monitoring Prediksi — OxyPredict",
    page_icon="assets/favicon-64x64.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_page("Monitoring Prediksi — OxyPredict")


def st_html(html_str):
    cleaned_lines = [line.lstrip() for line in html_str.splitlines()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)


# ─── Ambil Riwayat Prediksi ──────────────────────────────────────────────────
df_history = get_prediction_history()

# ─── HEADER ──────────────────────────────────────────────────────────────────
st_html(render_page_header(
    "📊",
    "Monitoring Prediksi",
    "Pemantauan aktivitas klinis CDSS secara real-time, log prediksi, distribusi risiko pasien, dan statistik penggunaan sistem."
))

# ─── EMPTY STATE ─────────────────────────────────────────────────────────────
if df_history.empty:
    st_html(render_empty_state(
        "📊",
        "Belum Ada Riwayat Prediksi",
        "Lakukan prediksi pasien terlebih dahulu untuk mulai memantau aktivitas sistem."
    ))
else:
    total_prediksi = len(df_history)
    butuh_oksigen = int((df_history["Prediction"] == "Yes").sum())
    tidak_butuh = int((df_history["Prediction"] == "No").sum())
    rata_keyakinan = float(df_history["Confidence"].mean()) if total_prediksi > 0 else 0.0

    # ─── RINGKASAN METRIK ─────────────────────────────────────────────────────
    st_html("""<h3 class="section-title-custom">📈 Ringkasan Aktivitas Klinis</h3>""")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Prediksi", f"{total_prediksi}")
    m2.metric("Membutuhkan Oksigen", f"{butuh_oksigen}")
    m3.metric("Tidak Membutuhkan Oksigen", f"{tidak_butuh}")
    m4.metric("Rata-rata Tingkat Keyakinan", f"{rata_keyakinan:.1f}%")

    st_html(render_section_divider())

    # ─── RINGKASAN KLINIS ─────────────────────────────────────────────────────
    st_html("""<h3 class="section-title-custom">🩺 Ringkasan Klinis</h3>""")

    risk_counts = df_history["Risk Level"].value_counts()
    risiko_terbanyak = risk_counts.index[0] if not risk_counts.empty else "Risiko Rendah"
    pct_butuh_oksigen = (butuh_oksigen / total_prediksi) * 100 if total_prediksi > 0 else 0.0

    st_html(f"""
    <div class="cdss-card" style="border-left: 5px solid #3282B8;">
        <p style="margin: 0; color: #1E293B; font-size: 16px; font-weight: 500; line-height: 1.7;">
            <strong style="color: #0F4C75;">Ringkasan Klinis:</strong> Berdasarkan {total_prediksi} prediksi yang telah dilakukan,
            sebanyak {pct_butuh_oksigen:.0f}% pasien diklasifikasikan membutuhkan terapi oksigen.
            Rata-rata tingkat keyakinan model adalah {rata_keyakinan:.1f}%.
            Sebagian besar pasien masuk dalam kategori <strong>{risiko_terbanyak}</strong>.
        </p>
    </div>
    """)

    st_html(render_section_divider())

    # ─── GRAFIK ANALITIK ─────────────────────────────────────────────────────
    st_html("""<h3 class="section-title-custom">📊 Analitik Prediksi</h3>""")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 18px; color: #0F172A; margin: 0 0 1rem 0; font-weight: 700;\">Distribusi Hasil Prediksi</h4>")
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Membutuhkan Oksigen", "Tidak Membutuhkan"],
            values=[butuh_oksigen, tidak_butuh],
            hole=.4,
            marker_colors=["#0F4C75", "#3282B8"],
            textinfo="percent+label",
            showlegend=False
        )])
        fig_donut.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=220)
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        st_html("</div>")

    with col_chart2:
        st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 18px; color: #0F172A; margin: 0 0 1rem 0; font-weight: 700;\">Distribusi Tingkat Risiko</h4>")
        r_labels = ["Risiko Rendah", "Risiko Sedang", "Risiko Tinggi", "Risiko Sangat Tinggi"]
        r_keys   = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
        r_colors = ["#22C55E", "#F59E0B", "#EA580C", "#EF4444"]
        r_counts = [int((df_history["Risk Level"].str.contains(k, case=False, na=False)).sum()) for k in r_keys]

        fig_risk = go.Figure(data=[go.Bar(
            x=r_labels, y=r_counts, marker_color=r_colors,
            text=r_counts, textposition="auto"
        )])
        fig_risk.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220,
                               xaxis_title=None, yaxis_title="Jumlah Pasien")
        st.plotly_chart(fig_risk, use_container_width=True, config={'displayModeBar': False})
        st_html("</div>")

    # Distribusi Tingkat Keyakinan
    st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 18px; color: #0F172A; margin: 0 0 1rem 0; font-weight: 700;\">Distribusi Tingkat Keyakinan Model</h4>")
    c_bins = ["60–70%", "70–80%", "80–90%", "90–100%"]
    c_counts = [0, 0, 0, 0]
    for _, row in df_history.iterrows():
        c_val = row["Confidence"]
        if c_val >= 90.0:   c_counts[3] += 1
        elif c_val >= 80.0: c_counts[2] += 1
        elif c_val >= 70.0: c_counts[1] += 1
        else:               c_counts[0] += 1

    fig_conf = go.Figure(data=[go.Bar(
        x=c_bins, y=c_counts, marker_color="#3282B8",
        text=c_counts, textposition="auto"
    )])
    fig_conf.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220,
                           xaxis_title="Rentang Tingkat Keyakinan",
                           yaxis_title="Jumlah Pasien")
    st.plotly_chart(fig_conf, use_container_width=True, config={'displayModeBar': False})
    st_html("</div>")

    st_html(render_section_divider())

    # ─── TREN PENGGUNAAN ─────────────────────────────────────────────────────
    st_html("""<h3 class="section-title-custom">📈 Tren Penggunaan</h3>""")

    try:
        df_history["Tanggal"] = pd.to_datetime(df_history["Timestamp"]).dt.date
        trend_df = df_history.groupby("Tanggal").size().reset_index(name="Jumlah Prediksi")

        if len(trend_df) >= 2:
            fig_trend = px.line(
                trend_df, x="Tanggal", y="Jumlah Prediksi",
                markers=True
            )
            fig_trend.update_traces(line_color="#3282B8", marker=dict(size=8, color="#0F4C75"))
            fig_trend.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240,
                                    xaxis_title="Tanggal", yaxis_title="Total Prediksi")
            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("ℹ️ Data historis belum cukup untuk menampilkan tren penggunaan.")
    except Exception:
        st.info("ℹ️ Data historis belum cukup untuk menampilkan tren penggunaan.")

    st_html(render_section_divider())

    # ─── LOG PREDIKSI TERBARU & STATISTIK ─────────────────────────────────
    col_table, col_stats = st.columns([2, 1])

    with col_table:
        st_html("""
        <div class="cdss-card" style="height: 100%;">
            <h4 style="font-size: 18px; color: #0F172A; margin: 0 0 1rem 0; font-weight: 700;">📋 Log Prediksi Terbaru (10 Terakhir)</h4>
        """)
        recent_df = df_history.tail(10).iloc[::-1].copy()
        recent_table_df = pd.DataFrame({
            "Waktu Prediksi": recent_df["Timestamp"],
            "Usia Pasien (bln)": recent_df["Age"].astype(int),
            "Hasil Prediksi": recent_df["Prediction"].map({"Yes": "Butuh Oksigen", "No": "Tidak Butuh"}),
            "Tingkat Keyakinan": recent_df["Confidence"].map(lambda x: f"{x:.1f}%"),
            "Tingkat Risiko": recent_df["Risk Level"]
        })
        st.dataframe(recent_table_df, use_container_width=True, hide_index=True)
        st_html("</div>")

    with col_stats:
        single_cnt = int((df_history["Type"] == "Single").sum())
        batch_cnt  = int((df_history["Type"] == "Batch").sum())
        last_time  = df_history["Timestamp"].iloc[-1] if not df_history.empty else "-"

        st_html(f"""
        <div class="cdss-card" style="height: 100%;">
            <h4 style="font-size: 18px; color: #0F172A; margin: 0 0 1rem 0; font-weight: 700;">⚙️ Statistik Penggunaan</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 0.6rem 0; color: #64748b; font-weight: 500;">Prediksi Pasien Tunggal</td>
                    <td style="padding: 0.6rem 0; color: #0f172a; font-weight: 700; text-align: right;">{single_cnt}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 0.6rem 0; color: #64748b; font-weight: 500;">Prediksi Massal</td>
                    <td style="padding: 0.6rem 0; color: #0f172a; font-weight: 700; text-align: right;">{batch_cnt}</td>
                </tr>
                <tr>
                    <td style="padding: 0.6rem 0; color: #64748b; font-weight: 500;">Waktu Prediksi Terakhir</td>
                    <td style="padding: 0.6rem 0; color: #0f172a; font-weight: 700; text-align: right; font-size: 0.75rem;">{last_time}</td>
                </tr>
            </table>
        </div>
        """)

st.markdown("")
st_html(render_footer())
