"""
Batch Prediction & Population Analysis Page for OxyPredict.
Redesigned as a modern hospital Clinical Decision Support System (CDSS) dashboard.
"""

import streamlit as st
import pandas as pd
import io
import datetime

from utils.config import ALL_FEATURES, setup_page, render_page_header, render_section_divider, render_footer, render_empty_state
from utils.batch_prediction import validate_uploaded_file, run_batch_prediction
from utils.statistics import calculate_population_stats, generate_population_narrative
from utils.charts import (
    create_pie_chart,
    create_risk_distribution_chart,
    create_probability_histogram,
    create_avg_prob_per_risk_chart,
)
from utils.batch_report import generate_batch_pdf_report
from utils.session_analytics import (
    init_analytics_state,
    track_batch_prediction,
    track_excel_download,
    track_csv_download,
    track_pdf_report_generated,
)


st.set_page_config(page_title="Batch Analysis — OxyPredict", page_icon="🫁", layout="wide")
setup_page("Batch Analysis — OxyPredict")
init_analytics_state()


# Helper function to render HTML safely without markdown layout side-effects
def st_html(html_str):
    cleaned_lines = [line.lstrip() for line in html_str.splitlines()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)


# ─── Custom Layout Styling ───────────────────────────────────────────────────
st_html("""
<style>
    /* Page-specific overrides only */
</style>
""")

# ─── Header ──────────────────────────────────────────────────────────────────
st_html(render_page_header(
    "📂",
    "CDSS Population & Batch Analysis",
    "Sistem Pendukung Keputusan Klinis untuk Analisis Populasi Terapi Oksigen (Skripsi Demo)"
))

# ─── SECTION 1: UPLOAD ───────────────────────────────────────────────────────
st_html("<h3 class=\"section-title-custom\">📁 Section 1: Upload Patient Dataset</h3>")

col_upload_left, col_upload_right = st.columns([3, 2])

with col_upload_left:
    uploaded_file = st.file_uploader(
        "Pilih file CSV atau XLSX",
        type=["csv", "xlsx"],
        help="Upload file berisi data klinis pasien untuk dianalisis kebutuhan oksigennya.",
    )

with col_upload_right:
    st_html("""
    <div class="cdss-card" style="background-color: #f8fafc; border-color: #e2e8f0; height: 100%; min-height: 140px; padding: 1.1rem 1.4rem;">
        <h5 style="margin: 0 0 0.5rem 0; color: #0f172a; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Supported Format</h5>
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.6rem;">
            <span style="background-color: #3b82f6; color: white; font-size: 0.72rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 5px;">CSV</span>
            <span style="background-color: #10b981; color: white; font-size: 0.72rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 5px;">Excel (.xlsx)</span>
        </div>
        <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">
            <strong>Required Fitur:</strong> 44 Fitur Klinis Pasien<br>
            <strong>Dataset Validasi:</strong> Otomatis oleh Sistem
        </div>
    </div>
    """)

# ─── Show required columns ──────────────────────────────────────────────────
with st.expander("📋 Lihat Daftar Kolom yang Diperlukan (44 fitur)", expanded=False):
    col_a, col_b = st.columns(2)
    half = len(ALL_FEATURES) // 2 + 1
    with col_a:
        for i, feat in enumerate(ALL_FEATURES[:half], 1):
            st.markdown(f"`{i}.` {feat}")
    with col_b:
        for i, feat in enumerate(ALL_FEATURES[half:], half + 1):
            st.markdown(f"`{i}.` {feat}")

# Process file if uploaded
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {str(e)}")
        st.stop()

    # ─── SECTION 2: DATA VALIDATION ──────────────────────────────────────────
    st_html("<h3 class=\"section-title-custom\">✅ Section 2: Dataset Verification</h3>")

    is_valid, missing_cols, extra_cols = validate_uploaded_file(df_raw)

    v1, v2, v3, v4, v5 = st.columns(5)
    
    with v1:
        st.metric("👥 Patients Count", len(df_raw))
    with v2:
        st.metric("📋 Columns Count", len(df_raw.columns))
    with v3:
        st.metric("Missing Columns", len(missing_cols), delta=None, delta_color="inverse")
    with v4:
        st.metric("Extra Columns", len(extra_cols))
    with v5:
        st.metric("Ready for Prediction", "Yes" if is_valid else "No")

    st.markdown("")

    if is_valid:
        st_html("""
        <div style="
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 5px solid #16a34a;
            border-radius: 10px;
            padding: 0.8rem 1.2rem;
            margin-bottom: 1rem;
        ">
            <span style="font-size: 0.9rem; font-weight: 700; color: #14532d;">✔ Validasi Sukses:</span>
            <span style="font-size: 0.85rem; color: #166534;">Semua 44 kolom klinis ditemukan. Dataset siap dianalisis.</span>
        </div>
        """)

        with st.expander("👀 Preview Data (5 baris pertama)", expanded=False):
            st.dataframe(df_raw.head(), use_container_width=True)

        st.markdown("")

        # ─── SECTION 3: RUN PREDICTION ──────────────────────────────────────
        st_html("<h3 class=\"section-title-custom\">🔮 Section 3: Population Diagnostics</h3>")
        
        col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
        with col_btn_c:
            predict_all = st.button(
                "🔮 Predict All Patients",
                use_container_width=True,
                type="primary"
            )

        if predict_all or st.session_state.get("batch_predicted", False):
            # Maintain state so prediction results don't vanish on UI interaction
            if predict_all or "batch_results" not in st.session_state:
                with st.spinner("Running Random Forest Model..."):
                    try:
                        result_df = run_batch_prediction(df_raw)
                        track_batch_prediction()
                        try:
                            from utils.monitoring import record_predictions_from_df
                            record_predictions_from_df(result_df, type="Batch")
                        except Exception:
                            pass
                        # Save in session state
                        st.session_state.batch_results = result_df
                        st.session_state.batch_predicted = True
                    except Exception as pred_err:
                        st.error(f"❌ Terjadi kesalahan saat memproses prediksi: {pred_err}")
                        st.stop()

            # Retrieve results
            result_df = st.session_state.batch_results
            stats = calculate_population_stats(result_df)
            narrative = generate_population_narrative(stats)

            st.markdown("---")

            # ─── SECTION 4: POPULATION SUMMARY ───────────────────────────────
            st_html("<h3 class=\"section-title-custom\">📊 Section 4: Population Summary Metrics</h3>")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Patients", f"{stats['total_patients']}")
            m2.metric("Patients Need Oxygen", f"{stats['need_oxy']}", f"{stats['need_oxy_pct']:.1f}%", delta_color="inverse")
            m3.metric("Patients Without Oxygen", f"{stats['no_oxy']}")
            m4.metric("Average Probability", f"{stats['avg_probability']:.1f}%")

            m5, m6, m7 = st.columns(3)
            m5.metric("High Risk Patients", f"{stats['high_risk']}")
            m6.metric("Medium Risk Patients", f"{stats['med_risk']}")
            m7.metric("Low Risk Patients", f"{stats['low_risk']}")

            st.markdown("")

            # ─── SECTION 5: VISUALIZATION ────────────────────────────────────
            st_html("<h3 class=\"section-title-custom\">📈 Section 5: Graphical Analytics Dashboard</h3>")
            
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Oxygen Therapy Need Ratio</h4>")
                st.plotly_chart(create_pie_chart(result_df), use_container_width=True)
                st_html("</div>")
            with v_col2:
                st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Risk Level Distribution</h4>")
                st.plotly_chart(create_risk_distribution_chart(result_df), use_container_width=True)
                st_html("</div>")

            v_col3, v_col4 = st.columns(2)
            with v_col3:
                st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Prediction Probability Distribution Histogram</h4>")
                st.plotly_chart(create_probability_histogram(result_df), use_container_width=True)
                st_html("</div>")
            with v_col4:
                st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 0.95rem; color: #0a2e52; margin: 0 0 1rem 0; font-weight: 700;\">Average Probability per Risk Level</h4>")
                st.plotly_chart(create_avg_prob_per_risk_chart(result_df), use_container_width=True)
                st_html("</div>")

            # ─── SECTION 7: POPULATION CLINICAL SUMMARY ──────────────────────
            st_html("<h3 class=\"section-title-custom\">🩺 Section 6: Population Clinical Interpretation</h3>")
            
            st_html(f"""
            <div style="
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-left: 6px solid #2563eb;
                border-radius: 14px;
                padding: 1.5rem;
                box-shadow: 0 4px 10px rgba(0,0,0,0.02);
                margin-bottom: 1.5rem;
            ">
                <h4 style="margin: 0 0 0.8rem 0; color: #0a2e52; font-weight: 700; font-size: 0.95rem;">CDSS Population Narrative Summary</h4>
                <p style="margin: 0; color: #334155; font-size: 0.88rem; line-height: 1.7; text-align: justify;">
                    {narrative}
                </p>
            </div>
            """)

            # ─── SECTION 6: INTERACTIVE TABLE & SEARCH & FILTER ──────────────
            st_html("<h3 class=\"section-title-custom\">📋 Section 7: Enriched Patient Dataset & Query Filters</h3>")
            
            # Filters block
            st_html("<div class=\"cdss-card\" style=\"background-color: #f8fafc;\"><h5 style=\"margin:0 0 1rem 0; font-size:0.85rem; font-weight:700; color:#475569;\">🔍 FILTER CRITERIA</h5>")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                filter_pred = st.selectbox("Prediction", ["All", "Yes", "No"])
            with f2:
                filter_risk = st.selectbox("Risk Level", ["All", "Low Risk", "Low-Moderate Risk", "Moderate Risk", "High Risk", "Very High Risk"])
            with f3:
                min_prob, max_prob = st.slider("Probability Range (%)", 0.0, 100.0, (0.0, 100.0))
            with f4:
                search_query = st.text_input("Search Patient ID", "")
            st_html("</div>")

            # Apply filters
            filtered_df = result_df.copy()
            
            # Make sure a Patient ID column exists or generate one for search/indexing
            if "Patient ID" not in filtered_df.columns:
                filtered_df.insert(0, "Patient ID", [f"Patient #{i+1}" for i in range(len(filtered_df))])

            if filter_pred != "All":
                filtered_df = filtered_df[filtered_df["Prediction"] == filter_pred]
            if filter_risk != "All":
                filtered_df = filtered_df[filtered_df["Risk Level"] == filter_risk]
            
            filtered_df = filtered_df[(filtered_df["Probability"] >= min_prob) & (filtered_df["Probability"] <= max_prob)]
            
            if search_query:
                filtered_df = filtered_df[filtered_df["Patient ID"].astype(str).str.contains(search_query, case=False)]

            # Table display order
            display_cols = ["Patient ID", "Prediction", "Probability", "Confidence Level", "Risk Level", "Recommendation", "Priority"]
            key_features = ["Age (months)", "Gender", "Weight (Kg)", "Oxygen saturation (SaO2) at admission", "Respiratory rate", "Heart rate"]
            display_order = display_cols + [c for c in key_features if c in filtered_df.columns]
            other_cols = [c for c in filtered_df.columns if c not in display_order]

            # Style row highlights & gradient
            def highlight_row(row):
                pred = row["Prediction"]
                if pred == "Yes":
                    return ["background-color: #fee2e2; color: #991b1b;"] * len(row)
                elif pred == "No":
                    return ["background-color: #dcfce7; color: #14532d;"] * len(row)
                return [""] * len(row)

            # Style display df
            final_df = filtered_df[display_order + other_cols]
            
            st.dataframe(
                final_df.style.apply(highlight_row, axis=1, subset=["Prediction", "Patient ID"]),
                use_container_width=True,
                height=350
            )

            st.markdown("")

            # ─── SECTION 11: EXPORTS ─────────────────────────────────────────
            st_html("<h3 class=\"section-title-custom\">💾 Section 8: Download & Export Reports</h3>")
            
            dl1, dl2, dl3 = st.columns(3)

            # Excel download with enriched columns
            excel_buffer = io.BytesIO()
            final_df.to_excel(excel_buffer, index=False, engine="openpyxl")
            excel_buffer.seek(0)
            with dl1:
                st.download_button(
                    label="📥 Download Excel Clinical Data",
                    data=excel_buffer,
                    file_name=f"OxyPredict_Batch_Export_{datetime.date.today()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    on_click=track_excel_download,
                )

            # CSV download with enriched columns
            csv_buffer = final_df.to_csv(index=False).encode("utf-8")
            with dl2:
                st.download_button(
                    label="📥 Download CSV Dataset",
                    data=csv_buffer,
                    file_name=f"OxyPredict_Batch_Export_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    on_click=track_csv_download,
                )

            # PDF Batch Report download
            try:
                pdf_report_bytes = generate_batch_pdf_report(result_df, stats, narrative)
            except Exception as pdf_err:
                st.error(f"❌ Gagal membuat PDF Report: {pdf_err}")
                pdf_report_bytes = None

            with dl3:
                if pdf_report_bytes:
                    st.download_button(
                        label="📄 Download Batch Clinical Report (PDF)",
                        data=pdf_report_bytes,
                        file_name=f"OxyPredict_Batch_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        on_click=track_pdf_report_generated,
                    )
                else:
                    st.info("ℹ️ Run PDF compiler to debug report generation.")

    else:
        # Columns verification failed
        st_html(f"""
        <div style="
            background: #fdf2f2;
            border: 1px solid #fecaca;
            border-left: 5px solid #dc2626;
            border-radius: 10px;
            padding: 0.8rem 1.2rem;
            margin-bottom: 1rem;
        ">
            <span style="font-size: 0.9rem; font-weight: 700; color: #7f1d1d;">❌ Validasi Gagal:</span>
            <span style="font-size: 0.85rem; color: #991b1b;">Dataset tidak lengkap. Ditemukan {len(missing_cols)} kolom yang hilang.</span>
        </div>
        """)

        with st.expander("🔍 Lihat Kolom yang Hilang", expanded=True):
            for col in sorted(missing_cols):
                st.markdown(f"- ❌ `{col}`")

        if extra_cols:
            with st.expander("ℹ️ Kolom Tambahan (tidak digunakan model)", expanded=False):
                for col in sorted(extra_cols):
                    st.markdown(f"- `{col}`")

else:
    # No file uploaded yet — show empty state
    st_html(render_empty_state(
        "📂",
        "Belum Ada File yang Diupload",
        "Upload file CSV atau XLSX berisi data klinis pasien untuk memulai analisis batch prediction."
    ))

# ─── Footer ──────────────────────────────────────────────────────────────────
st_html(render_footer())
