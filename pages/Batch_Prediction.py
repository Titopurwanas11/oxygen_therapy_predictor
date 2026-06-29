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


st.set_page_config(page_title="Prediksi Massal — OxyPredict", page_icon="assets/favicon-64x64.png", layout="wide")
setup_page("Prediksi Massal — OxyPredict")
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
    div.stDownloadButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-color: #2563EB !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }
    div.stButton > button {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-color: #0F172A !important;
    }
    div.stButton > button:hover {
        background-color: #1E293B !important;
        border-color: #1E293B !important;
    }
</style>
""")

# ─── Header ──────────────────────────────────────────────────────────────────
st_html(render_page_header(
    "",
    "Prediksi Massal Pasien",
    "Sistem Pendukung Keputusan Klinis untuk Analisis Kebutuhan Terapi Oksigen pada Populasi Pasien."
))

# ─── SECTION 1: UPLOAD ───────────────────────────────────────────────────────
st_html("<h3 class=\"section-title-custom\">Unggah Data Pasien</h3>")

col_upload_left, col_upload_right = st.columns([3, 2])

with col_upload_left:
    uploaded_file = st.file_uploader(
        "Pilih file CSV atau XLSX",
        type=["csv", "xlsx"],
        help="Upload file berisi data klinis pasien untuk dianalisis kebutuhan oksigennya.",
    )

with col_upload_right:
    st_html("""
    <div class="cdss-card" style="background-color: #F8FAFC; border-color: #D6E4F0; height: 100%; min-height: 140px; padding: 1.1rem 1.4rem;">
        <h5 style="margin: 0 0 0.5rem 0; color: #1E293B; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Format yang Didukung</h5>
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.6rem;">
            <span style="background-color: #3282B8; color: white; font-size: 0.72rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 5px;">CSV</span>
            <span style="background-color: #14B8A6; color: white; font-size: 0.72rem; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 5px;">Excel (.xlsx)</span>
        </div>
        <div style="font-size: 0.8rem; color: #64748B; line-height: 1.4;">
            <strong>Parameter yang Dibutuhkan:</strong> 44 Fitur Klinis Pasien<br>
            <strong>Validasi Data:</strong> Otomatis oleh Sistem
        </div>
    </div>
    """)

# ─── Show required columns ──────────────────────────────────────────────────
with st.expander("Lihat Daftar Kolom yang Diperlukan (44 fitur)", expanded=False):
    col_a, col_b = st.columns(2)
    half = len(ALL_FEATURES) // 2 + 1
    with col_a:
        for i, feat in enumerate(ALL_FEATURES[:half], 1):
            st.markdown(f"`{i}.` {feat}")
    with col_b:
        for i, feat in enumerate(ALL_FEATURES[half:], half + 1):
            st.markdown(f"`{i}.` {feat}")

# Process file if uploaded
# Process file if uploaded
if uploaded_file is not None:
    # 1. File size check (limit 50MB)
    max_size = 50 * 1024 * 1024
    if uploaded_file.size > max_size:
        from utils.config import show_error_card
        show_error_card(
            "Ukuran Berkas Melebihi Batas Maksimum",
            "Ukuran berkas maksimal yang diperbolehkan adalah 50 MB. Silakan kompres berkas Anda atau bagi menjadi beberapa bagian."
        )
        st.stop()

    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        from utils.config import logger, show_error_card
        logger.error("Failed to read batch file: %s", str(e), exc_info=True)
        show_error_card(
            "Gagal Membaca File",
            "Pastikan file yang diunggah tidak rusak, tidak terkunci oleh aplikasi lain, dan memiliki format .csv atau .xlsx yang valid."
        )
        st.stop()

    # 2. Empty dataset check
    if len(df_raw) == 0:
        from utils.config import show_warning_card
        show_warning_card(
            "Data Pasien Tidak Ditemukan",
            "Berkas yang diunggah kosong. Pastikan berkas Anda berisi baris data pasien."
        )
        st.stop()

    # 3. Duplicate column validation
    has_duplicates = len(df_raw.columns) != len(set(df_raw.columns))
    if has_duplicates:
        duplicate_cols = list(set([col for col in df_raw.columns if list(df_raw.columns).count(col) > 1]))
        from utils.config import show_warning_card
        show_warning_card(
            "Kolom Duplikat Ditemukan",
            "Data Anda mengandung kolom duplikat berikut:<br>• " + "<br>• ".join(duplicate_cols) + "<br><br>Perbaiki berkas Anda dan coba lagi."
        )
        st.stop()

    # 4. Column validation (Missing / Extra columns)
    is_valid, missing_cols, extra_cols = validate_uploaded_file(df_raw)
    if not is_valid:
        from utils.config import show_warning_card
        show_warning_card(
            "Kolom berikut belum tersedia:",
            "• " + "<br>• ".join(missing_cols) + "<br><br>Silakan gunakan template resmi OxyPredict."
        )
        st.stop()

    # 5. Missing values and Categorical options validation
    null_cols = []
    invalid_cat_cols = {}
    from utils.config import NUMERICAL_RANGES, CATEGORICAL_OPTIONS
    for col in df_raw.columns:
        # Check null/missing values
        null_cnt = df_raw[col].isna().sum()
        if null_cnt > 0:
            null_cols.append(f"{col} ({null_cnt} data kosong)")
            
        # Check categorical values validity
        if col in CATEGORICAL_OPTIONS:
            allowed_opts = set(CATEGORICAL_OPTIONS[col])
            unique_vals = set(df_raw[col].dropna().unique())
            # Convert values to string for safe checks
            allowed_opts_str = {str(o) for o in allowed_opts}
            unique_vals_str = {str(v) for v in unique_vals}
            invalid_vals = unique_vals_str - allowed_opts_str
            if invalid_vals:
                invalid_cat_cols[col] = list(invalid_vals)
                
    if null_cols:
        from utils.config import show_warning_card
        show_warning_card(
            "Data Tidak Lengkap Terdeteksi",
            "Data pasien mengandung nilai kosong pada kolom berikut:<br>• " + "<br>• ".join(null_cols) + "<br><br>Lengkapi nilai kosong sebelum memproses prediksi."
        )
        st.stop()
        
    if invalid_cat_cols:
        invalid_desc = []
        for col, vals in invalid_cat_cols.items():
            allowed = ", ".join([str(o) for o in CATEGORICAL_OPTIONS[col]])
            invalid_desc.append(f"Kolom '{col}': Nilai {vals} tidak valid (Pilihan: {allowed})")
        from utils.config import show_warning_card
        show_warning_card(
            "Nilai Kategori Tidak Valid",
            "• " + "<br>• ".join(invalid_desc) + "<br><br>Sesuaikan nilai kategori sesuai ketentuan template OxyPredict."
        )
        st.stop()

    v1, v2, v3, v4, v5 = st.columns(5)
    
    with v1:
        st.metric("👥 Jumlah Pasien", len(df_raw))
    with v2:
        st.metric("Jumlah Kolom", len(df_raw.columns))
    with v3:
        st.metric("Kolom Tidak Ditemukan", len(missing_cols), delta=None, delta_color="inverse")
    with v4:
        st.metric("Kolom Tambahan", len(extra_cols))
    with v5:
        st.metric("Siap Diproses", "Ya" if is_valid else "Tidak")

    st.markdown("")

    if is_valid:
        from utils.config import show_success_card
        show_success_card("Validasi Sukses", "Semua 44 kolom klinis ditemukan. Dataset siap dianalisis.")

        with st.expander("👀 Preview Data (5 baris pertama)", expanded=False):
            st.dataframe(df_raw.head(), use_container_width=True)

        st.markdown("")

        # ─── SECTION 3: JALANKAN PREDIKSI ────────────────────────────────────
        st_html("<h3 class=\"section-title-custom\">Proses Prediksi Massal</h3>")
        
        col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
        with col_btn_c:
            predict_all = st.button(
                "Mulai Prediksi Semua Pasien",
                use_container_width=True,
                type="primary"
            )

        if predict_all or st.session_state.get("batch_predicted", False):
            # Maintain state so prediction results don't vanish on UI interaction
            if predict_all or "batch_results" not in st.session_state:
                with st.spinner("Memproses Model Prediksi Klinis..."):
                    try:
                        from utils.prediction import ModelLoadError
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
                        
                        from utils.config import show_success_card
                        show_success_card(
                            "Prediksi Massal Berhasil Dilakukan",
                            f"Berhasil memproses analisis klinis untuk {len(result_df)} pasien."
                        )
                    except ModelLoadError:
                        from utils.config import show_error_card
                        show_error_card(
                            "Model Prediksi Tidak Dapat Dimuat",
                            "Kemungkinan penyebab:<br>• Berkas model tidak ditemukan.<br>• Model tidak kompatibel.<br>• Versi model tidak sesuai.<br><br>Silakan hubungi administrator sistem."
                        )
                        st.stop()
                    except Exception as pred_err:
                        from utils.config import logger, show_error_card
                        logger.error("Batch prediction failed: %s", str(pred_err), exc_info=True)
                        show_error_card(
                            "Prediksi Gagal Diproses",
                            "Sistem tidak dapat melakukan prediksi karena terjadi gangguan internal. Silakan hubungi administrator sistem."
                        )
                        st.stop()

            # Retrieve results
            result_df = st.session_state.batch_results
            stats = calculate_population_stats(result_df)
            narrative = generate_population_narrative(stats)

            st.markdown("---")

            # ─── SECTION 4: RINGKASAN HASIL ───────────────────────────────────
            st_html("<h3 class=\"section-title-custom\">Ringkasan Hasil Prediksi Populasi</h3>")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Pasien", f"{stats['total_patients']}")
            m2.metric("Membutuhkan Oksigen", f"{stats['need_oxy']}", f"{stats['need_oxy_pct']:.1f}%", delta_color="inverse")
            m3.metric("Tidak Membutuhkan Oksigen", f"{stats['no_oxy']}")
            m4.metric("Rata-rata Probabilitas", f"{stats['avg_probability']:.1f}%")

            m5, m6, m7 = st.columns(3)
            m5.metric("Pasien Risiko Tinggi", f"{stats['high_risk']}")
            m6.metric("Pasien Risiko Sedang", f"{stats['med_risk']}")
            m7.metric("Pasien Risiko Rendah", f"{stats['low_risk']}")

            st.markdown("")

            # ─── SECTION 5: VISUALIZATION ────────────────────────────────────
            st_html("<h3 class=\"section-title-custom\">Grafik Analitik Populasi</h3>")
            
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 18px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700;\">Rasio Kebutuhan Terapi Oksigen</h4>")
                st.plotly_chart(create_pie_chart(result_df), use_container_width=True)
                st_html("</div>")
            with v_col2:
                st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 18px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700;\">Distribusi Tingkat Risiko</h4>")
                st.plotly_chart(create_risk_distribution_chart(result_df), use_container_width=True)
                st_html("</div>")

            v_col3, v_col4 = st.columns(2)
            with v_col3:
                st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 18px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700;\">Histogram Distribusi Probabilitas Prediksi</h4>")
                st.plotly_chart(create_probability_histogram(result_df), use_container_width=True)
                st_html("</div>")
            with v_col4:
                st_html("<div class=\"cdss-card\"><h4 style=\"font-size: 18px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700;\">Rata-rata Probabilitas per Tingkat Risiko</h4>")
                st.plotly_chart(create_avg_prob_per_risk_chart(result_df), use_container_width=True)
                st_html("</div>")

            # ─── SECTION 6: RINGKASAN KLINIS POPULASI ────────────────────────
            st_html("<h3 class=\"section-title-custom\">Interpretasi Klinis Populasi</h3>")
            
            st_html(f"""
            <div class="cdss-card" style="border-left: 6px solid #3282B8;">
                <h4 style="margin: 0 0 0.8rem 0; color: #0F4C75; font-weight: 700; font-size: 18px;">Ringkasan Klinis Populasi</h4>
                <p style="margin: 0; color: #1E293B; font-size: 16px; line-height: 1.7; text-align: justify;">
                    {narrative}
                </p>
            </div>
            """)

            # ─── SECTION 7: TABEL INTERAKTIF & FILTER ────────────────────────
            st_html("<h3 class=\"section-title-custom\">Data Pasien & Filter Pencarian</h3>")
            
            # Filters block
            st_html("<div class=\"cdss-card\" style=\"background-color: #F8FAFC; border-color: #D6E4F0;\"><h5 style=\"margin:0 0 1rem 0; font-size: 13px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px;\">KRITERIA FILTER</h5>")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                filter_pred = st.selectbox("Hasil Prediksi", ["Semua", "Yes", "No"])
            with f2:
                filter_risk = st.selectbox("Tingkat Risiko", ["Semua", "Risiko Rendah", "Risiko Rendah-Sedang", "Risiko Sedang", "Risiko Tinggi", "Risiko Sangat Tinggi"])
            with f3:
                min_prob, max_prob = st.slider("Rentang Probabilitas (%)", 0.0, 100.0, (0.0, 100.0))
            with f4:
                search_query = st.text_input("Cari ID Pasien", "")
            st_html("</div>")

            # Apply filters
            filtered_df = result_df.copy()
            
            # Make sure a Patient ID column exists or generate one for search/indexing
            if "Patient ID" not in filtered_df.columns:
                filtered_df.insert(0, "Patient ID", [f"Patient #{i+1}" for i in range(len(filtered_df))])

            if filter_pred != "Semua":
                filtered_df = filtered_df[filtered_df["Prediction"] == filter_pred]
            if filter_risk != "Semua":
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
                    return ["background-color: #FEF2F2; color: #EF4444;"] * len(row)
                elif pred == "No":
                    return ["background-color: #ECFDF5; color: #22C55E;"] * len(row)
                return [""] * len(row)

            # Style display df
            final_df = filtered_df[display_order + other_cols]
            
            st.dataframe(
                final_df.style.apply(highlight_row, axis=1, subset=["Prediction", "Patient ID"]),
                use_container_width=True,
                height=350
            )

            st.markdown("")

            # ─── SECTION 8: UNDUH LAPORAN ─────────────────────────────────────
            st_html("<h3 class=\"section-title-custom\">Unduh Laporan Prediksi</h3>")
            
            dl1, dl2, dl3 = st.columns(3)

            # Excel download with enriched columns
            excel_ok = False
            try:
                excel_buffer = io.BytesIO()
                final_df.to_excel(excel_buffer, index=False, engine="openpyxl")
                excel_buffer.seek(0)
                excel_ok = True
            except Exception as excel_err:
                from utils.config import logger, show_warning_card
                logger.error("Excel batch export failed: %s", str(excel_err), exc_info=True)
                show_warning_card(
                    "Ekspor Data Gagal",
                    "Gagal mengekspor data ke Excel. Pastikan berkas tidak sedang dibuka oleh aplikasi lain atau terkunci."
                )

            with dl1:
                if excel_ok:
                    st.download_button(
                        label="Unduh File Excel",
                        data=excel_buffer,
                        file_name=f"OxyPredict_Prediksi_Massal_{datetime.date.today()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        on_click=track_excel_download,
                    )

            # CSV download with enriched columns
            csv_buffer = final_df.to_csv(index=False).encode("utf-8")
            with dl2:
                st.download_button(
                    label="Unduh File CSV",
                    data=csv_buffer,
                    file_name=f"OxyPredict_Prediksi_Massal_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    on_click=track_csv_download,
                )

            # PDF Batch Report download
            pdf_report_bytes = None
            try:
                pdf_report_bytes = generate_batch_pdf_report(result_df, stats, narrative)
            except Exception as pdf_err:
                from utils.config import logger, show_error_card
                logger.error("Batch PDF generation failed: %s", str(pdf_err), exc_info=True)
                show_error_card(
                    "Laporan PDF Gagal Dibuat",
                    "Gagal membuat Laporan PDF. Silakan coba kembali beberapa saat lagi."
                )

            with dl3:
                if pdf_report_bytes:
                    st.download_button(
                        label="Unduh Laporan PDF",
                        data=pdf_report_bytes,
                        file_name=f"OxyPredict_Batch_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        on_click=track_pdf_report_generated,
                    )


    else:
        # Columns verification failed
        from utils.config import show_error_card
        show_error_card("Validasi Gagal", f"Dataset tidak lengkap. Ditemukan {len(missing_cols)} kolom yang hilang.")

        with st.expander("Lihat Kolom yang Hilang", expanded=True):
            for col in sorted(missing_cols):
                st.markdown(f"- `{col}`")

        if extra_cols:
            with st.expander("Kolom Tambahan (tidak digunakan model)", expanded=False):
                for col in sorted(extra_cols):
                    st.markdown(f"- `{col}`")

else:
    # No file uploaded yet — show empty state
    st_html(render_empty_state(
        "",
        "Belum Ada Berkas yang Diunggah",
        "Unggah berkas CSV atau XLSX berisi data klinis pasien untuk memulai prediksi massal."
    ))

# ─── Footer ──────────────────────────────────────────────────────────────────
st_html(render_footer())
