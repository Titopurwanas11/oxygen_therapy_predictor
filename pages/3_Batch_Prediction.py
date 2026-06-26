"""
Batch Prediction Page — Upload CSV/XLSX and predict for all patients.
"""

import streamlit as st
import pandas as pd
import io
from utils.config import ALL_FEATURES, setup_page
from utils.prediction import predict_batch, validate_columns

st.set_page_config(page_title="Batch Prediction — OxyPredict", page_icon="🫁", layout="wide")
setup_page("Batch Prediction — OxyPredict")

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0a2e52 0%, #1a4a7a 50%, #2563eb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(10, 46, 82, 0.18);
">
    <h1 style="
        margin: 0;
        color: white;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.3px;
    ">📊 Batch Prediction</h1>
    <p style="
        margin: 0.3rem 0 0 0;
        color: #93c5fd;
        font-size: 0.95rem;
    ">Upload file CSV atau XLSX untuk memprediksi kebutuhan terapi oksigen secara massal</p>
</div>
""", unsafe_allow_html=True)

# ─── Workflow Steps ──────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.2rem 2rem;
    display: flex;
    justify-content: space-around;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
">
    <div style="text-align: center; padding: 0.5rem;">
        <div style="
            width: 36px; height: 36px;
            background: linear-gradient(135deg, #1a73e8, #1557b0);
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        ">1</div>
        <div style="font-size: 0.78rem; color: #475569; font-weight: 600;">Upload File</div>
    </div>
    <div style="color: #cbd5e1; font-size: 1.2rem;">→</div>
    <div style="text-align: center; padding: 0.5rem;">
        <div style="
            width: 36px; height: 36px;
            background: linear-gradient(135deg, #7c3aed, #6d28d9);
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        ">2</div>
        <div style="font-size: 0.78rem; color: #475569; font-weight: 600;">Validasi Kolom</div>
    </div>
    <div style="color: #cbd5e1; font-size: 1.2rem;">→</div>
    <div style="text-align: center; padding: 0.5rem;">
        <div style="
            width: 36px; height: 36px;
            background: linear-gradient(135deg, #059669, #047857);
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        ">3</div>
        <div style="font-size: 0.78rem; color: #475569; font-weight: 600;">Prediksi</div>
    </div>
    <div style="color: #cbd5e1; font-size: 1.2rem;">→</div>
    <div style="text-align: center; padding: 0.5rem;">
        <div style="
            width: 36px; height: 36px;
            background: linear-gradient(135deg, #ea580c, #c2410c);
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        ">4</div>
        <div style="font-size: 0.78rem; color: #475569; font-weight: 600;">Download Hasil</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── File Upload ─────────────────────────────────────────────────────────────
st.markdown("""
<h3 style="color: #0a2e52; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
    📁 Step 1: Upload File Data Pasien
</h3>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Pilih file CSV atau XLSX",
    type=["csv", "xlsx"],
    help="File harus memiliki kolom yang sesuai dengan fitur model (44 fitur).",
)

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

# ─── Process uploaded file ───────────────────────────────────────────────────
if uploaded_file is not None:
    # Read file
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {str(e)}")
        st.stop()

    st.markdown("")

    # ─── File Info ───────────────────────────────────────────────────────
    st.markdown("""
    <h3 style="color: #0a2e52; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
        📄 Informasi File
    </h3>
    """, unsafe_allow_html=True)

    info1, info2, info3 = st.columns(3)
    info1.metric("📁 Nama File", uploaded_file.name)
    info2.metric("👥 Jumlah Baris", f"{len(df)}")
    info3.metric("📋 Jumlah Kolom", f"{len(df.columns)}")

    st.markdown("")

    # ─── Step 2: Column Validation ───────────────────────────────────────
    st.markdown("""
    <h3 style="color: #0a2e52; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
        ✅ Step 2: Validasi Kolom
    </h3>
    """, unsafe_allow_html=True)

    is_valid, missing_cols, extra_cols = validate_columns(df)

    if is_valid:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid #86efac;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.8rem;
        ">
            <div style="font-size: 1.5rem;">✅</div>
            <div>
                <p style="margin: 0; color: #065f46; font-weight: 700; font-size: 0.95rem;">
                    Validasi Berhasil
                </p>
                <p style="margin: 0.2rem 0 0 0; color: #475569; font-size: 0.82rem;">
                    Semua 44 kolom yang diperlukan ditemukan dalam file.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Preview data
        with st.expander("👀 Preview Data (5 baris pertama)", expanded=True):
            st.dataframe(df.head(), use_container_width=True)

        st.markdown("")

        # ─── Step 3: Run Prediction ──────────────────────────────────────
        st.markdown("""
        <h3 style="color: #0a2e52; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
            🔮 Step 3: Jalankan Prediksi
        </h3>
        """, unsafe_allow_html=True)

        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            run_prediction = st.button(
                "🚀  Prediksi Semua Pasien",
                use_container_width=True,
                type="primary",
            )

        if run_prediction:
            with st.spinner("⏳ Memproses prediksi untuk semua pasien..."):
                try:
                    result_df = predict_batch(df)

                    st.markdown("")

                    # Summary
                    total = len(result_df)
                    need_oxy = (result_df["Prediction"] == "Yes").sum()
                    no_oxy = (result_df["Prediction"] == "No").sum()
                    avg_prob = result_df["Probability"].mean()

                    st.markdown("""
                    <h3 style="color: #0a2e52; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
                        📈 Ringkasan Hasil Prediksi
                    </h3>
                    """, unsafe_allow_html=True)

                    s1, s2, s3, s4 = st.columns(4)

                    with s1:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #1a73e8, #1557b0);
                            border-radius: 14px; padding: 1.3rem; text-align: center;
                            box-shadow: 0 4px 16px rgba(26,115,232,0.2);
                        ">
                            <div style="font-size: 0.68rem; color: #bfdbfe; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.4rem;">Total Pasien</div>
                            <div style="font-size: 2rem; font-weight: 800; color: white;">{total}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with s2:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #dc2626, #b91c1c);
                            border-radius: 14px; padding: 1.3rem; text-align: center;
                            box-shadow: 0 4px 16px rgba(220,38,38,0.2);
                        ">
                            <div style="font-size: 0.68rem; color: #fecaca; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.4rem;">Butuh O₂</div>
                            <div style="font-size: 2rem; font-weight: 800; color: white;">{need_oxy}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with s3:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #059669, #047857);
                            border-radius: 14px; padding: 1.3rem; text-align: center;
                            box-shadow: 0 4px 16px rgba(5,150,105,0.2);
                        ">
                            <div style="font-size: 0.68rem; color: #a7f3d0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.4rem;">Tidak Butuh O₂</div>
                            <div style="font-size: 2rem; font-weight: 800; color: white;">{no_oxy}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with s4:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #7c3aed, #6d28d9);
                            border-radius: 14px; padding: 1.3rem; text-align: center;
                            box-shadow: 0 4px 16px rgba(124,58,237,0.2);
                        ">
                            <div style="font-size: 0.68rem; color: #ddd6fe; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.4rem;">Rata-rata Prob.</div>
                            <div style="font-size: 2rem; font-weight: 800; color: white;">{avg_prob:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("")

                    # Result table
                    st.markdown("""
                    <h3 style="color: #0a2e52; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
                        📋 Tabel Hasil Prediksi
                    </h3>
                    """, unsafe_allow_html=True)

                    # Show only key columns + prediction
                    display_cols = ["Prediction", "Probability"]
                    # Add some important features at the start
                    key_features = [
                        "Age (months)", "Gender", "Weight (Kg)",
                        "Oxygen saturation (SaO2) at admission",
                        "Respiratory rate", "Heart rate",
                    ]
                    display_order = [c for c in key_features if c in result_df.columns] + display_cols
                    other_cols = [c for c in result_df.columns if c not in display_order]

                    # Style the dataframe
                    def highlight_prediction(val):
                        if val == "Yes":
                            return "background-color: #fee2e2; color: #dc2626; font-weight: 700;"
                        elif val == "No":
                            return "background-color: #dcfce7; color: #059669; font-weight: 700;"
                        return ""

                    styled_df = result_df[display_order + other_cols].style.applymap(
                        highlight_prediction, subset=["Prediction"]
                    )

                    st.dataframe(styled_df, use_container_width=True, height=400)

                    st.markdown("")

                    # ─── Step 4: Download ────────────────────────────────────
                    st.markdown("""
                    <h3 style="color: #0a2e52; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
                        💾 Step 4: Download Hasil
                    </h3>
                    """, unsafe_allow_html=True)

                    dl1, dl2, dl3 = st.columns([1, 1, 2])

                    # CSV download
                    csv_buffer = result_df.to_csv(index=False).encode("utf-8")
                    with dl1:
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_buffer,
                            file_name="prediction_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                    # Excel download
                    excel_buffer = io.BytesIO()
                    result_df.to_excel(excel_buffer, index=False, engine="openpyxl")
                    excel_buffer.seek(0)
                    with dl2:
                        st.download_button(
                            label="📥 Download Excel",
                            data=excel_buffer,
                            file_name="prediction_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat melakukan prediksi: {str(e)}")

    else:
        # Validation failed
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 1px solid #fca5a5;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            display: flex;
            align-items: flex-start;
            gap: 0.8rem;
        ">
            <div style="font-size: 1.5rem; flex-shrink: 0;">❌</div>
            <div>
                <p style="margin: 0; color: #991b1b; font-weight: 700; font-size: 0.95rem;">
                    Validasi Gagal
                </p>
                <p style="margin: 0.2rem 0 0 0; color: #dc2626; font-size: 0.82rem;">
                    Terdapat <strong>{len(missing_cols)}</strong> kolom yang tidak ditemukan dalam file.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        with st.expander("🔍 Lihat Kolom yang Hilang", expanded=True):
            for col in sorted(missing_cols):
                st.markdown(f"- ❌ `{col}`")

        if extra_cols:
            with st.expander("ℹ️ Kolom Tambahan (tidak digunakan model)", expanded=False):
                for col in sorted(extra_cols):
                    st.markdown(f"- `{col}`")
