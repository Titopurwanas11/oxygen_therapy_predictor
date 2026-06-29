"""
Clinical Decision Guide Page — Integrated Clinical and User Guide for OxyPredict.
"""

import streamlit as st
from utils.config import (
    setup_page,
    render_page_header,
    render_section_divider,
    render_footer,
)

st.set_page_config(page_title="Panduan Keputusan Klinis — OxyPredict", page_icon="assets/favicon-64x64.png", layout="wide")
setup_page("Panduan Keputusan Klinis — OxyPredict")

# Helper function to render HTML safely
def st_html(html_str):
    cleaned_lines = [line.lstrip() for line in html_str.splitlines()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)

# ─── SECTION 1: HEADER ───────────────────────────────────────────────────────
st_html(render_page_header(
    "📘",
    "Panduan Keputusan Klinis",
    "Panduan lengkap untuk implementasi klinis, interpretasi hasil, dan penggunaan OxyPredict."
))

# ─── SECTION 1: Ikhtisar Dukungan Keputusan Klinis ──────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 0.8rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">local_hospital</span>
        1. Ikhtisar Dukungan Keputusan Klinis
    </h3>
    <p style="font-size: 16px; color: #1E293B; line-height: 1.7; margin-bottom: 1rem;">
        <strong>OxyPredict</strong> adalah Sistem Pendukung Keputusan Klinis (CDSS) yang dirancang untuk membantu tenaga kesehatan menilai apakah pasien pediatrik dengan Infeksi Saluran Pernapasan Akut (ISPA) dan Pneumonia memerlukan terapi oksigen tambahan.
    </p>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.2rem;">
        <div style="background: #F8FAFC; padding: 1rem; border-radius: 10px; border-left: 3px solid #3282B8;">
            <strong style="font-size: 15px; color: #1E293B; display: block; margin-bottom: 0.4rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle; color: #2563EB;">target</span>
                Tujuan Sistem
            </strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.5; display: block;">
                Memberikan opini kedua berbasis bukti mengenai kebutuhan dukungan oksigen, membantu optimalisasi sumber daya.
            </span>
        </div>
        <div style="background: #F8FAFC; padding: 1rem; border-radius: 10px; border-left: 3px solid #3282B8;">
            <strong style="font-size: 15px; color: #1E293B; display: block; margin-bottom: 0.4rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle; color: #2563EB;">group</span>
                Pengguna yang Ditujukan
            </strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.5; display: block;">
                Dokter, perawat, dokter anak, dan staf triase klinis yang menangani kasus pernapasan akut pada anak.
            </span>
        </div>
        <div style="background: #F8FAFC; padding: 1rem; border-radius: 10px; border-left: 3px solid #3282B8;">
            <strong style="font-size: 15px; color: #1E293B; display: block; margin-bottom: 0.4rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle; color: #2563EB;">schedule</span>
                Kapan Digunakan
            </strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.5; display: block;">
                Digunakan selama pemeriksaan klinis awal atau triase di unit gawat darurat dan bangsal anak.
            </span>
        </div>
    </div>
    <div style="background: #F8FAFC; padding: 1.2rem; border-radius: 10px; text-align: center;">
        <strong style="font-size: 13px; color: #64748B; display: block; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;">Alur Kerja Dukungan Keputusan Klinis</strong>
        <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 0.5rem; font-size: 14px; font-weight: 600;">
            <span style="background: white; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #D8E2EC; color: #1E293B;">Data Klinis Pasien</span>
            <span style="color: #64748B; font-size: 18px;">→</span>
            <span style="background: white; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #D8E2EC; color: #1E293B;">Prediksi AI</span>
            <span style="color: #64748B; font-size: 18px;">→</span>
            <span style="background: white; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #D8E2EC; color: #1E293B;">Penilaian Risiko</span>
            <span style="color: #64748B; font-size: 18px;">→</span>
            <span style="background: white; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #D8E2EC; color: #1E293B;">Kontribusi Fitur SHAP</span>
            <span style="color: #64748B; font-size: 18px;">→</span>
            <span style="background: #DBEAFE; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #93C5FD; color: #1D4ED8; font-weight: 700;">Keputusan Klinis</span>
        </div>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 2: How to Use OxyPredict ───────────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">settings</span>
        2. Cara Menggunakan OxyPredict
    </h3>
    <div style="position: relative; padding-left: 1.5rem; border-left: 2px solid #D8E2EC;">
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">1</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Navigasi Dashboard</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Periksa ikhtisar operasional real-time, distribusi pasien, rata-rata keyakinan model, tren risiko, dan catatan terbaru.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">2</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Pengajuan Prediksi Tunggal</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Buka halaman 'Single Prediction', masukkan data klinis pasien individu (usia, tanda vital, gejala fisik), lalu klik prediksi.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">3</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Interpretasi Prediksi & Penilaian Risiko</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Tinjau prediksi diagnostik, persentase probabilitas keyakinan, label level risiko, dan baca <strong>Ringkasan Klinis AI</strong> yang dinamis.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">4</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Telaah Penjelasan SHAP</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Lihat grafik kontribusi fitur SHAP untuk mengetahui secara tepat faktor fisiologis mana (mis. saturasi oksigen, retraksi dada, dll.) yang mempengaruhi keputusan model.</span>
        </div>
        <div style="margin-bottom: 1.2rem; position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">5</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Unduh Laporan Diagnostik</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Ekspor laporan PDF klinis yang memuat hasil prediksi, evaluasi risiko, rekomendasi klinis, dan visualisasi SHAP untuk berkas pasien.</span>
        </div>
        <div style="position: relative;">
            <div style="position: absolute; left: -2.1rem; top: 0.1rem; background: #3282B8; color: white; width: 1.2rem; height: 1.2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700;">6</div>
            <strong style="font-size: 16px; color: #1E293B; display: block;">Lakukan Prediksi Batch</strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.6;">Unggah berkas terstruktur (Excel atau CSV), jalankan prediksi model untuk semua pasien sekaligus, tinjau hasil, dan unduh laporan terpadu.</span>
        </div>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 3: Memahami Hasil Prediksi ─────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">analytics</span>
        3. Memahami Hasil Prediksi
    </h3>
    <p style="font-size: 14px; color: #64748B; margin-bottom: 1rem;">
        Setiap prediksi yang dihasilkan OxyPredict mencakup empat variabel klinis utama:
    </p>
    <table style="width: 100%; border-collapse: collapse; font-size: 14px; border: 1px solid #D6E4F0; border-radius: 16px; overflow: hidden;">
        <thead>
            <tr style="background-color: #0F4C75; border-bottom: 2px solid #3282B8;">
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Variabel</th>
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Makna</th>
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Aplikasi Klinis</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #EAF2F8;">
                <td style="padding: 0.7rem 1rem; font-weight: 600; color: #1E293B;">Prediksi</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Klasifikasi biner model: "Ya" (Memerlukan Oksigen) atau "Tidak" (Tidak Memerlukan Oksigen).</td>
                <td style="padding: 0.7rem 1rem; color: #64748B;">Menjadi indikasi klinis utama untuk prioritas triase.</td>
            </tr>
            <tr style="border-bottom: 1px solid #EAF2F8; background-color: #F8FAFC;">
                <td style="padding: 0.7rem 1rem; font-weight: 600; color: #1E293B;">Probabilitas</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Probabilitas internal model bahwa pasien berada dalam kelompok yang memerlukan oksigen (0%–100%).</td>
                <td style="padding: 0.7rem 1rem; color: #64748B;">Membantu menilai tingkat keparahan pasien (semakin tinggi probabilitas, semakin besar kemungkinan distres akut).</td>
            </tr>
            <tr style="border-bottom: 1px solid #EAF2F8;">
                <td style="padding: 0.7rem 1rem; font-weight: 600; color: #1E293B;">Level Keyakinan</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Keyakinan model berdasarkan jarak dari ambang keputusan.</td>
                <td style="padding: 0.7rem 1rem; color: #64748B;">Keyakinan tinggi menunjukkan pola fitur yang kuat; keyakinan rendah menunjukkan kasus ambang batas.</td>
            </tr>
            <tr>
                <td style="padding: 0.7rem 1rem; font-weight: 600; color: #1E293B;">Level Risiko</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Kategori risiko klinis yang diturunkan langsung dari probabilitas prediksi.</td>
                <td style="padding: 0.7rem 1rem; color: #64748B;">Mengarah pada jalur tindakan klinis yang direkomendasikan (misalnya evaluasi segera vs. observasi).</td>
            </tr>
        </tbody>
    </table>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 4: Interpretasi Level Risiko ───────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #EF4444;">warning</span>
        4. Interpretasi Level Risiko
    </h3>
    <p style="font-size: 14px; color: #64748B; margin-bottom: 1rem;">
        Level risiko memprioritaskan pemantauan pasien dan menentukan jalur tindakan:
    </p>
    <table style="width: 100%; border-collapse: collapse; font-size: 14px; border: 1px solid #D6E4F0; border-radius: 16px; overflow: hidden;">
        <thead>
            <tr style="background-color: #0F4C75; border-bottom: 2px solid #3282B8;">
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Kategori Risiko</th>
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Rentang Probabilitas</th>
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Status Pasien</th>
                <th style="padding: 0.75rem 1rem; text-align: left; color: #FFFFFF; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 0.3px;">Rekomendasi Klinis</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #EAF2F8;">
                <td style="padding: 0.7rem 1rem; font-weight: 600;"><span class="cdss-badge cdss-badge-success">Risiko Rendah</span></td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">&lt; 50%</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Stabil secara fisiologis; indikasi gagal napas berat rendah.</td>
                <td style="padding: 0.7rem 1rem; color: #64748B; font-weight: 600;">Observasi rutin direkomendasikan.</td>
            </tr>
            <tr style="border-bottom: 1px solid #EAF2F8; background-color: #F8FAFC;">
                <td style="padding: 0.7rem 1rem; font-weight: 600;"><span class="cdss-badge cdss-badge-warning">Risiko Sedang</span></td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">50% – 70%</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Distres pernapasan ringan; terdapat indikator ambang batas.</td>
                <td style="padding: 0.7rem 1rem; color: #64748B; font-weight: 600;">Monitoring ketat dan evaluasi ulang.</td>
            </tr>
            <tr>
                <td style="padding: 0.7rem 1rem; font-weight: 600;"><span class="cdss-badge cdss-badge-danger">Risiko Tinggi</span></td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">&gt; 70%</td>
                <td style="padding: 0.7rem 1rem; color: #1E293B;">Distres pernapasan berat; indikasi kritis hipoksia.</td>
                <td style="padding: 0.7rem 1rem; color: #EF4444; font-weight: 600;">Evaluasi klinis segera dan penilaian dukungan oksigen.</td>
            </tr>
        </tbody>
    </table>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 5: Confidence Level Interpretation ─────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">speed</span>
        5. Level Keyakinan Model
    </h3>
    <p style="font-size: 16px; color: #1E293B; line-height: 1.7; margin-bottom: 1.2rem;">
        Level keyakinan menunjukkan seberapa kuat fitur cocok dengan pola dari kasus pasien sebelumnya.
    </p>
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 0.3rem;">
            <strong style="color: #15803D;">Keyakinan Sangat Tinggi (mis. 95%)</strong>
            <span style="color: #64748B;">Parameter klinis sangat jelas; jalur diagnostik terdefinisi.</span>
        </div>
        <div style="background: #E2F0F9; height: 8px; border-radius: 6px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #15803D, #22C55E); width: 95%; height: 100%; border-radius: 6px;"></div>
        </div>
    </div>
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 0.3rem;">
            <strong style="color: #B45309;">Keyakinan Sedang (mis. 75%)</strong>
            <span style="color: #64748B;">Beberapa fitur bertentangan; pasien mendekati ambang keputusan.</span>
        </div>
        <div style="background: #E2F0F9; height: 8px; border-radius: 6px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #B45309, #F59E0B); width: 75%; height: 100%; border-radius: 6px;"></div>
        </div>
    </div>
    <div>
        <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 0.3rem;">
            <strong style="color: #B91C1C;">Keyakinan Rendah (mis. 55%)</strong>
            <span style="color: #64748B;">Variasi fisiologis signifikan; interpretasikan hasil dengan kehati-hatian tinggi.</span>
        </div>
        <div style="background: #E2F0F9; height: 8px; border-radius: 6px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #B91C1C, #EF4444); width: 55%; height: 100%; border-radius: 6px;"></div>
        </div>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 6: Memahami Penjelasan SHAP ──────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">biotech</span>
        6. Memahami Penjelasan SHAP
    </h3>
    <p style="font-size: 16px; color: #1E293B; line-height: 1.7; margin-bottom: 1rem;">
        <strong>SHAP (SHapley Additive exPlanations)</strong> adalah kerangka kerja AI yang dapat dijelaskan, yang memecah prediksi dengan menunjukkan kontribusi setiap parameter pasien:
    </p>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.2rem;">
        <div style="background: #FEF2F2; padding: 1rem; border-radius: 10px; border-left: 3px solid #EF4444;">
            <strong style="color: #EF4444; font-size: 15px; display: block; margin-bottom: 0.4rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle;">trending_up</span>
                Positive Contributions (Increases Risk)
            </strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.5; display: block;">
                Factors with red SHAP bars push the patient's likelihood <strong>toward</strong> requiring oxygen therapy (e.g., extremely low SpO2, severe chest indrawing, tachypnea).
            </span>
        </div>
        <div style="background: #ECFDF5; padding: 1rem; border-radius: 10px; border-left: 3px solid #22C55E;">
            <strong style="color: #15803D; font-size: 15px; display: block; margin-bottom: 0.4rem;">
                <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle;">trending_down</span>
                Negative Contributions (Decreases Risk)
            </strong>
            <span style="font-size: 14px; color: #64748B; line-height: 1.5; display: block;">
                Faktor dengan batang SHAP biru mengurangi kemungkinan pasien memerlukan terapi oksigen, menunjukkan parameter yang stabil (mis. SpO2 normal, denyut jantung normal, tidak ada retractio dada).
            </span>
        </div>
    </div>
    <div style="background: #F8FAFC; padding: 1rem; border-radius: 10px;">
        <strong style="font-size: 15px; color: #1E293B; display: block; margin-bottom: 0.4rem;">
            <span class="material-symbols-outlined" style="font-size: 18px; vertical-align: middle; color: #3282B8;">description</span>
            Narasi Ringkasan Klinis AI
        </strong>
        <span style="font-size: 14px; color: #64748B; line-height: 1.6; display: block;">
            The diagnostic interface generates a clinical summary based directly on the leading SHAP features. This allows medical practitioners to quickly review the primary physiological drivers behind the assessment.
        </span>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 7: Panduan Prediksi Batch ───────────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">folder_open</span>
        7. Panduan Prediksi Batch
    </h3>
    <p style="font-size: 16px; color: #1E293B; line-height: 1.7; margin-bottom: 1rem;">
        Halaman Prediksi Batch memungkinkan pemrosesan dataset multi-pasien secara bersamaan:
    </p>
    <ul style="font-size: 15px; color: #64748B; margin: 0 0 1rem 1.2rem; padding: 0; line-height: 1.7;">
        <li style="margin-bottom: 0.5rem;"><strong style="color: #1E293B;">Unggah Berkas:</strong> Unggah file Excel (.xlsx) atau CSV (.csv) yang berisi tanda vital pasien pediatrik.</li>
        <li style="margin-bottom: 0.5rem;"><strong style="color: #1E293B;">Validasi Kolom:</strong> Sistem otomatis memverifikasi bahwa semua 44 fitur klinis yang diperlukan model Random Forest hadir.</li>
        <li style="margin-bottom: 0.5rem;"><strong style="color: #1E293B;">Jalankan Batch Diagnostik:</strong> Klik jalankan untuk menghitung prediksi, persentase keyakinan, dan level risiko klinis untuk seluruh kohor.</li>
        <li style="margin-bottom: 0.5rem;"><strong style="color: #1E293B;">Ringkasan Kohor:</strong> Lihat statistik distribusi, rata-rata variabel, dan risiko kelompok.</li>
        <li><strong style="color: #1E293B;">Ekspor Terpadu:</strong> Unduh hasil diagnostik kohor lengkap dalam format Excel atau buat laporan pasien.</li>
    </ul>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 8: Rekomendasi Klinis ─────────────────────────────────────
st_html("""
<div class="cdss-card">
    <h3 style="font-size: 22px; color: #1E293B; margin: 0 0 1rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #15803D;">lightbulb</span>
        8. Rekomendasi Klinis
    </h3>
    <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 1.2rem;">
        <ul style="list-style-type: none; padding: 0; margin: 0; font-size: 15px; color: #15803D; line-height: 1.7;">
            <li style="margin-bottom: 0.6rem;">✓ <strong>Dukungan Klinis Saja:</strong> Gunakan hasil prediksi OxyPredict sebagai alat referensi pendukung bersama pedoman pediatrik standar.</li>
            <li style="margin-bottom: 0.6rem;">✓ <strong>Evaluasi Komprehensif:</strong> Selalu gabungkan keluaran model dengan pemeriksaan klinis di samping tempat tidur, penilaian fisik, dan riwayat medis pasien.</li>
            <li style="margin-bottom: 0.6rem;">✓ <strong>Pendekatan Berpusat pada Pasien:</strong> Evaluasi pasien secara holistik; jangan buat keputusan klinis penting hanya berdasarkan skor probabilitas numerik.</li>
            <li>✓ <strong>Prioritas Faktor Risiko:</strong> Tinjau secara seksama faktor fisiologis positif utama (meningkatkan risiko) yang diidentifikasi oleh grafik SHAP.</li>
        </ul>
    </div>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 9: Batasan Sistem ──────────────────────────────────────────
st_html("""
<div class="cdss-card" style="border-left: 5px solid #F59E0B;">
    <h3 style="font-size: 22px; color: #F59E0B; margin: 0 0 0.8rem 0; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 24px; color: #F59E0B;">gpp_maybe</span>
        9. Batasan Sistem & Penafian
    </h3>
    <ul style="font-size: 15px; color: #B45309; line-height: 1.7; margin: 0 0 0 1.2rem; padding: 0;">
        <li style="margin-bottom: 0.5rem;"><strong>Ruang Lingkup Pengembangan Model:</strong> Mesin prediktif dikembangkan dan divalidasi menggunakan data 801 kasus pernapasan akut pediatrik.</li>
        <li style="margin-bottom: 0.5rem;"><strong>Limitasi Populasi:</strong> Validitas klinis dikalibrasi untuk parameter demografis pediatrik spesifik yang didefinisikan dalam studi klinis.</li>
        <li style="margin-bottom: 0.5rem;"><strong>Catatan CDSS:</strong> OxyPredict tidak menggantikan, tidak meniadakan, dan tidak melampaui penilaian klinis profesional dari praktisi medis berlisensi.</li>
        <li><strong>Prioritas Bedside:</strong> Keputusan diagnostik akhir dan jalur perawatan klinis harus selalu mengutamakan gejala di samping tempat tidur dan protokol medis yang sudah mapan.</li>
    </ul>
</div>
""")

st_html(render_section_divider())

# ─── SECTION 10: Pertanyaan yang Sering Diajukan ──────────────────────────────────
st_html("""
<h3 class="section-title-custom" style="display: flex; align-items: center; gap: 0.5rem;">
    <span class="material-symbols-outlined" style="font-size: 24px; color: #3282B8;">help</span>
    10. Pertanyaan yang Sering Diajukan
</h3>
""")

with st.expander("Apa arti probabilitas prediksi?"):
    st.write(
        "Probabilitas prediksi adalah skor internal model yang mewakili peluang statistik bahwa "
        "pasien termasuk dalam kelompok yang memerlukan terapi oksigen tambahan. Skor lebih tinggi berarti kemungkinan lebih besar."
    )

with st.expander("Apa perbedaan antara level keyakinan dan level risiko?"):
    st.write(
        "Probabilitas prediksi menentukan Level Risiko (seberapa parah distresnya). Level Keyakinan menunjukkan seberapa "
        "kuat fitur data pasien cocok dengan pola yang dipelajari selama pelatihan (seberapa yakin model dalam keputusannya)."
    )

with st.expander("Mengapa penjelasan SHAP penting bagi klinisi?"):
    st.write(
        "SHAP membuat model pembelajaran mesin menjadi transparan. Alih-alih prediksi kotak hitam, SHAP menunjukkan secara tepat "
        "seberapa besar bobot setiap gejala (seperti laju napas atau retraksi dada) dalam mempengaruhi keputusan."
    )

with st.expander("Mengapa prediksi berbeda untuk pasien dengan tingkat SpO2 serupa?"):
    st.write(
        "OxyPredict memproses 44 fitur fisiologis secara bersamaan. Variabel tambahan seperti retraksi dada, "
        "denyut jantung, usia, dan suhu berinteraksi untuk membentuk gambaran holistik, sehingga dapat mengubah output."
    )

with st.expander("Dapatkah model menghasilkan prediksi yang salah?"):
    st.write(
        "Ya, seperti semua sistem statistik, kesalahan prediksi dapat terjadi. Itulah sebabnya CDSS diklasifikasikan sebagai alat "
        "opini kedua dan harus selalu dipasangkan dengan penilaian dokter."
    )

st.markdown("")

# ─── Footer ──────────────────────────────────────────────────────────────────
st_html(render_footer())
