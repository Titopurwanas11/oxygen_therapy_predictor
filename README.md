# OxyPredict

Aplikasi ini adalah dashboard Streamlit untuk prediksi kebutuhan terapi oksigen pada pasien anak berdasarkan data klinis.

## Struktur Proyek

- `app.py` - titik masuk utama aplikasi Streamlit.
- `pages/` - halaman-halaman aplikasi seperti `Single_Prediction.py`, `Batch_Prediction.py`, `Monitoring_Prediksi.py`, dan `Clinical_Decision_Guide.py`.
- `utils/` - modul utilitas untuk prediksi, laporan PDF, grafik, konfigurasi gaya, dan analitik sesi.
- `model/` - berisi model terlatih yang digunakan untuk prediksi.
- `assets/` - aset statis seperti ikon dan manifest.
- `.streamlit/config.toml` - pengaturan tema Streamlit.

## Persyaratan

Python 3.12 atau lebih baru.

## Menjalankan Aplikasi

1. Buka terminal di folder proyek:

```powershell
cd "c:\Users\titop\OneDrive\Dokumen\Kuliah\Semester 7\Skripsi\Project\project\oxygen_therapy_predictor"
```

2. Buat virtual environment dan aktifkan:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Perbarui `pip` dan install dependensi:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Jalankan aplikasi:

```powershell
python -m streamlit run app.py
```

5. Buka alamat yang ditampilkan, misalnya `http://localhost:8501`.

## Paket Utama

- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `shap`
- `scikit-learn`
- `reportlab`
- `joblib`
- `openpyxl`

## Catatan

- Pastikan model `best_random_forest.pkl` tersedia di folder `model/`.
- Jika file model belum ada, aplikasi akan gagal memuat model.
