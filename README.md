# Aplikasi Uji Coba: Inspeksi Cacat Produk Cor (UTS Deep Learning)

Aplikasi Streamlit untuk menguji model CNN hasil notebook `UTS_DeepLearning_Casting.ipynb`.

## Isi folder

| File | Asal |
| --- | --- |
| `app.py` | Kode aplikasi |
| `requirements.txt` | Daftar library |
| `model_casting.keras` | Model terlatih (folder `outputs/` dari notebook) |
| `metrics.json` | Metrik evaluasi (folder `outputs/` dari notebook) |
| `contoh_uji/` | Contoh citra test (folder `outputs/` dari notebook) |
| `kurva_training.png`, `confusion_roc.png`, `gradcam.png` | Grafik evaluasi (folder `outputs/` dari notebook) |

Jika notebook dijalankan ulang di Colab, ganti file-file di atas dengan isi `outputs.zip` yang baru.

## Menjalankan di laptop (macOS/Windows/Linux)

```bash
cd "UTS_Deep Learning/streamlit_app"
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Browser akan terbuka di `http://localhost:8501`.

## Deploy ke Streamlit Community Cloud (opsional, link publik)

1. Unggah isi folder ini ke repositori GitHub.
2. Buka https://share.streamlit.io, pilih **Create app**, lalu pilih repositori dan file `app.py`.
3. Pada **Advanced settings**, pilih Python 3.12.
4. Klik **Deploy**.

## Cara memakai

- Tab **Uji coba prediksi**: seret foto produk ke kotak upload, atau pilih citra contoh. Aplikasi menampilkan probabilitas cacat, keputusan TOLAK/LOLOS, dan heatmap Grad-CAM. Slider τ dipakai untuk mengubah ambang keputusan.
- Tab **Evaluasi model**: menampilkan metrik pada test set bersih, perbandingan dengan baseline, efek kebocoran data, dan uji ketahanan.
