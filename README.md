# Aplikasi Uji Coba: Inspeksi Cacat Produk Cor

Aplikasi Streamlit untuk menguji model CNN 

## Isi folder

| File | Asal |
| --- | --- |
| `app.py` | Kode aplikasi |
| `requirements.txt` | Daftar library |
| `model_casting.keras` | Model terlatih 
| `metrics.json` | Metrik evaluasi 
| `contoh_uji/` | Contoh citra test 
| `kurva_training.png`, `confusion_roc.png`, `gradcam.png` | Grafik evaluasi 


## Deploy ke Streamlit Community Cloud untuk menjalankan model

1. Unggah isi folder ini ke repositori GitHub.
2. Buka https://share.streamlit.io, pilih **Create app**, lalu pilih repositori dan file `app.py`.
3. Pada **Advanced settings**, pilih Python 3.12.
4. Klik **Deploy**.

## Cara memakai

- Tab **Uji coba prediksi**: seret foto produk ke kotak upload, atau pilih citra contoh. Aplikasi menampilkan probabilitas cacat, keputusan TOLAK/LOLOS, dan heatmap Grad-CAM. Slider τ dipakai untuk mengubah ambang keputusan.
- Tab **Evaluasi model**: menampilkan metrik pada test set bersih, perbandingan dengan baseline, efek kebocoran data, dan uji ketahanan.
