import os
import json

import cv2
import keras
from keras import layers
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE, "contoh_uji")
INT_COLS = ["parameter", "TP", "FP", "FN", "TN", "biaya_relatif"]

st.set_page_config(page_title="Inspeksi Cacat Produk Cor", page_icon="🔍", layout="wide")


def build_cnn(size):
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomBrightness(0.1, value_range=(0, 1)),
        layers.RandomContrast(0.1),
    ], name="augmentasi")

    inp = keras.Input((size, size, 1))
    x = layers.Rescaling(1 / 255)(inp)
    x = augment(x)
    for i, f in enumerate([32, 64, 128], start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalMaxPooling2D(name="gmp")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inp, out, name="cnn_casting")


# arsitektur dibangun ulang lalu bobotnya dimuat, supaya tetap jalan walau versi Keras beda dengan saat training
@st.cache_resource
def load_model(size):
    model = build_cnn(size)
    model.load_weights(os.path.join(BASE, "model_casting.keras"))
    return model


@st.cache_data
def load_metrics():
    with open(os.path.join(BASE, "metrics.json")) as f:
        return json.load(f)


def preprocess(img, size):
    img = ImageOps.exif_transpose(img).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype="float32")[..., None]


def gradcam(model, x, layer="relu3"):
    # pakai logit (sebelum sigmoid), kalau pakai output sigmoid gradiennya ~0 saat p mendekati 1
    dense_out = model.get_layer("output")
    grad_model = keras.Model(model.inputs, [model.get_layer(layer).output, model.get_layer("dropout").output])
    with tf.GradientTape() as tape:
        conv, feat = grad_model(tf.convert_to_tensor(x[None]), training=False)
        logit = tf.matmul(feat, dense_out.kernel)[:, 0] + dense_out.bias[0]
    grads = tape.gradient(logit, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam /= cam.max() + 1e-8
    return cv2.resize(cam, x.shape[:2][::-1])


def overlay(x, cam):
    gray = np.repeat(x.astype("uint8"), 3, axis=-1)
    heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)[..., ::-1]
    return (0.55 * gray + 0.45 * heat).astype("uint8")


def show_table(df):
    fmt = {c: "{:,.0f}" if c in INT_COLS else "{:.4f}" for c in df.columns}
    st.dataframe(df.style.format(fmt), width="stretch")


if not os.path.exists(os.path.join(BASE, "model_casting.keras")):
    st.error("model_casting.keras belum ada. Salin isi folder outputs/ dari notebook ke folder ini.")
    st.stop()

m = load_metrics()
size = m["img_size"]
model = load_model(size)

st.title("🔍 Inspeksi Visual Otomatis Produk Cor")
st.caption("UTS Deep Learning · CNN (±102 ribu parameter) untuk deteksi cacat impeller hasil pengecoran · "
           "dataset Kaggle Casting Product Image Data")

tab_pred, tab_eval = st.tabs(["Uji coba prediksi", "Evaluasi model"])

with tab_pred:
    tau = st.slider("Ambang keputusan τ (TOLAK jika probabilitas cacat ≥ τ)", 0.01, 0.99, float(m["ambang"]), 0.01,
                    help=f"Default {m['ambang']}: biaya kesalahan terkecil di data validasi (10 × cacat lolos + 1 × salah tolak).")
    uploads = st.file_uploader("Upload foto produk (JPG/PNG)", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True)
    images = [(f.name, Image.open(f)) for f in uploads or []]

    if not images and os.path.isdir(SAMPLE_DIR):
        names = sorted(os.listdir(SAMPLE_DIR))
        default = [n for n in names if "def" in n][:1] + [n for n in names if "ok" in n][:1]
        chosen = st.multiselect("Atau pilih contoh dari test set ('def' = cacat, 'ok' = baik)", names, default=default)
        images = [(n, Image.open(os.path.join(SAMPLE_DIR, n))) for n in chosen]

    show_cam = st.checkbox("Tampilkan Grad-CAM", value=True)

    for name, img in images:
        x = preprocess(img, size)
        p = float(model.predict(x[None], verbose=0)[0, 0])
        reject = p >= tau

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 1, 1.2])
            c1.image(x[..., 0].astype("uint8"), caption=name, width="stretch")
            if show_cam:
                # heatmap dikali p: tiap citra dinormalisasi ke nilai maksnya sendiri, jadi tanpa ini produk baik tetap terlihat "panas"
                cap = "Grad-CAM: area pemicu keputusan cacat" if reject else "Grad-CAM: tidak ada area yang kuat mengarah ke cacat"
                c2.image(overlay(x, gradcam(model, x) * p), caption=cap, width="stretch")
            c3.metric("Probabilitas cacat", f"{p:.3f}")
            if reject:
                c3.error(f"**TOLAK** — terindikasi cacat (p ≥ {tau:.2f})")
            else:
                c3.success(f"**LOLOS** — produk baik (p < {tau:.2f})")
            c3.progress(p)

    if not images:
        st.info("Upload foto atau pilih contoh citra.")
    st.caption("Model dilatih dengan foto tampak depan, pencahayaan dan latar seragam. "
               "Foto dengan kondisi lain (mis. kamera HP) bisa menghasilkan prediksi yang salah.")

with tab_eval:
    t = m["test_bersih"]
    st.subheader(f"Test set bersih (τ = {m['ambang']})")
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [("Recall cacat", "recall_cacat"), ("False reject rate", "fpr"),
                                        ("Precision", "precision_cacat"), ("F1-score", "f1"), ("ROC-AUC", "roc_auc")]):
        col.metric(label, f"{t[key]:.4f}")
    st.caption(f"FN (cacat lolos) = {t['FN']}, FP (produk baik ditolak) = {t['FP']}. "
               f"{m['duplikat_dihapus']} citra duplikat train–test sudah dibuang dari test set.")

    st.markdown("**Perbandingan model**")
    show_table(pd.DataFrame(m["perbandingan"]).T[["parameter", "ambang", "recall_cacat", "fpr",
                                                   "precision_cacat", "f1", "roc_auc", "FN", "FP"]])

    st.markdown("**Test bersih vs test asli (dengan duplikat)**")
    show_table(pd.DataFrame({"test bersih": m["test_bersih"], "test asli": m["test_asli"]}).T
               [["recall_cacat", "fpr", "f1", "accuracy", "FN", "FP"]])

    st.markdown("**Uji ketahanan**")
    show_table(pd.DataFrame(m["ketahanan"]).T)

    for fname, cap in [("kurva_training.png", "Kurva training"),
                       ("confusion_roc.png", "Confusion matrix & ROC"),
                       ("gradcam.png", "Contoh Grad-CAM")]:
        if os.path.exists(os.path.join(BASE, fname)):
            st.image(os.path.join(BASE, fname), caption=cap)
