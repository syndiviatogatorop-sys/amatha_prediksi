import os
import joblib
import requests
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Pola Nasabah",
    page_icon="📊",
    layout="wide"
)

MODEL_ID = "1rVbvV7R-aHT8ScnuV0QRWwegwma-XZ5h"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "model.joblib"
FEATURES_PATH = "features.joblib"  # pastikan file ini ada di repo yang sama

# =========================
# STYLE
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f8f9fa; }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        color: white;
        margin-bottom: 1.5rem;
    }

    .card {
        padding: 1.25rem 1.25rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    .hint {
        color: #475569;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
@st.cache_resource
def load_model():
    # model diunduh dari Google Drive (public link)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model..."):
            r = requests.get(MODEL_URL, timeout=120)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_features():
    # features.joblib harus ada di repo / folder yang sama saat deploy
    if not os.path.exists(FEATURES_PATH):
        st.error(f"File `{FEATURES_PATH}` tidak ditemukan. Upload file tersebut ke repository GitHub.")
        st.stop()
    return joblib.load(FEATURES_PATH)

def group_features(feats):
    """
    feats = daftar kolom dummy, format umumnya: 'NamaKolom_kategori'
    supaya UI rapi, kita grup berdasarkan prefix sebelum underscore pertama.
    """
    groups = {}
    for f in feats:
        key = f.split("_")[0] if "_" in f else "Lainnya"
        groups.setdefault(key, []).append(f)
    # rapihin urutan
    return {k: sorted(v) for k, v in sorted(groups.items(), key=lambda x: x[0].lower())}

def get_expected_features(model, fallback_features):
    # kalau model menyimpan fitur saat training
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(fallback_features)

def align_input(X: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    # samakan kolom input dengan kolom training
    X = X.reindex(columns=expected_features, fill_value=0)
    return X.fillna(0).astype(int)

# =========================
# LOAD ARTIFACTS
# =========================
model = load_model()
features = load_features()

# NOTE:
# - UI form akan mengikuti features.joblib (biar opsi lengkap)
# - input ke model akan di-align ke fitur yang diminta model (biar tidak error mismatch)
form_features = list(features)
expected_features = get_expected_features(model, features)

groups = group_features(form_features)
group_names = list(groups.keys())

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("📌 Menu Utama")
    st.info("Isi form untuk memprediksi status kredit nasabah.")
    st.divider()
    st.markdown("### 🛠️ Status Sistem")
    st.success("Model: Siap")
    st.success("Fitur: Siap")

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <h1 style="margin:0; color:white;">📊 Prediksi Pola Nasabah</h1>
    <p style="opacity:0.9; margin-top:.35rem;">
        Prediksi status kredit berdasarkan karakteristik dan data keuangan nasabah.
    </p>
</div>
""", unsafe_allow_html=True)

col_form, col_res = st.columns([1.45, 1])

# =========================
# FORM
# =========================
with col_form:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Form Data Nasabah")
    st.markdown("<p class='hint'>Pilih 1 nilai pada setiap bagian. Jika tidak ada yang sesuai, pilih <b>Pilih...</b>.</p>", unsafe_allow_html=True)

    input_data = {}

    with st.form("form_prediksi"):
        c1, c2 = st.columns(2)
        mid = max(1, len(group_names) // 2)

        def render_form(group_list, container):
            for g in group_list:
                with container.expander(f"📍 {g}", expanded=True):
                    options = ["Pilih..."] + groups[g]
                    choice = st.selectbox(
                        f"Pilih {g}",
                        options,
                        key=f"sb_{g}",
                        label_visibility="collapsed"
                    )
                    # set 0 semua dulu untuk grup ini
                    for feat in groups[g]:
                        input_data[feat] = 0
                    # yang dipilih jadi 1
                    if choice != "Pilih...":
                        input_data[choice] = 1

        render_form(group_names[:mid], c1)
        render_form(group_names[mid:], c2)

        submitted = st.form_submit_button(
            "🔍 Jalankan Prediksi",
            use_container_width=True,
            type="primary"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# RESULT
# =========================
with col_res:
    st.subheader("📊 Hasil Prediksi")

    if not submitted:
        st.info("Lengkapi form di samping, lalu klik **Jalankan Prediksi**.")

    if submitted:
        # input mentah dari form
        X_raw = pd.DataFrame([input_data])

        # align ke fitur yang diminta model
        X = align_input(X_raw, expected_features)

        # prediksi: 0=LANCAR, 1=MACET
        pred = int(model.predict(X)[0])

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if pred == 0:
            st.success("### ✅ HASIL: LANCAR")
            st.write("Nasabah diprediksi aman untuk diberikan pinjaman.")
        else:
            st.error("### ❌ HASIL: MACET")
            st.write("Peringatan: risiko gagal bayar terdeteksi tinggi.")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(proba.max())
            st.divider()
            st.metric("Tingkat Keyakinan Model", f"{confidence*100:.2f}%")
            st.progress(confidence)

        st.markdown("</div>", unsafe_allow_html=True)

st.caption("© 2026 Prediksi Pola Nasabah")
