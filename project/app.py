import streamlit as st
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Agriculture DSS",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* ---------- MAIN TITLE ---------- */
.main-title {
    font-size:30px;
    font-weight:700;
    color:#2E7D32;
}

/* ---------- SUBTITLE ---------- */
.subtitle {
    font-size:15px;
    color:#B0BEC5;
    margin-bottom:20px;
}

/* ---------- SECTION TITLE ---------- */
.section-title {
    font-size:18px;
    font-weight:600;
    margin-top:25px;
    margin-bottom:10px;
    color:#E0E0E0;
}

/* ---------- METRIC CARDS ---------- */
.metric-box {
    background: linear-gradient(135deg, #1B5E20, #2E7D32);
    padding:25px;
    border-radius:12px;
    text-align:center;
    color:white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: 0.3s;
}

.metric-box:hover {
    transform: scale(1.02);
}

.metric-title {
    font-size:14px;
    opacity:0.85;
}

.metric-value {
    font-size:30px;
    font-weight:700;
    margin-top:10px;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align:center;
    font-size:13px;
    color:#9E9E9E;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">Smart Agriculture Decision Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Crop Recommendation and Yield Prediction using Machine Learning Models</div>', unsafe_allow_html=True)

# ---------------- FIX PICKLE ----------------
class CustomSelectorForSaving:
    def __init__(self, mask):
        self.mask = mask

    def get_support(self, indices=False):
        return np.where(self.mask)[0] if indices else self.mask

    def transform(self, X):
        return X[:, self.mask]

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    crop_pipeline = joblib.load("models/crop_model.pkl")
    yield_preprocessor = joblib.load("models/yield_preprocessor.pkl")
    yield_selector = joblib.load("models/yield_selector.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    ann_scaler = joblib.load("models/ann_scaler.pkl")
    ann_model = load_model("models/ann_model.h5", custom_objects={'mse': MeanSquaredError()})

    le_crop = joblib.load("models/le_crop.pkl")
    le_region = joblib.load("models/le_region.pkl")

    return crop_pipeline, yield_preprocessor, yield_selector, xgb_model, ann_scaler, ann_model, le_crop, le_region


crop_pipeline, yield_preprocessor, yield_selector, xgb_model, ann_scaler, ann_model, le_crop, le_region = load_models()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("### Input Parameters")

region = st.sidebar.selectbox("Region", le_region.classes_)
soil = st.sidebar.slider("Soil Moisture", 0.0, 10.0, 5.0)
humidity = st.sidebar.slider("Humidity", 0.0, 100.0, 60.0)
temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
rain = st.sidebar.slider("Rainfall (mm)", 0.0, 2000.0, 500.0)
solar = st.sidebar.slider("Solar Radiation", 0.0, 10.0, 5.0)
fert = st.sidebar.slider("Fertilizer Residuals", 0.0, 500.0, 100.0)
pest = st.sidebar.slider("Pesticide Use", 0.0, 500.0, 100.0)

predict = st.sidebar.button("Run Prediction")

# ---------------- MAIN ----------------
if predict:

    st.markdown('<div class="section-title">Model Prediction Output</div>', unsafe_allow_html=True)

    # ---------------- INPUT ----------------
    region_enc = le_region.transform([region])[0]

    df = pd.DataFrame([{
        "Region": region_enc,
        "Soil_Moisture": soil,
        "Humidity": humidity,
        "Temperature": temp,
        "Rainfall": rain,
        "Solar_Radiation": solar,
        "Fertilizer_Residuals": fert,
        "Pesticide_Use": pest
    }])

    # ---------------- FEATURE ENGINEERING ----------------
    df["Temp_Rain"] = df["Temperature"] * df["Rainfall"]
    df["Humidity_Soil"] = df["Humidity"] * df["Soil_Moisture"]
    df["Solar_Temp"] = df["Solar_Radiation"] * df["Temperature"]
    df["Chemical_Load"] = df["Fertilizer_Residuals"] + df["Pesticide_Use"]
    df["Moisture_Balance"] = df["Rainfall"] / (df["Temperature"] + 1)

    crop_cols = [
        'Soil_Moisture','Humidity','Temperature','Rainfall','Solar_Radiation',
        'Fertilizer_Residuals','Pesticide_Use','Temp_Rain','Humidity_Soil',
        'Solar_Temp','Chemical_Load','Moisture_Balance'
    ]

    # ---------------- CROP PREDICTION ----------------
    X_crop = df[crop_cols]
    pred_crop = crop_pipeline.predict(X_crop)
    crop_name = le_crop.inverse_transform(pred_crop)[0]

    # ---------------- YIELD PREDICTION ----------------
    df["Crop_Type"] = pred_crop[0]
    df["Year"] = 2000

    yield_cols = [
        'Year','Crop_Type','Region','Soil_Moisture','Humidity','Temperature',
        'Rainfall','Solar_Radiation','Fertilizer_Residuals','Pesticide_Use',
        'Temp_Rain','Humidity_Soil','Solar_Temp','Chemical_Load','Moisture_Balance'
    ]

    X_yield = df[yield_cols]

    X_proc = yield_preprocessor.transform(X_yield)
    X_sel = yield_selector.transform(X_proc)

    xgb_pred = xgb_model.predict(X_sel)

    X_hybrid = np.column_stack((X_sel, xgb_pred))
    X_scaled = ann_scaler.transform(X_hybrid)

    residual = ann_model.predict(X_scaled).flatten()
    final_yield = xgb_pred + residual

    # ---------------- OUTPUT ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Recommended Crop</div>
            <div class="metric-value">{crop_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Predicted Yield</div>
            <div class="metric-value">{final_yield[0]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

   

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown('<div class="footer">Smart Agriculture System | Machine Learning + Deep Learning</div>', unsafe_allow_html=True)