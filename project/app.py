import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Agriculture System", layout="wide")

# ---------------- IEEE CLEAN UI ----------------
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: #1C1C1C;
}

/* ---------- BACKGROUND ---------- */
.stApp {
    background-color: #F8FBF8;
}

/* ---------- TITLE ---------- */
h1 {
    color: #1B5E20 !important;
    font-weight: 700;
    text-align: center;
}

h2, h3 {
    color: #2E7D32 !important;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E0E0;
}

/* ---------- FIXED SIDEBAR ---------- */
section[data-testid="stSidebar"] > div {
    position: fixed;
    width: 300px;
}

/* ---------- SELECTBOX MAIN ---------- */
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border: 1px solid #C8E6C9 !important;
    border-radius: 6px !important;
}

/* ---------- DROPDOWN OPTIONS ---------- */
ul[role="listbox"] {
    background-color: #FFFFFF !important;
    color: black !important;
}

/* ---------- SELECT TEXT ---------- */
div[data-baseweb="select"] * {
    color: #1B5E20 !important;
}

/* ---------- METRIC ---------- */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
    text-align: center;
}

/* ---------- BUTTON (IMPORTANT FIX) ---------- */
.stButton>button {
    width: 100%;
    background-color: #2E7D32;
    color: white;
    font-size: 16px;
    font-weight: 600;
    padding: 12px;
    border-radius: 8px;
    border: none;
}

.stButton>button:hover {
    background-color: #1B5E20;
    transform: scale(1.02);
}

/* ---------- SPACING ---------- */
.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>Smart Agriculture Decision Support System</h1>", unsafe_allow_html=True)
st.markdown("<p>Crop Recommendation and Yield Prediction using Machine Learning</p>", unsafe_allow_html=True)

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
    crop_model = joblib.load("models/crop_model.pkl")
    yield_preprocessor = joblib.load("models/yield_preprocessor.pkl")
    yield_selector = joblib.load("models/yield_selector.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    le_crop = joblib.load("models/le_crop.pkl")
    le_region = joblib.load("models/le_region.pkl")

    return crop_model, yield_preprocessor, yield_selector, xgb_model, le_crop, le_region


crop_model, yield_preprocessor, yield_selector, xgb_model, le_crop, le_region = load_models()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Input Parameters")

region = st.sidebar.selectbox("Region", le_region.classes_)
soil = st.sidebar.slider("Soil Moisture", 0.0, 10.0, 5.0)
humidity = st.sidebar.slider("Humidity", 0.0, 100.0, 60.0)
temp = st.sidebar.slider("Temperature", 0.0, 50.0, 25.0)
rain = st.sidebar.slider("Rainfall", 0.0, 2000.0, 500.0)
solar = st.sidebar.slider("Solar Radiation", 0.0, 10.0, 5.0)
fert = st.sidebar.slider("Fertilizer Residuals", 0.0, 500.0, 100.0)
pest = st.sidebar.slider("Pesticide Use", 0.0, 500.0, 100.0)

predict = st.sidebar.button("Run Prediction")

# ---------------- MAIN ----------------
if predict:

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

    # Feature Engineering
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

    # ---------------- CROP ----------------
    X_crop = df[crop_cols]
    pred_crop = crop_model.predict(X_crop)
    crop_name = le_crop.inverse_transform(pred_crop)[0]

    # ---------------- YIELD ----------------
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

    final_yield = float(xgb_model.predict(X_sel)[0])

    # ---------------- RESULTS ----------------
    st.markdown("## Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Recommended Crop", crop_name)

    with col2:
        st.metric("Predicted Yield", f"{final_yield:.2f}")

    st.markdown("---")

    # ---------------- FEATURE CONTRIBUTION ----------------
    st.markdown("## Feature Contribution")

    feature_values = np.array([soil, humidity, temp, rain, solar, fert, pest])
    feature_names = ["Soil", "Humidity", "Temperature", "Rainfall", "Solar", "Fertilizer", "Pesticide"]

    percent = (feature_values / feature_values.sum()) * 100

    df_plot = pd.DataFrame({
        "Feature": feature_names,
        "Contribution (%)": percent
    }).sort_values(by="Contribution (%)", ascending=True)

    fig, ax = plt.subplots(figsize=(7,5))

    bars = ax.barh(df_plot["Feature"], df_plot["Contribution (%)"], color="#2E7D32")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}%", va='center')

    ax.set_xlabel("Contribution (%)")
    ax.set_title("Feature Importance")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.grid(False)

    plt.tight_layout()

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>© Smart Agriculture System | IEEE Conference Presentation</p>", unsafe_allow_html=True)
