import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Agriculture System",
    layout="wide"
)

# ---------------- IEEE STYLE THEME ----------------
st.markdown("""
<style>

/* -------- Background -------- */
.stApp {
    background-color: #FAFAFA;
}

/* -------- Typography -------- */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* -------- Title -------- */
h1 {
    color: #1B5E20;
    font-size: 34px;
    font-weight: 600;
}

h2, h3 {
    color: #2E7D32;
    font-weight: 500;
}

/* -------- Cards -------- */
.block-container {
    padding-top: 2rem;
}

/* -------- Metrics Card -------- */
div[data-testid="metric-container"] {
    background-color: #F1F8F4;
    border: 1px solid #E0E0E0;
    padding: 20px;
    border-radius: 10px;
}

/* -------- Sidebar -------- */
section[data-testid="stSidebar"] {
    background-color: #F5F5F5;
}

/* -------- Buttons -------- */
.stButton>button {
    background-color: #2E7D32;
    color: white;
    border-radius: 8px;
    padding: 10px;
    border: none;
}

.stButton>button:hover {
    background-color: #1B5E20;
}

/* -------- Divider -------- */
hr {
    border: 0.5px solid #E0E0E0;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("Smart Agriculture Decision Support System")
st.markdown("A Machine Learning-Based Framework for Crop Recommendation and Yield Prediction")

# ---------------- FIX PICKLE ERROR ----------------
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

    xgb_pred = xgb_model.predict(X_sel)
    final_yield = float(xgb_pred[0])

    # ---------------- OUTPUT ----------------
    st.markdown("### Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Recommended Crop", crop_name)

    with col2:
        st.metric("Predicted Yield", f"{final_yield:.2f}")

    st.markdown("---")

    # ---------------- GRAPH 1 ----------------
    st.subheader("Feature Analysis")

    feature_values = [soil, humidity, temp, rain, solar, fert, pest]
    feature_names = ["Soil", "Humidity", "Temperature", "Rainfall", "Solar", "Fertilizer", "Pesticide"]

    fig, ax = plt.subplots()
    ax.barh(feature_names, feature_values, color="#2E7D32")
    ax.set_xlabel("Value")
    ax.set_title("Input Features")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig)

    # ---------------- GRAPH 2 ----------------
    st.subheader("Yield Comparison")

    avg_yield = 50

    fig2, ax2 = plt.subplots()
    ax2.bar(["Predicted", "Baseline"], [final_yield, avg_yield], color=["#1B5E20", "#A5D6A7"])
    ax2.set_ylabel("Yield")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    st.pyplot(fig2)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("© Smart Agriculture Decision Support System | IEEE Project Presentation")
