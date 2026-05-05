import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Agriculture System", layout="wide")

# ---------------- CUSTOM GREEN THEME ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #F1F8F4;
    }
    h1, h2, h3 {
        color: #1B5E20;
    }
    .stMetric {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("Smart Agriculture Decision Support System")
st.markdown("Crop Recommendation and Yield Prediction using Machine Learning")

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

predict = st.sidebar.button("Predict")

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
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Recommended Crop", crop_name)

    with col2:
        st.metric("Predicted Yield", f"{final_yield:.2f}")

    st.markdown("---")

    # ---------------- GRAPH 1: FEATURE BAR ----------------
    st.subheader("Input Feature Analysis")

    feature_values = [soil, humidity, temp, rain, solar, fert, pest]
    feature_names = ["Soil", "Humidity", "Temperature", "Rainfall", "Solar", "Fertilizer", "Pesticide"]

    colors = ["#A5D6A7","#81C784","#66BB6A","#4CAF50","#388E3C","#2E7D32","#1B5E20"]

    fig, ax = plt.subplots()
    ax.barh(feature_names, feature_values, color=colors)
    ax.set_title("Input Parameters")
    ax.grid(False)
    st.pyplot(fig)

    # ---------------- GRAPH 2: DONUT CHART ----------------
    st.subheader("Resource Contribution")

    fig2, ax2 = plt.subplots()
    ax2.pie(
        feature_values,
        labels=feature_names,
        autopct="%1.1f%%",
        colors=colors,
        wedgeprops={'width':0.4}
    )
    ax2.set_title("Input Contribution Distribution")
    st.pyplot(fig2)

    # ---------------- GRAPH 3: YIELD COMPARISON ----------------
    st.subheader("Yield Comparison")

    avg_yield = 50

    fig3, ax3 = plt.subplots()
    ax3.bar(
        ["Predicted", "Average"],
        [final_yield, avg_yield],
        color=["#1B5E20","#81C784"]
    )
    ax3.set_ylabel("Yield")
    st.pyplot(fig3)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Smart Agriculture System | Green AI Deployment")
