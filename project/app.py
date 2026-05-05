import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Agriculture System", layout="wide")

# ---------------- IEEE CLEAN UI ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: #212121;
}
.stApp {
    background-color: #F6FAF6;
}
h1 {
    color: #1B5E20 !important;
    font-weight: 700;
    text-align: center;
}
h2 {
    color: #2E7D32 !important;
}
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E0E0;
}
section[data-testid="stSidebar"] > div {
    position: fixed;
    width: 300px;
}
label {
    color: #1B5E20 !important;
    font-weight: 600 !important;
}
.stSlider span {
    color: #2E7D32 !important;
}
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
}
[data-testid="stMetricValue"] {
    color: #1B5E20 !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #2E7D32, #1B5E20);
    color: white;
    font-size: 18px;
    padding: 14px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>Smart Agriculture Decision Support System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Crop Recommendation and Yield Prediction using Machine Learning</p>", unsafe_allow_html=True)

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
    return (
        joblib.load("models/crop_model.pkl"),
        joblib.load("models/yield_preprocessor.pkl"),
        joblib.load("models/yield_selector.pkl"),
        joblib.load("models/xgb_model.pkl"),
        joblib.load("models/le_crop.pkl"),
        joblib.load("models/le_region.pkl"),
    )

crop_model, yield_preprocessor, yield_selector, xgb_model, le_crop, le_region = load_models()

# ---------------- STATES ----------------
india_regions = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand",
    "Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur",
    "Meghalaya","Mizoram","Nagaland","Odisha","Punjab",
    "Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal",
    "Andaman and Nicobar Islands","Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi","Jammu and Kashmir","Ladakh","Lakshadweep","Puducherry"
]

# ---------------- SIDEBAR ----------------
st.sidebar.header("Input Parameters")

region = st.sidebar.selectbox("Region", india_regions)
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

    region_enc = le_region.transform([region])[0] if region in le_region.classes_ else 0

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

    # Feature engineering
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

    pred_crop = crop_model.predict(df[crop_cols])
    crop_name = le_crop.inverse_transform(pred_crop)[0]

    df["Crop_Type"] = pred_crop[0]
    df["Year"] = 2000

    yield_cols = [
        'Year','Crop_Type','Region','Soil_Moisture','Humidity','Temperature',
        'Rainfall','Solar_Radiation','Fertilizer_Residuals','Pesticide_Use',
        'Temp_Rain','Humidity_Soil','Solar_Temp','Chemical_Load','Moisture_Balance'
    ]

    X_proc = yield_preprocessor.transform(df[yield_cols])
    X_sel = yield_selector.transform(X_proc)
    final_yield = float(xgb_model.predict(X_sel)[0])

    # ---------------- DISPLAY ----------------
    st.markdown("## Prediction Results")

    col1, col2 = st.columns(2)
    col1.metric("Recommended Crop", crop_name)
    col2.metric("Predicted Yield", f"{final_yield:.2f}")

    st.markdown("---")

    # ---------------- GRAPH ----------------
    st.markdown("## Feature Contribution")

    feature_values = np.array([soil, humidity, temp, rain, solar, fert, pest])
    percent = (feature_values / feature_values.sum()) * 100

    names = ["Soil","Humidity","Temperature","Rainfall","Solar","Fertilizer","Pesticide"]

    df_plot = pd.DataFrame({"Feature": names, "Contribution (%)": percent}).sort_values(by="Contribution (%)")

    fig, ax = plt.subplots(figsize=(16,8))
    ax.barh(df_plot["Feature"], df_plot["Contribution (%)"], color="#2E7D32")

    for i, v in enumerate(df_plot["Contribution (%)"]):
        ax.text(v+1, i, f"{v:.1f}%", va='center')

    ax.set_title("Feature Importance")
    ax.grid(False)

    st.pyplot(fig)

    # ---------------- PDF GENERATION ----------------
    def generate_pdf():
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()

        content = []

        content.append(Paragraph("Smart Agriculture Report", styles["Title"]))
        content.append(Spacer(1, 10))

        content.append(Paragraph(f"<b>Region:</b> {region}", styles["Normal"]))
        content.append(Paragraph(f"<b>Recommended Crop:</b> {crop_name}", styles["Normal"]))
        content.append(Paragraph(f"<b>Predicted Yield:</b> {final_yield:.2f}", styles["Normal"]))
        content.append(Spacer(1, 10))

        content.append(Paragraph("<b>Input Parameters:</b>", styles["Heading2"]))
        for name, val in zip(names, feature_values):
            content.append(Paragraph(f"{name}: {val}", styles["Normal"]))

        doc.build(content)

        with open("report.pdf", "rb") as f:
            return f.read()

    pdf_data = generate_pdf()

    st.download_button(
        label="Download PDF Report",
        data=pdf_data,
        file_name="Smart_Agriculture_Report.pdf",
        mime="application/pdf"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>© Smart Agriculture System | IEEE Conference Presentation</p>", unsafe_allow_html=True)
