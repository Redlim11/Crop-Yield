import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Agriculture System", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp { background-color: #F6FAF6; }
h1 { color:#1B5E20; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Smart Agriculture Decision Support System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Crop Recommendation and Yield Prediction</p>", unsafe_allow_html=True)

# ---------------- FIX PICKLE ----------------
class CustomSelectorForSaving:
    def __init__(self, mask): self.mask = mask
    def get_support(self, indices=False): return np.where(self.mask)[0] if indices else self.mask
    def transform(self, X): return X[:, self.mask]

# ---------------- LOAD ----------------
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
    "Andhra Pradesh","Kerala","Tamil Nadu","Karnataka","Maharashtra",
    "Punjab","Rajasthan","Uttar Pradesh","West Bengal","Gujarat"
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
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    col1.metric("Recommended Crop", crop_name)
    col2.metric("Predicted Yield", f"{final_yield:.2f}")

    # ---------------- FARMER RECOMMENDATIONS ----------------
    st.subheader("Farmer Recommendations")

    rec = []

    if rain < 300:
        rec.append("Low rainfall detected → consider irrigation support.")
    if soil < 3:
        rec.append("Low soil moisture → improve irrigation or mulching.")
    if fert > 300:
        rec.append("High fertilizer usage → reduce for sustainable farming.")
    if pest > 300:
        rec.append("High pesticide use → consider organic pest control.")

    if not rec:
        rec.append("Conditions are optimal for farming.")

    for r in rec:
        st.success(r)

    # ---------------- GRAPH ----------------
    st.subheader("Feature Contribution")

    values = np.array([soil, humidity, temp, rain, solar, fert, pest])
    names = ["Soil","Humidity","Temperature","Rainfall","Solar","Fertilizer","Pesticide"]

    percent = (values / values.sum()) * 100

    df_plot = pd.DataFrame({"Feature": names, "Contribution": percent}).sort_values(by="Contribution")

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(df_plot["Feature"], df_plot["Contribution"], color="#2E7D32")

    for i, v in enumerate(df_plot["Contribution"]):
        ax.text(v+1, i, f"{v:.1f}%", va='center')

    ax.set_title("Feature Contribution")
    ax.grid(False)

    st.pyplot(fig)

    # Save graph image
    fig.savefig("graph.png")

    # ---------------- PDF ----------------
    def generate_pdf():
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()
        content = []

        content.append(Paragraph("Smart Agriculture Report", styles["Title"]))
        content.append(Spacer(1,10))

        content.append(Paragraph(f"Region: {region}", styles["Normal"]))
        content.append(Paragraph(f"Crop: {crop_name}", styles["Normal"]))
        content.append(Paragraph(f"Yield: {final_yield:.2f}", styles["Normal"]))
        content.append(Spacer(1,10))

        content.append(Paragraph("Recommendations:", styles["Heading2"]))
        for r in rec:
            content.append(Paragraph(f"- {r}", styles["Normal"]))

        content.append(Spacer(1,10))
        content.append(Paragraph("Feature Contribution Graph:", styles["Heading2"]))
        content.append(Image("graph.png", width=400, height=250))

        doc.build(content)

        with open("report.pdf", "rb") as f:
            return f.read()

    pdf = generate_pdf()

    st.download_button(
        "Download PDF Report",
        pdf,
        "Smart_Agriculture_Report.pdf",
        "application/pdf"
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>© IEEE Smart Agriculture Project</p>", unsafe_allow_html=True)
