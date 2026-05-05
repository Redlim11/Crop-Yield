
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Agriculture System", layout="wide")

# ---------------- IEEE CLEAN UI ----------------
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: #212121;
}

/* ---------- BACKGROUND ---------- */
.stApp {
    background-color: #F6FAF6;
}

/* ---------- TITLE ---------- */
h1 {
    color: #1B5E20 !important;
    font-weight: 700;
    text-align: center;
}

h2 {
    color: #2E7D32 !important;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E0E0;
}

section[data-testid="stSidebar"] > div {
    position: fixed;
    width: 300px;
}

/* ---------- INPUT LABELS ---------- */
label {
    color: #1B5E20 !important;
    font-weight: 600 !important;
}

/* ---------- SLIDER VALUE ---------- */
.stSlider span {
    color: #2E7D32 !important;
    font-weight: 600 !important;
}

/* ---------- SELECTBOX ---------- */
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border: 1px solid #A5D6A7 !important;
    border-radius: 6px !important;
}

/* ---------- METRIC ---------- */
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

[data-testid="stMetricLabel"] {
    color: #424242 !important;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #2E7D32, #1B5E20);
    color: white;
    font-size: 18px;
    font-weight: 600;
    padding: 14px;
    border-radius: 8px;
    border: none;
}
/* ---------- SUCCESS BOX TEXT DARK GREEN ---------- */
div[data-testid="stAlert"] {
    background-color: #E8F5E9 !important;   /* light green background */
    border-left: 5px solid #1B5E20 !important;
}

div[data-testid="stAlert"] p {
    color: #1B5E20 !important;   /* DARK GREEN TEXT */
    font-weight: 600;
}

/* ---------- SPACING ---------- */
.block-container {
    padding-top: 1.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
}
/* ---------- DOWNLOAD BUTTON MATCH ---------- */
div.stDownloadButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2E7D32, #1B5E20);
    color: white;
    font-size: 18px;
    font-weight: 600;
    padding: 14px;
    border-radius: 8px;
    border: none;
}

div.stDownloadButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #1B5E20, #0D3B12);
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1> </h1>", unsafe_allow_html=True)
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
    crop_model = joblib.load("models/crop_model.pkl")
    yield_preprocessor = joblib.load("models/yield_preprocessor.pkl")
    yield_selector = joblib.load("models/yield_selector.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    le_crop = joblib.load("models/le_crop.pkl")
    le_region = joblib.load("models/le_region.pkl")

    return crop_model, yield_preprocessor, yield_selector, xgb_model, le_crop, le_region

crop_model, yield_preprocessor, yield_selector, xgb_model, le_crop, le_region = load_models()

# ---------------- ALL INDIA STATES + UT ----------------
india_regions = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand",
    "Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur",
    "Meghalaya","Mizoram","Nagaland","Odisha","Punjab",
    "Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal",
    "Andaman and Nicobar Islands","Chandigarh","Dadra and Nagar Haveli and Daman and Diu",
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

    # Safe encoding
    if region in le_region.classes_:
        region_enc = le_region.transform([region])[0]
    else:
        region_enc = le_region.transform([le_region.classes_[0]])[0]

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

    # Crop Prediction
    pred_crop = crop_model.predict(df[crop_cols])
    crop_name = le_crop.inverse_transform(pred_crop)[0]

    # Yield Prediction
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

    # ---------------- RESULTS ----------------
    st.markdown("## Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Recommended Crop", crop_name)

    with col2:
        st.metric("Predicted Yield", f"{final_yield:.2f}")

    st.markdown("---")

    # ---------------- FARMER RECOMMENDATIONS ----------------
    st.markdown("## Farmer Recommendations")
    
    rec = []
    
    # 🌧️ Rainfall analysis
    if rain < 300:
        rec.append("Rainfall is insufficient (<300 mm). Adopt drip irrigation or sprinkler systems to maintain soil moisture.")
    elif rain > 1200:
        rec.append("High rainfall detected (>1200 mm). Ensure proper drainage to prevent waterlogging and root damage.")
    
    # 🌱 Soil moisture
    if soil < 3:
        rec.append("Soil moisture is low. Apply mulching and increase irrigation frequency to retain water.")
    elif soil > 8:
        rec.append("Excess soil moisture detected. Improve drainage to avoid root diseases.")
    
    # 🌡️ Temperature suitability (crop-aware)
    if temp > 35:
        rec.append(f"High temperature (>35°C) may stress {crop_name}. Consider heat-resistant varieties or shade management.")
    elif temp < 15:
        rec.append(f"Low temperature (<15°C) may slow growth of {crop_name}. Adjust sowing time or use protective cultivation.")
    
    # 💧 Humidity
    if humidity < 30:
        rec.append("Low humidity may cause plant stress. Increase irrigation and reduce evaporation losses.")
    elif humidity > 85:
        rec.append("High humidity may promote fungal diseases. Apply preventive fungicides and improve airflow.")
    
    # 🧪 Fertilizer usage
    if fert > 300:
        rec.append("Excess fertilizer detected. Overuse can degrade soil health. Follow soil testing and balanced NPK application.")
    elif fert < 50:
        rec.append("Low fertilizer input detected. Consider applying balanced nutrients based on soil testing.")
    
    # 🐛 Pesticide usage
    if pest > 300:
        rec.append("High pesticide usage detected. Switch to Integrated Pest Management (IPM) and biological control methods.")
    elif pest < 50:
        rec.append("Low pesticide usage. Monitor crops regularly for early pest detection.")
    
    # 🌾 Crop-specific recommendation
    if crop_name.lower() == "rice":
        rec.append("Rice requires standing water. Maintain field flooding during vegetative stages.")
    elif crop_name.lower() == "wheat":
        rec.append("Wheat performs best in well-drained soils. Avoid excess irrigation during maturity stage.")
    elif crop_name.lower() == "maize":
        rec.append("Maize needs good sunlight and moderate water. Ensure proper spacing for yield optimization.")
    
    # ✅ fallback
    if not rec:
        rec.append("All environmental conditions are optimal. Maintain current farming practices.")
    
    # Display
    for r in rec:
        st.success(r)

    # ---------------- CROP ROTATION ----------------
    st.markdown("## Crop Rotation Recommendation")
    
    rotation_map = {
        "Rice": "Follow with pulses (lentil/gram) to restore nitrogen in soil.",
        "Wheat": "Rotate with legumes (chickpea/pea) for soil fertility improvement.",
        "Maize": "Rotate with soybean or groundnut to enhance nitrogen fixation.",
        "Sugarcane": "Rotate with legumes or vegetables to avoid soil depletion.",
        "Soybean": "Rotate with cereals like wheat or maize for balanced nutrient usage."
    }
    
    rotation_advice = rotation_map.get(crop_name, 
        "Adopt crop rotation with legumes to maintain long-term soil fertility.")
    
    st.success(rotation_advice)
    # ---------------- FEATURE GRAPH ----------------
    st.markdown("## Feature Contribution")

    feature_values = np.array([soil, humidity, temp, rain, solar, fert, pest])
    feature_names = ["Soil", "Humidity", "Temperature", "Rainfall", "Solar", "Fertilizer", "Pesticide"]

    percent = (feature_values / feature_values.sum()) * 100

    df_plot = pd.DataFrame({
        "Feature": feature_names,
        "Contribution (%)": percent
    }).sort_values(by="Contribution (%)", ascending=True)

    fig, ax = plt.subplots(figsize=(16,8))

   # Sort already done (ascending=True)
    values = df_plot["Contribution (%)"].values
    
    # Normalize values (0 → 1)
    norm = (values - values.min()) / (values.max() - values.min())
    
    # Use GREEN gradient (light → dark)
    colors = plt.cm.Greens(0.3 + 0.7 * norm)
    

    ax.invert_yaxis()
    
    # Plot
    bars = ax.barh(df_plot["Feature"], values, color=colors)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}%", va='center', fontsize=13)

    # Axis labels size
    ax.set_xlabel("Contribution (%)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Features", fontsize=16, fontweight='bold')
    # Increase axis tick label size
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.grid(False)

    plt.tight_layout()

    col1, col2, col3 = st.columns([1,5,1])
    with col2:
        st.pyplot(fig)
    # Save graph image
    fig.savefig("graph.png")
        # ---------------- PDF GENERATION ----------------
    def generate_pdf():

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        content = []

        content.append(Paragraph("Smart Agriculture Report", styles["Title"]))
        content.append(Spacer(1, 10))

        content.append(Paragraph(f"<b>Region:</b> {region}", styles["Normal"]))
        content.append(Paragraph(f"<b>Recommended Crop:</b> {crop_name}", styles["Normal"]))
        content.append(Paragraph(f"<b>Predicted Yield:</b> {final_yield:.2f}", styles["Normal"]))
        content.append(Spacer(1, 10))

        content.append(Paragraph("Farmer Recommendations:", styles["Heading2"]))
        for r in rec:
            content.append(Paragraph(f"• {r}", styles["Normal"]))

        content.append(Spacer(1, 10))
        content.append(Paragraph("Crop Rotation Advice:", styles["Heading2"]))
        content.append(Paragraph(rotation_advice, styles["Normal"]))
        # Save graph into buffer (NOT file)
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png")
        img_buffer.seek(0)

        content.append(Paragraph("Feature Contribution Graph:", styles["Heading2"]))
        content.append(Image(img_buffer, width=400, height=250))

        doc.build(content)

        pdf = buffer.getvalue()
        buffer.close()

        return pdf


    pdf = generate_pdf()

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf,
        file_name="Smart_Agriculture_Report.pdf",
        mime="application/pdf"
    )
# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>© Hybrid Model for Smart Agriculture Decision Support System </p>", unsafe_allow_html=True)
