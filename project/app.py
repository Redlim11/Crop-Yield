st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: #212121 !important;
}

.stApp {
    background-color: #FAFAFA;
}

/* ---------- HEADINGS ---------- */
h1 {
    color: #1B5E20 !important;
    font-weight: 700;
}

h2, h3 {
    color: #2E7D32 !important;
}

/* ---------- SIDEBAR FIXED ---------- */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E0E0;
    
    position: fixed !important;
    height: 100vh;
    overflow-y: auto;
}

/* ---------- MAIN SHIFT (important!) ---------- */
section.main > div {
    margin-left: 320px;
}

/* ---------- SELECTBOX (Region Dropdown FIX) ---------- */
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border: 1px solid #C8E6C9 !important;
    border-radius: 6px !important;
}

div[data-baseweb="select"] * {
    color: #1B5E20 !important;
}

/* ---------- SLIDERS ---------- */
div[data-testid="stSlider"] {
    padding-top: 10px;
}

/* ---------- METRIC CARDS ---------- */
div[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 3px 6px rgba(0,0,0,0.05);
}

/* ---------- FIX FADED TEXT ---------- */
[data-testid="stMetricValue"] {
    color: #1B5E20 !important;
    font-weight: 700;
    opacity: 1 !important;
}

[data-testid="stMetricLabel"] {
    color: #424242 !important;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background-color: #2E7D32;
    color: white;
    border-radius: 6px;
    padding: 10px 16px;
    border: none;
}

.stButton>button:hover {
    background-color: #1B5E20;
}

</style>
""", unsafe_allow_html=True)
