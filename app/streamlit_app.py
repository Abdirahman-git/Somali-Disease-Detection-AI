import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import altair as alt # Keep altair for the chart

# Add parent directory for config.py
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR

# ------------------------------------
# PAGE SETTINGS
# ------------------------------------
st.set_page_config(
    page_title="Somali Disease Detection AI",
    page_icon="🧬",
    layout="wide"
)

# ------------------------------------
# ADVANCED PREMIUM CSS (High-Contrast, Dark Theme)
# ------------------------------------
st.markdown("""
<style>

/* --- 1. Global & Background (Dark Theme) --- */
body {
    background-color: #0b0c10; /* Very Dark Background */
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* --- 2. Headers & Typography --- */
.main-header {
    text-align: center;
    font-size: 60px; /* Bigger, bolder */
    font-weight: 900;
    /* Neon/Electric Gradient for contrast */
    background: linear-gradient(90deg, #66ccff, #ff66cc); /* Bright Blue to Magenta */
    -webkit-background-clip: text;
    color: transparent;
    margin-bottom: 0.5rem;
}

.sub-header {
    text-align: center;
    font-size: 24px;
    color: #a9b3c1; /* Light gray for readability on dark background */
    margin-top: 0px;
    margin-bottom: 2.5rem;
}

/* --- 3. Card Style (General) --- */
.stContainer > div, .card {
    background: #1f2833; /* Darker Blue/Gray Card Background */
    padding: 30px; /* Increased padding */
    border-radius: 16px; /* Rounder corners */
    border: 1px solid #333f50; /* Subtle border for separation */
    box-shadow: 0px 10px 25px rgba(0, 0, 0, 0.4); /* Stronger, deeper shadow */
    transition: all 0.3s ease-in-out;
    margin-bottom: 20px;
}

.card:hover {
    transform: translateY(-8px); /* More pronounced lift */
    box-shadow: 0px 18px 40px rgba(102, 204, 255, 0.15); /* Shadow glows with accent color */
}

/* --- 4. Metric Card (Results) --- */
.metric-box {
    padding: 30px 20px;
    /* High-contrast gradient */
    background: linear-gradient(145deg, #1f2833, #0b0c10); 
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(102, 204, 255, 0.3); /* Stronger accent shadow */
    border: 2px solid #66ccff; /* Feature line as a full border */
}

.metric-title {
    font-size: 18px;
    color: #45a29e; /* Teal accent color */
    font-weight: 700;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 48px; /* Very large number */
    font-weight: 900;
    color: #66ccff; /* Primary Bright Blue */
}

/* --- 5. Sidebar Styling --- */
.css-1lcbmhc {
    background-color: #1a1a2e; /* Slightly lighter dark sidebar */
    box-shadow: 4px 0 15px rgba(0, 0, 0, 0.6);
}
.sidebar h2 {
    color: #ff66cc; /* Secondary accent color (Magenta) */
    border-bottom: 2px solid #333f50;
    padding-bottom: 12px;
}
/* Ensure sidebar text and inputs are readable */
.st-emotion-cache-1jmpsa8 { /* Targeting generic Streamlit text/label in dark mode */
    color: #a9b3c1 !important;
}

/* --- 6. Button Styling --- */
.stButton button {
    height: auto !important;
    padding: 14px 10px !important; /* Larger button */
    border-radius: 10px;
    background: linear-gradient(90deg, #66ccff, #ff66cc);
    color: #0b0c10; /* Dark text on bright button */
    font-weight: 800;
    font-size: 1.2rem;
    border: none;
    transition: all 0.3s ease;
}

.stButton button:hover {
    background: linear-gradient(90deg, #ff66cc, #66ccff); /* Reverse colors on hover */
    box-shadow: 0 6px 20px rgba(255, 102, 204, 0.6); /* Glowing hover shadow */
}

/* --- 7. Info/Alert Box Styling (Diagnosis Explanation) --- */
.stAlert > div {
    background-color: #153c48; /* Dark teal background for info */
    color: #66ccff;
    border-left: 6px solid #45a29e; /* Teal feature line */
    border-radius: 8px;
    font-size: 1.05rem;
    padding: 20px;
}

/* Style for general markdown text to be light on dark background */
.st-emotion-cache-12qukbp { /* Targeting main body markdown text container */
    color: #a9b3c1;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------
# HEADER
# ------------------------------------
st.markdown("<h1 class='main-header'>🧬 Somali Disease Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Advanced Machine Learning for Somali Healthcare Diagnosis</p>", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------
# LOAD MODEL
# ------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODELS_DIR / 'disease_model.pkl')
        label_encoder = joblib.load(MODELS_DIR / 'label_encoder.pkl')
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        return model, label_encoder, scaler
    except:
        st.error("❌ Model files missing! Run: `python train_model.py`")
        return None, None, None

model, label_encoder, scaler = load_model()


# ------------------------------------
# DISEASE DETAILS
# ------------------------------------
DISEASE_INFO = {
    "Malaria": {"so": "Malariya waxaa keena kaneeco.", "en": "Mosquito-borne disease.", "sym": ["Fever","Chills","Fatigue"]},
    "TB": {"so": "TB waa xanuun sambabada ku dhaca.", "en": "Airborne bacterial infection.", "sym": ["Cough","Sweat","Fatigue"]},
    "Typhoid": {"so": "Typhoid waxaa keena Salmonella.", "en": "Typhoid fever bacteria.", "sym": ["Fever","Vomiting","Headache"]},
    "Cholera": {"so": "Koolera waa shuban biyood halis ah.", "en": "Severe diarrheal disease.", "sym": ["Diarrhea","Vomiting"]},
    "Measles": {"so": "Jadeeco waa rash leh.", "en": "Highly contagious viral disease.", "sym": ["Rash","Fever","Cough"]},
    "COVID-19": {"so": "Koronaha cusub.", "en": "Coronavirus disease.", "sym": ["Fever","Cough","Fatigue"]},
    "Diabetes": {"so": "Suka waa xanuun insulin la’aan ah.", "en": "Insulin-related condition.", "sym": ["Thirst","Fatigue"]},
    "Hypertension": {"so": "Cadaadis dhiig sare.", "en": "High blood pressure.", "sym": ["Headache","Dizziness"]},
    "Healthy": {"so": "Caafimaad buuxa.", "en": "No disease.", "sym": ["None"]},
}

# ------------------------------------
# SIDEBAR INPUTS
# ------------------------------------
with st.sidebar:
    st.header("📋 Patient Information")
    
    # Use st.form for a cleaner separation and potential future batch input
    with st.form("patient_form"):
        age = st.number_input("Age", 1, 120, 30, help="Patient's age in years.")
        temperature = st.number_input("Temperature (°C)", 35.0, 45.0, 37.0, step=0.1, format="%.1f", help="Patient's current body temperature.")

        st.header("🤒 Symptoms Check")
        colA, colB = st.columns(2)
        
        with colA:
            headache = st.checkbox("Headache", help="Does the patient have a headache?")
            cough = st.checkbox("Cough", help="Is the patient coughing?")
            vomiting = st.checkbox("Vomiting", help="Is the patient experiencing vomiting?")
            diarrhea = st.checkbox("Diarrhea", help="Is the patient experiencing diarrhea?")

        with colB:
            fatigue = st.checkbox("Fatigue", help="Is the patient unusually tired?")
            rash = st.checkbox("Rash", help="Does the patient have a skin rash?")
            bleeding = st.checkbox("Bleeding", help="Is there any unexplained bleeding?")
            anaemia = st.checkbox("Anaemia", help="Is the patient diagnosed with or showing signs of anaemia?")

        detect = st.form_submit_button("🔍 Detect Disease")


# ------------------------------------
# LOGIC
# ------------------------------------
if detect and model is not None:
    features = [
        age, temperature,
        int(headache), int(cough), int(vomiting), int(diarrhea),
        int(fatigue), int(rash), int(bleeding), int(anaemia)
    ]

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    prob = model.predict_proba(features_scaled)

    disease = label_encoder.inverse_transform(prediction)[0]
    confidence = max(prob[0]) * 100

    st.markdown("### 🧪 Diagnosis Result")
    
    # ---------------- METRIC CARDS (Using the updated metric-box CSS) -----------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Detected Disease</div><div class='metric-value'>{disease}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><div class='metric-title'>Confidence</div><div class='metric-value'>{confidence:.1f}%</div></div>", unsafe_allow_html=True)


    # ---------------- DISEASE EXPLANATION (Inside a structured Card) ----------------
    st.markdown("### ℹ️ Disease Information")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Use st.expander for a clean, collapsible section
    with st.expander(f"**Description & Symptoms for {disease}**", expanded=True):
        st.subheader("Description")
        st.info(f"**Somali:** {DISEASE_INFO[disease]['so']}")
        st.info(f"**English:** {DISEASE_INFO[disease]['en']}")

        st.subheader("Common Symptoms")
        symptoms_list = [f"• **{s}**" for s in DISEASE_INFO[disease]["sym"]]
        st.markdown('\n'.join(symptoms_list))

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PROBABILITY DISTRIBUTION --------------------
    st.subheader("📊 Probability Distribution")
    
    # Wrap in a card for better visual grouping
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    df_prob = pd.DataFrame({
        "Disease": label_encoder.classes_,
        "Probability": [p * 100 for p in prob[0]]
    })
    
    
    # Use altair for better control over visual - Now with a dark theme color scheme
    chart = alt.Chart(df_prob).mark_bar().encode(
        x=alt.X('Probability', axis=alt.Axis(title='Probability (%)', titleColor='#a9b3c1', labelColor='#a9b3c1')),
        y=alt.Y('Disease', sort='-x', axis=alt.Axis(title='Disease', titleColor='#a9b3c1', labelColor='#a9b3c1')),
        tooltip=['Disease', alt.Tooltip('Probability', format='.1f')],
        color=alt.condition(
            alt.datum.Probability == df_prob['Probability'].max(), # Highlight max probability
            alt.value('#66ccff'),  # Primary Bright Blue for max
            alt.value('#335d68')   # Muted teal/blue for others
        )
    ).properties(
        title=alt.TitleParams('Model Confidence Across All Diseases', anchor='middle', fontSize=18, color='#a9b3c1'),
        height=400 
    ).configure_view(
        stroke='transparent'
    ).configure_axis(
        gridColor='#333f50'
    )
    
    st.altair_chart(chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # ---------------- WELCOME CARD --------------------
    st.markdown("### 👋 Welcome")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    This advanced AI system assists Somali healthcare professionals by detecting potential diseases based on patient information and symptoms.

    **To begin:**
    1.  Navigate to the **Patient Information** section on the left sidebar.
    2.  Enter the patient's **Age** and **Temperature (°C)**.
    3.  Check all relevant **Symptoms**.
    4.  Click the **🔍 Detect Disease** button to view the predicted diagnosis and confidence level.

    *This tool is for informational support and should not replace professional medical advice.*
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------
# FOOTER
# ------------------------------------
st.markdown("---")
st.markdown("#### 🌍 Somali Disease Detection AI © 2025 — Advanced Edition")
st.caption("Built for Healthcare • Powered by Machine Learning")