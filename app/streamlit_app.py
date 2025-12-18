import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import altair as alt
import time

# ========== PATH CONFIGURATION ==========
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent 
MODELS_DIR = project_root / "models"
DATA_DIR = project_root / "data"

# ========== PAGE SETTINGS ==========
st.set_page_config(
    page_title="Somali Disease Detection AI",
    page_icon="🧬",
    layout="wide"
)

# ========== PREMIUM CSS (DARK THEME) ==========
st.markdown("""
<style>
    /* Global Styles */
    [data-testid="stSidebar"] { background-color: #0b0c10; }
    .stApp { background-color: #0b0c10; color: #c5c6c7; }
    
    /* Headers */
    .main-header {
        text-align: center; font-size: 50px; font-weight: 900;
        background: linear-gradient(90deg, #66ccff, #ff66cc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        text-align: center; font-size: 18px; color: #45a29e;
        margin-bottom: 2rem; letter-spacing: 1px;
    }

    /* Professional Cards */
    .card {
        background: #1f2833; padding: 25px; border-radius: 15px;
        border: 1px solid #333f50; margin-bottom: 20px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Results Styling */
    .metric-box {
        padding: 25px; border-radius: 15px; text-align: center;
        background: #1a1a2e; border: 2px solid #66ccff;
        box-shadow: 0 0 15px rgba(102, 204, 255, 0.2);
    }
    .metric-value { font-size: 40px; font-weight: 900; color: #66ccff; }
    .metric-label { font-size: 14px; color: #45a29e; text-transform: uppercase; }

    /* Buttons */
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3em;
        background: linear-gradient(90deg, #66ccff, #ff66cc);
        color: #0b0c10; font-weight: bold; border: none;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(102, 204, 255, 0.5);
        color: #0b0c10; transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ========== DATA & MODELS ==========
DISEASE_INFO = {
    "Malaria": {"so": "Malariya waxaa keena kaneeco.", "en": "Mosquito-borne disease.", "sym": ["Fever","Chills","Fatigue"]},
    "TB": {"so": "TB waa xanuun sambabada ku dhaca.", "en": "Airborne bacterial infection.", "sym": ["Cough","Sweat","Fatigue"]},
    "Typhoid": {"so": "Typhoid waxaa keena Salmonella.", "en": "Typhoid fever bacteria.", "sym": ["Fever","Vomiting","Headache"]},
    "Cholera": {"so": "Koolera waa shuban biyood halis ah.", "en": "Severe diarrheal disease.", "sym": ["Diarrhea","Vomiting"]},
    "Measles": {"so": "Jadeeco waa rash leh.", "en": "Highly contagious viral disease.", "sym": ["Rash","Fever","Cough"]},
    "COVID-19": {"so": "Koronaha cusub.", "en": "Coronavirus disease.", "sym": ["Fever","Cough","Fatigue"]},
    "Diabetes": {"so": "Suka waa xanuun insulin la'aan ah.", "en": "Insulin-related condition.", "sym": ["Thirst","Fatigue"]},
    "Hypertension": {"so": "Cadaadis dhiig sare.", "en": "High blood pressure.", "sym": ["Headache","Dizziness"]},
    "Healthy": {"so": "Caafimaad buuxa.", "en": "No disease detected.", "sym": ["None"]}
}

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODELS_DIR / 'disease_model.pkl')
        le = joblib.load(MODELS_DIR / 'label_encoder.pkl')
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        return model, le, scaler
    except:
        return None, None, None

model, label_encoder, scaler = load_assets()

# ========== UI LAYOUT ==========
st.markdown("<h1 class='main-header'>🧬 SOMALI DISEASE AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>INTELLIGENT DIAGNOSTIC ASSISTANCE SYSTEM</p>", unsafe_allow_html=True)

# Sidebar Inputs
with st.sidebar:
    st.markdown("### 📋 Patient Profile")
    with st.container():
        age = st.number_input("Age", 1, 110, 25)
        temp = st.slider("Temperature (°C)", 34.0, 42.0, 37.0, 0.1)
    
    st.markdown("---")
    st.markdown("### 🤒 Symptoms")
    # Grouping symptoms for better UX
    col_a, col_b = st.columns(2)
    with col_a:
        headache = st.checkbox("Headache")
        cough = st.checkbox("Cough")
        vomiting = st.checkbox("Vomiting")
        diarrhea = st.checkbox("Diarrhea")
    with col_b:
        fatigue = st.checkbox("Fatigue")
        rash = st.checkbox("Rash")
        bleeding = st.checkbox("Bleeding")
        anaemia = st.checkbox("Anaemia")
    
    st.markdown("---")
    predict_btn = st.button("RUN DIAGNOSIS 🔍")

# Main Logic
if predict_btn:
    if model:
        # 1. Processing State
        with st.spinner("Analyzing physiological markers..."):
            time.sleep(1.2) # Adds "AI Weight" to the experience
            
            # 2. Prediction
            features = [age, temp, int(headache), int(cough), int(vomiting), 
                        int(diarrhea), int(fatigue), int(rash), int(bleeding), int(anaemia)]
            
            scaled_data = scaler.transform([features])
            prediction = model.predict(scaled_data)
            probs = model.predict_proba(scaled_data)[0]
            
            disease_name = label_encoder.inverse_transform(prediction)[0]
            confidence = max(probs) * 100

        # 3. Display Results
        st.markdown("### 🧪 Diagnostic Report")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>Detected Condition</div>
                <div class='metric-value'>{disease_name}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>Confidence Level</div>
                <div class='metric-value'>{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # 4. Disease Details
        st.write("")
        with st.expander("📝 Clinical Details & Somali Context", expanded=True):
            info = DISEASE_INFO.get(disease_name, DISEASE_INFO["Healthy"])
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"**English:** {info['en']}")
            with c2:
                st.success(f"**Af-Soomaali:** {info['so']}")
            
            st.markdown("**Typical Symptoms:** " + ", ".join(info['sym']))

        # 5. Data Visualization
        st.markdown("### 📊 Statistical Probability")
        prob_df = pd.DataFrame({
            'Disease': label_encoder.classes_,
            'Probability': probs * 100
        }).sort_values('Probability', ascending=False)

        chart = alt.Chart(prob_df).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X('Probability:Q', title='Confidence %'),
            y=alt.Y('Disease:N', sort='-x', title=''),
            color=alt.condition(
                alt.datum.Probability > 50,
                alt.value('#66ccff'), alt.value('#335d68')
            ),
            tooltip=['Disease', 'Probability']
        ).properties(height=350).configure_axis(
            labelColor='#a9b3c1', titleColor='#a9b3c1', grid=False
        ).configure_view(strokeOpacity=0)

        st.altair_chart(chart, use_container_width=True)
    else:
        st.error("Error: Model files not found. Please train the model.")

else:
    # Empty state / Landing
    st.markdown("""
    <div class='card' style='text-align: center;'>
        <h2 style='color: #66ccff;'>Ready for Analysis</h2>
        <p>Please enter patient symptoms in the sidebar and click 'Run Diagnosis' to begin.</p>
        <div style='display: flex; justify-content: space-around; margin-top: 30px; opacity: 0.6;'>
            <div>🌡️ Temp Check</div>
            <div>🧬 DNA Scanned</div>
            <div>📊 Probabilistic ML</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("© 2025 Somali Disease Detection AI | For Research & Educational Use Only")