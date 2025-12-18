import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import os

# ========== FIXED: CORRECT PATH TO CONFIG ==========
# Get the absolute path to the parent directory (Somali-Disease-Detection folder)
current_file = Path(__file__).resolve()  # This is streamlit_app.py
project_root = current_file.parent.parent  # Go up two levels to Somali-Disease-Detection folder

# Add the project root to Python path
sys.path.append(str(project_root))

# Now import from config - it should work!
try:
    from config import MODELS_DIR, DATA_FILE
    print(f"✅ Config loaded successfully!")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Models dir: {MODELS_DIR}")
except ImportError as e:
    st.error(f"❌ Failed to import config: {e}")
    st.error(f"Looking in: {project_root}")
    # Create fallback paths
    MODELS_DIR = project_root / "models"
    DATA_FILE = project_root / "data" / "somali_synthetic_diseases.csv"

# ========== REST OF YOUR APP CODE CONTINUES HERE ==========
# Set page config
st.set_page_config(
    page_title="Somali Disease Detection AI",
    page_icon="🏥",
    layout="wide"
)
# ... continue with the rest of your app code ...