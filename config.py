import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
APP_DIR = BASE_DIR / "app"

# Data file
DATA_FILE = DATA_DIR / "somali_synthetic_diseases.csv"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, APP_DIR]:
    dir_path.mkdir(exist_ok=True)

print(f"✅ Config loaded. Data file path: {DATA_FILE}")
print(f"✅ Directories checked/created.")