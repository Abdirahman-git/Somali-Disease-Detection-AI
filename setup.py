import subprocess
import sys

def install_requirements():
    """Install required packages"""
    requirements = [
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'joblib',
        'xgboost'
    ]
    
    print("Installing requirements...")
    for package in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("✅ All packages installed!")

def check_structure():
    """Check project structure"""
    import os
    from pathlib import Path
    
    print("\nChecking project structure...")
    required = [
        'data/somali_synthetic_diseases.csv',
        'config.py',
        'train_model.py'
    ]
    
    for file in required:
        path = Path(file)
        if path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
    
    return True

def main():
    print("="*50)
    print("SOMALI DISEASE DETECTION AI - SETUP")
    print("="*50)
    
    # Install requirements
    install_requirements()
    
    # Check structure
    check_structure()
    
    print("\n" + "="*50)
    print("SETUP COMPLETE! Next steps:")
    print("1. Train model: python train_model.py")
    print("2. Run app: cd app && streamlit run streamlit_app.py")
    print("="*50)

if __name__ == "__main__":
    main()