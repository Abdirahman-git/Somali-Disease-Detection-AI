# train_model.py - UPDATED VERSION

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import from config
from config import DATA_FILE, MODELS_DIR, TEST_SIZE, RANDOM_STATE

print("="*60)
print("SOMALI DISEASE DETECTION AI - MODEL TRAINING")
print("="*60)

def load_and_preprocess_data():
    """Load and preprocess the data"""
    print(f"\n📂 Loading data from: {DATA_FILE}")
    
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at: {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)
    print(f"   ✅ Loaded {len(df)} records")
    print(f"   Features: {list(df.columns)}")
    
    # Separate features and target
    X = df.drop('disease', axis=1)
    y = df['disease']
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"   Diseases to classify: {list(label_encoder.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    print("\n🤖 Training ML models...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss'),
        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'accuracy': accuracy,
            'model': model,
            'report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        trained_models[name] = model
        
        print(f"   ✅ {name} Accuracy: {accuracy:.4f}")
    
    return results, trained_models

def save_best_model(models_dict, label_encoder, scaler):
    """Save the best performing model and preprocessing objects"""
    # Find best model
    best_model_name = max(models_dict, key=lambda x: models_dict[x]['accuracy'])
    best_model = models_dict[best_model_name]['model']
    best_accuracy = models_dict[best_model_name]['accuracy']
    
    print(f"\n🏆 Best model: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Save model and preprocessing objects
    joblib.dump(best_model, MODELS_DIR / 'disease_model.pkl')
    joblib.dump(label_encoder, MODELS_DIR / 'label_encoder.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    
    print(f"   💾 Model saved to: {MODELS_DIR / 'disease_model.pkl'}")
    print(f"   💾 Label encoder saved to: {MODELS_DIR / 'label_encoder.pkl'}")
    print(f"   💾 Scaler saved to: {MODELS_DIR / 'scaler.pkl'}")
    
    return best_model_name, best_accuracy

def plot_results(results, y_test, label_encoder):
    """Plot model comparison and confusion matrix"""
    print("\n📊 Generating visualizations...")
    
    # Model comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'model_comparison.png', dpi=100)
    print(f"   ✅ Saved: model_comparison.png")
    
    # Confusion matrix for best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    y_pred = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'confusion_matrix.png', dpi=100)
    print(f"   ✅ Saved: confusion_matrix.png")
    
    # Feature importance for tree-based models
    best_model = results[best_model_name]['model']
    try:
        if hasattr(best_model, 'feature_importances_'):
            feature_names = ['age', 'temperature', 'headache', 'cough', 'vomiting', 
                           'diarrhea', 'fatigue', 'rash', 'bleeding', 'anaemia']
            
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
            bars = plt.bar(range(len(importances)), importances[indices], color='#2ecc71')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Importance', fontsize=12)
            plt.tight_layout()
            plt.savefig(MODELS_DIR / 'feature_importance.png', dpi=100)
            print(f"   ✅ Saved: feature_importance.png")
    except:
        pass  # Some models don't have feature_importance

def main():
    """Main training pipeline"""
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, label_encoder, scaler = load_and_preprocess_data()
        
        # Train models
        results, trained_models = train_models(X_train, X_test, y_train, y_test)
        
        # Save best model
        best_model_name, best_accuracy = save_best_model(results, label_encoder, scaler)
        
        # Generate plots
        plot_results(results, y_test, label_encoder)
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best model: {best_model_name}")
        print(f"Accuracy: {best_accuracy:.4f}")
        print(f"\nNext steps:")
        print(f"1. Check the 'models/' folder for saved files")
        print(f"2. Run the web app: cd app && python app.py")
        print(f"3. Open browser to: http://localhost:5000")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure data/somali_synthetic_diseases.csv exists")
        print("2. Install dependencies: pip install scikit-learn xgboost joblib")
        print("3. Check file permissions")

if __name__ == "__main__":
    main()