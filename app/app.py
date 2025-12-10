from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import MODELS_DIR

app = Flask(__name__)

# Load trained model and preprocessing objects
try:
    model = joblib.load(MODELS_DIR / 'disease_model.pkl')
    label_encoder = joblib.load(MODELS_DIR / 'label_encoder.pkl')
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please train the model first by running: python train_model.py")
    model = None
    label_encoder = None
    scaler = None

# Disease descriptions
DISEASE_INFO = {
    'Malaria': {
        'so': 'Malariya waa cudur ku faafa iyadoo ay sabab u tahay cayayaanka.',
        'en': 'Malaria is a mosquito-borne infectious disease.',
        'symptoms': ['High fever', 'Headache', 'Fatigue', 'Chills']
    },
    'TB': {
        'so': 'Tuberculosis waa cudur ku faafa ah sambabada neefmaraqa.',
        'en': 'Tuberculosis is a bacterial infection that spreads through air.',
        'symptoms': ['Cough', 'Fatigue', 'Weight loss', 'Night sweats']
    },
    'Typhoid': {
        'so': 'Typhoid waa cudur sabab u ah bacillada Salmonella Typhi.',
        'en': 'Typhoid fever is caused by Salmonella Typhi bacteria.',
        'symptoms': ['High fever', 'Headache', 'Diarrhea', 'Vomiting', 'Weakness']
    },
    'Cholera': {
        'so': 'Koolera waa cudur baafitaal ah oo sabab u ah bacillada Vibrio cholerae.',
        'en': 'Cholera is an acute diarrheal illness caused by Vibrio cholerae.',
        'symptoms': ['Diarrhea', 'Vomiting', 'Dehydration']
    },
    'Measles': {
        'so': 'Jadeeco waa cudur ku faafa ah oo sabab u tahay firfircooniga.',
        'en': 'Measles is a highly contagious viral disease.',
        'symptoms': ['Fever', 'Cough', 'Rash', 'Fatigue', 'Runny nose']
    },
    'COVID-19': {
        'so': 'COVID-19 waa cudur ku faafa ah oo sabab u tahay koronafayraska.',
        'en': 'COVID-19 is a contagious disease caused by coronavirus.',
        'symptoms': ['Fever', 'Cough', 'Fatigue', 'Headache', 'Loss of taste/smell']
    },
    'Diabetes': {
        'so': 'Suka waa cudur ka dhasha hormoonka insulin.',
        'en': 'Diabetes is a chronic condition affecting insulin.',
        'symptoms': ['Fatigue', 'Increased thirst', 'Frequent urination', 'Blurred vision']
    },
    'Hypertension': {
        'so': 'Cadaadis dhiig waa heerka sare ee cadaadiska dhiigga.',
        'en': 'Hypertension is high blood pressure condition.',
        'symptoms': ['Often asymptomatic', 'Headache', 'Fatigue', 'Dizziness']
    },
    'Healthy': {
        'so': 'Caafimaad wanaagsan, ma jiraan calaamadaha cudurka.',
        'en': 'Good health, no disease symptoms present.',
        'symptoms': ['No symptoms']
    }
}

@app.route('/')
def home():
    """Render home page"""
    diseases = label_encoder.classes_ if label_encoder else []
    return render_template('index.html', diseases=diseases)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None:
        return render_template('error.html', 
                             error="Model not loaded. Please train the model first.")
    
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Convert to features array (ensure all features are present)
        features = [
            float(data.get('age', 30)),
            float(data.get('temperature', 37)),
            int(data.get('headache', 0)),
            int(data.get('cough', 0)),
            int(data.get('vomiting', 0)),
            int(data.get('diarrhea', 0)),
            int(data.get('fatigue', 0)),
            int(data.get('rash', 0)),
            int(data.get('bleeding', 0)),
            int(data.get('anaemia', 0))
        ]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        # Decode prediction
        disease = label_encoder.inverse_transform(prediction)[0]
        
        # Get probability
        prob = max(probabilities[0]) * 100
        
        # Get disease info
        disease_info = DISEASE_INFO.get(disease, {
            'so': 'Macluumaad aan la helin',
            'en': 'Information not available',
            'symptoms': ['Unknown']
        })
        
        return render_template('result.html',
                             disease=disease,
                             probability=f"{prob:.1f}%",
                             description_so=disease_info['so'],
                             description_en=disease_info['en'],
                             symptoms=disease_info['symptoms'])
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Train model first.'}), 500
    
    try:
        data = request.get_json()
        
        # Convert to features array
        features = [
            float(data.get('age', 30)),
            float(data.get('temperature', 37)),
            int(data.get('headache', 0)),
            int(data.get('cough', 0)),
            int(data.get('vomiting', 0)),
            int(data.get('diarrhea', 0)),
            int(data.get('fatigue', 0)),
            int(data.get('rash', 0)),
            int(data.get('bleeding', 0)),
            int(data.get('anaemia', 0))
        ]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        # Decode prediction
        disease = label_encoder.inverse_transform(prediction)[0]
        
        # Get probabilities for all classes
        prob_dict = {}
        for i, cls in enumerate(label_encoder.classes_):
            prob_dict[cls] = float(probabilities[0][i] * 100)
        
        disease_info = DISEASE_INFO.get(disease, {
            'so': 'Macluumaad aan la helin',
            'en': 'Information not available',
            'symptoms': ['Unknown']
        })
        
        return jsonify({
            'prediction': disease,
            'probabilities': prob_dict,
            'disease_info': disease_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model else 'model_not_loaded',
        'model_loaded': model is not None,
        'diseases': label_encoder.classes_.tolist() if label_encoder else []
    })

if __name__ == '__main__':
    print("🌐 Starting Somali Disease Detection AI Web App...")
    print("   Open: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)