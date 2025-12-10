from flask import Flask, render_template_string
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
    <h1>Somali Disease Detection AI</h1>
    <p>Flask is working! 🎉</p>
    <p>Next steps:</p>
    <ol>
        <li>Train model: python train_model.py</li>
        <li>Check models folder has .pkl files</li>
        <li>Run main app: python app.py</li>
    </ol>
    ''')
    
if __name__ == '__main__':
    print("✅ Flask test app running at http://localhost:5001")
    app.run(debug=True, port=5001)