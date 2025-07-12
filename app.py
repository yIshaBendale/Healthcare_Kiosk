from flask import Flask, request, jsonify
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load model and features
model = pickle.load(open('model.pkl', 'rb'))
with open('features.json') as f:
    data = json.load(f)
    SELECTED_FEATURES = data['selected_features']
    disease_dict = data['disease_mapping']

@app.route('/predict', methods=['POST'])
def predict():
    user_symptoms = request.json.get('symptoms', [])
    input_vector = [1 if symptom in user_symptoms else 0 
                   for symptom in SELECTED_FEATURES]
    
    proba = model.predict_proba([input_vector])[0]
    disease_idx = str(np.argmax(proba) + 1)  # Match dictionary keys
    
    return jsonify({
        'disease': disease_dict.get(disease_idx, "Unknown"),
        'confidence': float(np.max(proba))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)