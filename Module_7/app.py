from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models and encoders
try:
    price_model = joblib.load('flight_price_model.joblib')
    price_scaler = joblib.load('feature_scaler.joblib')
    le_dict = joblib.load('label_encoders.joblib')
    gender_model = joblib.load('gender_clf_model.joblib')
    le_gender = joblib.load('gender_label_encoder.joblib')
except Exception as e:
    print(f"Error loading models: {e}. Run train_models.py first.")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Travel Prediction API is running!"})

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        
        # Extract and preprocess features
        # Expected: from, to, flightType, agency, time, distance, date, price_per_km, time_dist
        
        df = pd.DataFrame([data])
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        # Encoding
        for col in ['from', 'to', 'flightType', 'agency']:
            df[f'{col}_enc'] = le_dict[col].transform(df[col])
            
        FEATURE_COLS = ['flightType_enc', 'agency_enc', 'from_enc', 'to_enc', 
                        'time', 'distance', 'month', 'dayofweek', 'quarter', 
                        'price_per_km', 'time_dist']
        
        X = df[FEATURE_COLS]
        X_scaled = price_scaler.transform(X)
        
        prediction = price_model.predict(X_scaled)
        
        return jsonify({
            "predicted_price": float(prediction[0]),
            "currency": "USD"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict_gender', methods=['POST'])
def predict_gender():
    try:
        data = request.get_json()
        
        # Expected: flightType, agency, time, distance, date, price
        df = pd.DataFrame([data])
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['dayofweek'] = df['date'].dt.dayofweek
        
        # Encoding
        for col in ['flightType', 'agency']:
            df[f'{col}_enc'] = le_dict[col].transform(df[col])
            
        CLF_FEATURES = ['flightType_enc', 'agency_enc', 'time', 'distance', 'month', 'dayofweek', 'price']
        
        X = df[CLF_FEATURES]
        prediction = gender_model.predict(X)
        gender = le_gender.inverse_transform(prediction)
        
        return jsonify({
            "predicted_gender": gender[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
