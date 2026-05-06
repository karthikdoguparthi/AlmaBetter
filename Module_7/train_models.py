import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

def train_models():
    print("Loading datasets...")
    try:
        flights = pd.read_csv('flights.csv')
        hotels = pd.read_csv('hotels.csv')
        users = pd.read_csv('users.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure flights.csv, hotels.csv, and users.csv are in the current directory.")
        return

    # --- Preprocessing for Regression ---
    print("Preprocessing for Regression...")
    le_dict = {}
    for col in ['from', 'to', 'flightType', 'agency']:
        le = LabelEncoder()
        flights[f'{col}_enc'] = le.fit_transform(flights[col])
        le_dict[col] = le

    flights['date'] = pd.to_datetime(flights['date'])
    flights['month'] = flights['date'].dt.month
    flights['dayofweek'] = flights['date'].dt.dayofweek
    flights['quarter'] = flights['date'].dt.quarter
    
    flights['price_per_km'] = flights['price'] / flights['distance']
    flights['time_dist'] = flights['time'] / flights['distance']

    FEATURE_COLS = ['flightType_enc', 'agency_enc', 'from_enc', 'to_enc', 
                    'time', 'distance', 'month', 'dayofweek', 'quarter', 
                    'price_per_km', 'time_dist']
    
    X = flights[FEATURE_COLS]
    y = flights['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("Training Flight Price Regressor...")
    best_gb = GradientBoostingRegressor(n_estimators=100, max_depth=7, learning_rate=0.2, subsample=1.0, random_state=42)
    best_gb.fit(X_train, y_train)

    joblib.dump(best_gb, 'flight_price_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    joblib.dump(le_dict, 'label_encoders.joblib')
    print("✅ Regression models and encoders saved.")

    # --- Preprocessing for Classification ---
    print("Preprocessing for Classification...")
    le_gender = LabelEncoder()
    users['gender_enc'] = le_gender.fit_transform(users['gender'].fillna('none'))
    
    flights_users = flights.merge(users[['code', 'gender_enc']], left_on='userCode', right_on='code', how='left')
    merged_clf = flights_users.dropna(subset=['gender_enc'])
    
    CLF_FEATURES = ['flightType_enc', 'agency_enc', 'time', 'distance', 'month', 'dayofweek', 'price']
    X_clf = merged_clf[CLF_FEATURES]
    y_clf = merged_clf['gender_enc']

    X_tr_clf, X_te_clf, y_tr_clf, y_te_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    sm = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr_clf, y_tr_clf)

    print("Training Gender Classifier...")
    best_clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    best_clf.fit(X_tr_res, y_tr_res)

    joblib.dump(best_clf, 'gender_clf_model.joblib')
    joblib.dump(le_gender, 'gender_label_encoder.joblib')
    print("✅ Gender classifier saved.")

if __name__ == "__main__":
    train_models()
