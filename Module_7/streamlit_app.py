import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Travel & Hotel Insights", layout="wide")

st.title("🌍 Travel & Hotel Recommendation Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    flights = pd.read_csv('flights.csv')
    hotels = pd.read_csv('hotels.csv')
    users = pd.read_csv('users.csv')
    return flights, hotels, users

try:
    flights, hotels, users = load_data()
    st.sidebar.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure CSV files are in the directory.")
    st.stop()

# --- Sidebar ---
menu = st.sidebar.selectbox("Choose a Section", ["EDA & Insights", "Hotel Recommendations", "Model Predictions"])

if menu == "EDA & Insights":
    st.header("📊 Data Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flight Price Distribution by Flight Type")
        fig, ax = plt.subplots()
        sns.boxplot(data=flights, x='flightType', y='price', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("User Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(users['age'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Top Flight Agencies by Volume")
    agency_counts = flights['agency'].value_counts()
    st.bar_chart(agency_counts)

elif menu == "Hotel Recommendations":
    st.header("🏨 Personalized Hotel Suggestions")
    
    user_id = st.number_input("Enter User Code", min_value=int(users['code'].min()), max_value=int(users['code'].max()), value=0)
    
    if st.button("Get Recommendations"):
        # Recommendation Logic
        user_hotel = hotels.pivot_table(index='userCode', columns='name', values='total', aggfunc='sum', fill_value=0)
        
        if user_id not in user_hotel.index:
            st.warning(f"No booking history found for User {user_id}. Showing popular hotels instead.")
            st.write(hotels['name'].value_counts().head(5))
        else:
            user_sim = cosine_similarity(user_hotel)
            user_sim_df = pd.DataFrame(user_sim, index=user_hotel.index, columns=user_hotel.index)
            
            similar_users = user_sim_df[user_id].sort_values(ascending=False).iloc[1:11].index.tolist()
            hotels_seen = set(user_hotel.loc[user_id][user_hotel.loc[user_id] > 0].index)
            
            recs = (hotels[hotels['userCode'].isin(similar_users)]['name']
                    .value_counts()
                    .drop(labels=[h for h in hotels_seen if h in hotels['name'].value_counts().index], errors='ignore')
                    .head(5))
            
            if recs.empty:
                st.info("No new hotels to recommend based on similar users. Try these popular ones:")
                st.write(hotels['name'].value_counts().head(3))
            else:
                st.success(f"Top recommendations for User {user_id}:")
                st.write(recs)

elif menu == "Model Predictions":
    st.header("🔮 Real-time Predictions")
    st.info("Note: Ensure the Flask API is running on http://localhost:5001")
    
    task = st.radio("Select Prediction Task", ["Flight Price", "User Gender"])
    
    if task == "Flight Price":
        with st.form("price_form"):
            col1, col2 = st.columns(2)
            with col1:
                f_from = st.selectbox("From", flights['from'].unique())
                f_to = st.selectbox("To", flights['to'].unique())
                f_type = st.selectbox("Flight Type", flights['flightType'].unique())
                f_agency = st.selectbox("Agency", flights['agency'].unique())
            with col2:
                f_time = st.number_input("Time (hours)", value=2.0)
                f_dist = st.number_input("Distance", value=500.0)
                f_date = st.date_input("Date")
                f_ppk = st.number_input("Price per KM", value=2.0)
                f_tdr = st.number_input("Time Dist Ratio", value=0.004)
            
            submit = st.form_submit_button("Predict Price")
            
            if submit:
                payload = {
                    "from": f_from, "to": f_to, "flightType": f_type, "agency": f_agency,
                    "time": f_time, "distance": f_dist, "date": str(f_date),
                    "price_per_km": f_ppk, "time_dist": f_tdr
                }
                try:
                    res = requests.post("http://localhost:5001/predict_price", json=payload)
                    st.success(f"Predicted Price: ${res.json()['predicted_price']:.2f}")
                except Exception as e:
                    st.error(f"API Error: {e}")

    else:
        with st.form("gender_form"):
            f_type = st.selectbox("Flight Type", flights['flightType'].unique())
            f_agency = st.selectbox("Agency", flights['agency'].unique())
            f_time = st.number_input("Time (hours)", value=2.0)
            f_dist = st.number_input("Distance", value=500.0)
            f_date = st.date_input("Date")
            f_price = st.number_input("Price Paid", value=1000.0)
            
            submit = st.form_submit_button("Predict Gender")
            
            if submit:
                payload = {
                    "flightType": f_type, "agency": f_agency, "time": f_time,
                    "distance": f_dist, "date": str(f_date), "price": f_price
                }
                try:
                    res = requests.post("http://localhost:5001/predict_gender", json=payload)
                    st.success(f"Predicted Gender: {res.json()['predicted_gender']}")
                except Exception as e:
                    st.error(f"API Error: {e}")
