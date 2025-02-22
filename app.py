import streamlit as st
import requests

st.title("⚡ AI Energy Predictions Dashboard")

# Backend API URL (Update if hosted)
API_BASE_URL = "http://localhost:5000"

# Button to retrain AI model
if st.button("Retrain AI Model"):
    response = requests.post(f"{API_BASE_URL}/retrain")
    if response.status_code == 200:
        st.success("✅ Model retrained successfully!")
    else:
        st.error(f"❌ Model retraining failed: {response.json().get('error', 'Unknown error')}")

st.write("### 📊 Grid Load Forecast")

# Input fields for prediction
previous_demand = st.number_input("Previous Demand (MW)", min_value=100, max_value=5000, value=1500)
weather_conditions = st.slider("Weather Conditions (°C)", min_value=-10, max_value=50, value=30)
time_of_day = st.slider("Time of Day (Hour)", min_value=0, max_value=23, value=16)
special_event = st.selectbox("Special Event", ["No", "Yes"])
economic_index = st.slider("Economic Index", min_value=1, max_value=10, value=5)

# Prediction Button
if st.button("Predict Grid Load"):
    payload = {
        "previous_demand": previous_demand,
        "weather_conditions": weather_conditions,
        "time_of_day": time_of_day,
        "special_event": special_event,
        "economic_index": economic_index
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict/grid", json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"🔮 Predicted Grid Load: **{prediction:.2f} MW**")
        else:
            st.error(f"❌ Prediction failed: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"🚨 API Connection Error: {str(e)}")

st.write("### 🔋 BESS Storage Forecast")

# Prediction Button for BESS
if st.button("Predict BESS Storage"):
    try:
        response = requests.post(f"{API_BASE_URL}/predict/bess", json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"🔮 Predicted BESS Storage Requirement: **{prediction:.2f} MWh**")
        else:
            st.error(f"❌ Prediction failed: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"🚨 API Connection Error: {str(e)}")
