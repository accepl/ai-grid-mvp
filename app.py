import streamlit as st
import requests

st.title("AI Energy Predictions Dashboard")

if st.button("Retrain AI Model"):
    requests.post("http://localhost:5000/retrain")
    st.success("Model retrained!")

st.write("### Grid Load Forecast")
pred = requests.get("http://localhost:5000/predict?hours=10").json()
st.line_chart(pred["predictions"])
