import streamlit as st
import requests
import pandas as pd

st.title("AI Grid Load Forecasting")

# Fetch predictions
response = requests.get("http://localhost:5000/predict?hours=10")
data = response.json()
predictions = pd.DataFrame({"Hour": range(1, 11), "Predicted Load": data["predictions"]})

# Display data
st.line_chart(predictions.set_index("Hour"))
st.write(predictions)

# Retrain Model Button
if st.button("Retrain AI Model"):
    response = requests.post("http://localhost:5000/retrain")
    st.write(response.json())
