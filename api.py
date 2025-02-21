from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

MODEL_PATH_GRID = "models/rf_grid_model.pkl"
MODEL_PATH_BESS = "models/rf_bess_model.pkl"

# Load models (or train if they don't exist)
def load_or_train_models():
    try:
        global rf_model_grid, rf_model_bess
        rf_model_grid = joblib.load(MODEL_PATH_GRID)
        rf_model_bess = joblib.load(MODEL_PATH_BESS)
        print("Models loaded successfully")
    except (FileNotFoundError, IOError):
        print("Models not found. Training new models...")
        train_models()

def train_models():
    # Example grid data for training
    grid_data = pd.DataFrame({
        "previous_demand": [1500, 1550, 1600, 1580, 1550, 1530],
        "weather_conditions": [32, 30, 28, 31, 33, 29],
        "time_of_day": [16, 14, 18, 15, 12, 13],
        "day_of_week": [3, 4, 5, 6, 0, 1],
        "special_event": [0, 0, 1, 0, 1, 0],
        "economic_index": [5.0, 5.1, 5.0, 4.8, 5.2, 5.1],
        "season": [1, 2, 1, 2, 1, 1]  # Winter = 1, Summer = 2, etc.
    })

    bess_data = grid_data.copy()  # Use the same data for BESS as an example

    X_grid = grid_data.drop("previous_demand", axis=1)  # Features
    y_grid = grid_data["previous_demand"]  # Target

    X_bess = bess_data.drop("previous_demand", axis=1)
    y_bess = bess_data["previous_demand"]

    global rf_model_grid, rf_model_bess
    rf_model_grid = RandomForestRegressor(n_estimators=100)
    rf_model_grid.fit(X_grid, y_grid)

    rf_model_bess = RandomForestRegressor(n_estimators=100)
    rf_model_bess.fit(X_bess, y_bess)

    # Save models
    joblib.dump(rf_model_grid, MODEL_PATH_GRID)
    joblib.dump(rf_model_bess, MODEL_PATH_BESS)

# Preprocess the input data
def preprocess_input(data):
    # Convert time_of_day to numeric value (if it's not already)
    data["time_of_day"] = int(data["time_of_day"][0])  # assuming a single hour is passed

    # Convert day_of_week to numeric value
    data["day_of_week"] = int(data["day_of_week"][0])  # assuming it's an integer

    # Convert special_event to binary (0 for No, 1 for Yes)
    data["special_event"] = int(data["special_event"][0])

    # Convert economic_index to numeric value (percentage)
    data["economic_index"] = float(data["economic_index"][0])

    # Convert season to numeric value (e.g., Winter = 1, Summer = 2)
    data["season"] = int(data["season"][0])

    return data

@app.route('/predict/grid', methods=['POST'])
def predict_grid():
    data = request.get_json()

    # Preprocess input data
    data = preprocess_input(data)

    # Convert to DataFrame for model input
    input_data = pd.DataFrame([data])

    # Make prediction with the grid model
    prediction = rf_model_grid.predict(input_data)
    return jsonify({"predicted_demand": prediction[0]})

@app.route('/predict/bess', methods=['POST'])
def predict_bess():
    data = request.get_json()

    # Preprocess input data
    data = preprocess_input(data)

    # Convert to DataFrame for model input
    input_data = pd.DataFrame([data])

    # Make prediction with the BESS model
    prediction = rf_model_bess.predict(input_data)
    return jsonify({"predicted_bess_demand": prediction[0]})

if __name__ == '__main__':
    load_or_train_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
