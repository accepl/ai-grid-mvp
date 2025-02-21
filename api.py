import os
import joblib
import pandas as pd
import random
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Define paths for saving and loading models
MODEL_PATH_GRID = 'models/rf_grid_model.pkl'
MODEL_PATH_BESS = 'models/rf_bess_model.pkl'

# Create the models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load or train models
def load_models():
    if os.path.exists(MODEL_PATH_GRID) and os.path.exists(MODEL_PATH_BESS):
        print("Models found. Loading...")
        grid_model = joblib.load(MODEL_PATH_GRID)
        bess_model = joblib.load(MODEL_PATH_BESS)
    else:
        print("Models not found. Training new models...")
        grid_model, bess_model = train_models()
    
    return grid_model, bess_model

# Train models (This is just a placeholder for actual model training)
def train_models():
    # Create mock data for training (replace with your actual data)
    grid_data = pd.DataFrame({
        'previous_demand': [1500, 1550, 1600, 1580, 1550, 1530],
        'weather_conditions': [32, 30, 28, 31, 33, 32],
        'time_of_day': ['16:00'] * 6,
        'day_of_week': ['Wednesday'] * 6,
        'special_event': ['No'] * 6,
        'economic_index': ['5%'] * 6,
        'season': ['Winter'] * 6
    })
    bess_data = pd.DataFrame({
        'previous_battery_level': [0.8, 0.75, 0.7, 0.85, 0.9, 0.88],
        'solar_generation': [500, 520, 480, 450, 510, 495],
        'wind_speed': [15, 16, 14, 15, 17, 13],
        'time_of_day': ['16:00'] * 6,
        'special_event': ['No'] * 6
    })
    grid_target = [1600, 1550, 1600, 1580, 1540, 1530]  # Mock target values for grid
    bess_target = [0.75, 0.8, 0.7, 0.85, 0.9, 0.88]  # Mock target values for BESS

    # Train grid model
    grid_model = RandomForestRegressor()
    grid_model.fit(grid_data, grid_target)
    joblib.dump(grid_model, MODEL_PATH_GRID)

    # Train BESS model
    bess_model = RandomForestRegressor()
    bess_model.fit(bess_data, bess_target)
    joblib.dump(bess_model, MODEL_PATH_BESS)

    print("Models trained and saved.")
    return grid_model, bess_model

# Predict grid load
@app.route('/predict/grid', methods=['POST'])
def predict_grid():
    try:
        # Get the data from the POST request body
        data = request.get_json()

        # Validate input data structure
        if 'previous_demand' not in data or 'weather_conditions' not in data:
            return jsonify({"error": "'previous_demand' and 'weather_conditions' are required fields."}), 400
        
        previous_demand = data['previous_demand']
        weather_conditions = data['weather_conditions']

        if len(previous_demand) != len(weather_conditions):
            return jsonify({"error": "Mismatch in input array lengths for 'previous_demand' and 'weather_conditions'."}), 400

        # Optional data with fallback values
        time_of_day = data.get('time_of_day', "00:00")
        day_of_week = data.get('day_of_week', "Unknown")
        special_event = data.get('special_event', "No")
        economic_index = data.get('economic_index', "0%")
        season = data.get('season', "Unknown")

        # Prepare data for the model prediction
        grid_data = pd.DataFrame({
            'previous_demand': previous_demand,
            'weather_conditions': weather_conditions,
            'time_of_day': [time_of_day] * len(previous_demand),
            'day_of_week': [day_of_week] * len(previous_demand),
            'special_event': [special_event] * len(previous_demand),
            'economic_index': [economic_index] * len(previous_demand),
            'season': [season] * len(previous_demand)
        })

        # Load the trained grid model
        grid_model, _ = load_models()

        # Predict grid demand
        prediction = grid_model.predict(grid_data)

        # Calculate the Mean Squared Error (MSE) of the predictions
        mse = mean_squared_error([1600, 1550, 1600, 1580, 1540, 1530], prediction)  # Mock true values

        return jsonify({
            "predictions": prediction.tolist(),
            "MSE": mse
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predict BESS charge level
@app.route('/predict/bess', methods=['POST'])
def predict_bess():
    try:
        # Get the data from the POST request body
        data = request.get_json()

        # Validate input data structure
        if 'previous_battery_level' not in data or 'solar_generation' not in data or 'wind_speed' not in data:
            return jsonify({"error": "'previous_battery_level', 'solar_generation', and 'wind_speed' are required fields."}), 400

        previous_battery_level = data['previous_battery_level']
        solar_generation = data['solar_generation']
        wind_speed = data['wind_speed']

        if len(previous_battery_level) != len(solar_generation) or len(solar_generation) != len(wind_speed):
            return jsonify({"error": "Mismatch in input array lengths for 'previous_battery_level', 'solar_generation', and 'wind_speed'."}), 400

        # Optional data with fallback values
        time_of_day = data.get('time_of_day', "00:00")
        special_event = data.get('special_event', "No")

        # Prepare data for the model prediction
        bess_data = pd.DataFrame({
            'previous_battery_level': previous_battery_level,
            'solar_generation': solar_generation,
            'wind_speed': wind_speed,
            'time_of_day': [time_of_day] * len(previous_battery_level),
            'special_event': [special_event] * len(previous_battery_level)
        })

        # Load the trained BESS model
        _, bess_model = load_models()

        # Predict BESS charge levels
        prediction = bess_model.predict(bess_data)

        # Calculate the Mean Squared Error (MSE) of the predictions
        mse = mean_squared_error([0.75, 0.8, 0.7, 0.85, 0.9, 0.88], prediction)  # Mock true values

        return jsonify({
            "predictions": prediction.tolist(),
            "MSE": mse
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
