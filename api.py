from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

app = Flask(__name__)

# Model Paths (Update these paths as necessary)
MODEL_PATH_GRID = 'models/rf_grid_model.pkl'
MODEL_PATH_BESS = 'models/rf_bess_model.pkl'

# Create sample data for training (Placeholder, you should replace it with actual data)
def generate_sample_data():
    # Random data for demonstration, replace with actual data
    grid_data = {
        'time_of_day': [12, 14, 16, 18, 20],
        'weather': [30, 28, 35, 33, 31],
        'previous_demand': [1500, 1550, 1600, 1580, 1590],
        'economic_index': [2.5, 3.0, 3.1, 3.2, 3.3],
    }
    bess_data = {
        'current_soc': [60, 65, 70, 80, 85],
        'grid_demand': [1600, 1550, 1580, 1700, 1650],
        'energy_price': [0.10, 0.12, 0.11, 0.13, 0.09],
        'solar_generation': [500, 600, 550, 650, 700],
    }
    return grid_data, bess_data

# Train models (you can call this once when starting the app)
def train_models():
    # Grid Power Demand Model
    grid_data, bess_data = generate_sample_data()
    
    # Convert data to DataFrame
    df_grid = pd.DataFrame(grid_data)
    df_bess = pd.DataFrame(bess_data)
    
    # Grid Model: Random Forest for Power Demand Prediction
    X_grid = df_grid.drop(columns=['previous_demand'])
    y_grid = df_grid['previous_demand']
    X_train, X_test, y_train, y_test = train_test_split(X_grid, y_grid, test_size=0.2, random_state=42)
    rf_model_grid = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_grid.fit(X_train, y_train)
    
    # Evaluate Grid Model
    grid_preds = rf_model_grid.predict(X_test)
    mse_grid = mean_squared_error(y_test, grid_preds)
    print(f"Grid Model MSE: {mse_grid}")

    # Battery Energy Storage System Model (BESS)
    X_bess = df_bess.drop(columns=['current_soc'])
    y_bess = df_bess['current_soc']
    X_train_bess, X_test_bess, y_train_bess, y_test_bess = train_test_split(X_bess, y_bess, test_size=0.2, random_state=42)
    rf_model_bess = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_bess.fit(X_train_bess, y_train_bess)
    
    # Evaluate BESS Model
    bess_preds = rf_model_bess.predict(X_test_bess)
    mse_bess = mean_squared_error(y_test_bess, bess_preds)
    print(f"BESS Model MSE: {mse_bess}")
    
    # Save the models
    joblib.dump(rf_model_grid, MODEL_PATH_GRID)
    joblib.dump(rf_model_bess, MODEL_PATH_BESS)
    
train_models()

# Load models (if they already exist, otherwise train models)
def load_models():
    try:
        rf_model_grid = joblib.load(MODEL_PATH_GRID)
        rf_model_bess = joblib.load(MODEL_PATH_BESS)
        return rf_model_grid, rf_model_bess
    except FileNotFoundError:
        print("Model files not found, retraining models...")
        train_models()
        return load_models()

rf_model_grid, rf_model_bess = load_models()

@app.route('/predict/grid-demand', methods=['POST'])
def predict_grid_demand():
    data = request.get_json()
    
    # Extract input features from the request data
    time_of_day = data['time_of_day']
    weather = data['weather']
    previous_demand = data['previous_demand']
    economic_index = data['economic_index']
    
    # Prepare data for prediction
    input_data = np.array([[time_of_day, weather, previous_demand, economic_index]])
    
    # Make prediction
    prediction = rf_model_grid.predict(input_data)
    
    return jsonify({
        'predicted_demand': prediction[0],
        'confidence_interval': {
            'low': prediction[0] - 100,
            'high': prediction[0] + 100
        }
    })

@app.route('/predict/bess-optimization', methods=['POST'])
def predict_bess_optimization():
    data = request.get_json()
    
    # Extract input features from the request data
    current_soc = data['current_soc']
    grid_demand = data['grid_demand']
    energy_price = data['energy_price']
    solar_generation = data['solar_generation']
    
    # Prepare data for prediction
    input_data = np.array([[grid_demand, energy_price, solar_generation]])
    
    # Make prediction
    prediction = rf_model_bess.predict(input_data)
    
    # Determine optimal action (Charge or Discharge)
    optimal_action = 'Discharge' if grid_demand > 1600 else 'Charge'
    
    return jsonify({
        'predicted_soc': prediction[0],
        'optimal_action': optimal_action,
        'predicted_energy_discharge': 300 if optimal_action == 'Discharge' else 0
    })

if __name__ == "__main__":
    app.run(debug=True)
