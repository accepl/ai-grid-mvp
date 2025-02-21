import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Sample Data for Grid Optimization (Replace with actual data or modify as needed)
grid_data = {
    'hour': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'load_demand': [200, 210, 230, 220, 210, 205, 230, 240, 250, 260],
    'wind_power': [10, 15, 25, 30, 28, 20, 15, 12, 10, 15],
    'solar_power': [50, 55, 70, 85, 90, 110, 130, 140, 150, 160]
}

# Convert to DataFrame
df_grid = pd.DataFrame(grid_data)

# Feature engineering
df_grid['net_demand'] = df_grid['load_demand'] - (df_grid['wind_power'] + df_grid['solar_power'])

# Grid Optimization Model (Example)
def optimize_grid(df):
    # Here we just predict net demand for now, you can replace it with actual optimization logic
    X = df[['wind_power', 'solar_power']]
    y = df['net_demand']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    # Predictions
    df['optimized_demand'] = model.predict(X)
    
    return df

# BESS Model (Battery Energy Storage System)
def bess_model(df):
    # Just a dummy model for now, predicting the energy storage required based on grid demand
    df['energy_storage'] = df['optimized_demand'] * 0.2  # Example 20% of net demand goes to BESS
    return df

# Prediction Endpoint for Grid Optimization
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        hour = data.get('hour')
        wind_power = data.get('wind_power')
        solar_power = data.get('solar_power')
        
        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'hour': [hour],
            'wind_power': [wind_power],
            'solar_power': [solar_power]
        })
        
        # Use the models to get predictions
        optimized_data = optimize_grid(input_data)
        bess_data = bess_model(optimized_data)
        
        response = {
            'optimized_demand': bess_data['optimized_demand'][0],
            'energy_storage': bess_data['energy_storage'][0]
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running'})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
