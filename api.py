import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize Flask app
app = Flask(__name__)

# Define paths for saving the models
MODEL_PATH_GRID = 'models/rf_grid_model.pkl'
MODEL_PATH_BESS = 'models/rf_bess_model.pkl'

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Function to train models with more detailed features
def train_models():
    # Example training data (replace with actual data)
    grid_data = pd.DataFrame({
        'previous_demand': [1500, 1550, 1600, 1580, 1550, 1530],
        'weather_conditions': [32, 30, 28, 31, 33],
        'time_of_day': [16, 17, 18, 19, 20],
        'day_of_week': [1, 2, 3, 4, 5],  # 1 = Monday, 7 = Sunday
        'special_event': [0, 0, 1, 0, 0],  # 0 = No, 1 = Yes
        'economic_index': [5, 4.8, 5.1, 4.9, 5],
        'season': [1, 1, 2, 2, 3],  # 1 = Winter, 2 = Spring, 3 = Summer
        'target': [100, 105, 110, 108, 107]
    })
    bess_data = pd.DataFrame({
        'previous_demand': [1500, 1550, 1600, 1580, 1550, 1530],
        'weather_conditions': [32, 30, 28, 31, 33],
        'time_of_day': [16, 17, 18, 19, 20],
        'day_of_week': [1, 2, 3, 4, 5],
        'special_event': [0, 0, 1, 0, 0],
        'economic_index': [5, 4.8, 5.1, 4.9, 5],
        'season': [1, 1, 2, 2, 3],
        'target': [100, 105, 110, 108, 107]
    })

    # Features and Target for Grid and BESS models
    X_grid = grid_data.drop(columns=['target'])
    y_grid = grid_data['target']
    X_bess = bess_data.drop(columns=['target'])
    y_bess = bess_data['target']

    # Initialize models
    rf_model_grid = RandomForestRegressor()
    rf_model_bess = RandomForestRegressor()

    # Train the models
    print("Training Grid Model...")
    rf_model_grid.fit(X_grid, y_grid)
    print("Training BESS Model...")
    rf_model_bess.fit(X_bess, y_bess)

    # Evaluate models
    grid_mse = mean_squared_error(y_grid, rf_model_grid.predict(X_grid))
    bess_mse = mean_squared_error(y_bess, rf_model_bess.predict(X_bess))

    # Log the Mean Squared Error (MSE) for both models
    print(f"Grid Model MSE: {grid_mse}")
    print(f"BESS Model MSE: {bess_mse}")

    # Save the trained models
    joblib.dump(rf_model_grid, MODEL_PATH_GRID)
    joblib.dump(rf_model_bess, MODEL_PATH_BESS)

# Load models (or train them if they don't exist)
def load_models():
    global rf_model_grid, rf_model_bess
    if os.path.exists(MODEL_PATH_GRID) and os.path.exists(MODEL_PATH_BESS):
        print("Loading saved models...")
        rf_model_grid = joblib.load(MODEL_PATH_GRID)
        rf_model_bess = joblib.load(MODEL_PATH_BESS)
    else:
        print("Models not found. Training new models...")
        train_models()

# Initialize models when the app starts
load_models()

# Define prediction endpoint for Grid with more detailed features
@app.route('/predict/grid', methods=['POST'])
def predict_grid():
    data = request.get_json()
    # Extract detailed features from the incoming request
    try:
        features = [
            data['previous_demand'], 
            data['weather_conditions'], 
            data['time_of_day'], 
            data['day_of_week'], 
            data['special_event'], 
            data['economic_index'], 
            data['season']
        ]

        # Prepare the input data for prediction
        input_data = [
            [
                features['previous_demand'], 
                features['weather_conditions'], 
                features['time_of_day'], 
                features['day_of_week'], 
                features['special_event'], 
                features['economic_index'], 
                features['season']
            ]
        ]
        
        prediction = rf_model_grid.predict(input_data)

        return jsonify({
            'prediction': prediction.tolist(),
            'mse': mean_squared_error([features['target']], prediction), 
            'details': 'Grid Model prediction based on demand, weather, and other factors.'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Define prediction endpoint for BESS with more detailed features
@app.route('/predict/bess', methods=['POST'])
def predict_bess():
    data = request.get_json()
    # Extract detailed features from the incoming request
    try:
        features = [
            data['previous_demand'], 
            data['weather_conditions'], 
            data['time_of_day'], 
            data['day_of_week'], 
            data['special_event'], 
            data['economic_index'], 
            data['season']
        ]

        # Prepare the input data for prediction
        input_data = [
            [
                features['previous_demand'], 
                features['weather_conditions'], 
                features['time_of_day'], 
                features['day_of_week'], 
                features['special_event'], 
                features['economic_index'], 
                features['season']
            ]
        ]
        
        prediction = rf_model_bess.predict(input_data)

        return jsonify({
            'prediction': prediction.tolist(),
            'mse': mean_squared_error([features['target']], prediction), 
            'details': 'BESS Model prediction based on demand, weather, and other factors.'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
