import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Define model paths
MODEL_PATH_GRID = 'models/rf_grid_model.pkl'
MODEL_PATH_BESS = 'models/rf_bess_model.pkl'

# Load the models
def load_models():
    global rf_model_grid, rf_model_bess
    if os.path.exists(MODEL_PATH_GRID):
        rf_model_grid = joblib.load(MODEL_PATH_GRID)
        print("Grid model loaded successfully.")
    else:
        print(f"Grid model not found at {MODEL_PATH_GRID}. Please train the model first.")
    
    if os.path.exists(MODEL_PATH_BESS):
        rf_model_bess = joblib.load(MODEL_PATH_BESS)
        print("BESS model loaded successfully.")
    else:
        print(f"BESS model not found at {MODEL_PATH_BESS}. Please train the model first.")

# Load models when the app starts
load_models()

# Define prediction route for Grid
@app.route('/predict/grid', methods=['POST'])
def predict_grid():
    try:
        data = request.get_json()

        # Ensure the data has the correct format
        expected_keys = ['previous_demand', 'weather_conditions', 'time_of_day', 'day_of_week', 'special_event', 'economic_index', 'season']
        if not all(key in data for key in expected_keys):
            return jsonify({"error": "Missing required input parameters"}), 400

        # Convert input data into a dataframe
        input_data = pd.DataFrame([data])
        
        # Prediction using the loaded grid model
        prediction = rf_model_grid.predict(input_data)
        
        return jsonify({"predicted_demand": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": f"Error during grid prediction: {str(e)}"}), 500

# Define prediction route for BESS
@app.route('/predict/bess', methods=['POST'])
def predict_bess():
    try:
        data = request.get_json()

        # Ensure the data has the correct format
        expected_keys = ['previous_demand', 'weather_conditions', 'time_of_day', 'day_of_week', 'special_event', 'economic_index', 'season']
        if not all(key in data for key in expected_keys):
            return jsonify({"error": "Missing required input parameters"}), 400

        # Convert input data into a dataframe
        input_data = pd.DataFrame([data])
        
        # Prediction using the loaded BESS model
        prediction = rf_model_bess.predict(input_data)
        
        return jsonify({"predicted_storage": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": f"Error during BESS prediction: {str(e)}"}), 500

# Train models if they are not found
def train_models():
    # Example training code (using dummy data)
    grid_data = pd.DataFrame({
        'previous_demand': [1500, 1550, 1600, 1580, 1550, 1530],
        'weather_conditions': [32, 30, 28, 31, 33],
        'time_of_day': [16, 16, 17, 16, 18],
        'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'special_event': ['No', 'No', 'Yes', 'No', 'No'],
        'economic_index': [5, 6, 4, 5, 3],
        'season': ['Winter', 'Winter', 'Winter', 'Winter', 'Winter']
    })
    
    bess_data = pd.DataFrame({
        'previous_demand': [1500, 1550, 1600, 1580, 1550, 1530],
        'weather_conditions': [32, 30, 28, 31, 33],
        'time_of_day': [16, 16, 17, 16, 18],
        'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'special_event': ['No', 'No', 'Yes', 'No', 'No'],
        'economic_index': [5, 6, 4, 5, 3],
        'season': ['Winter', 'Winter', 'Winter', 'Winter', 'Winter']
    })
    
    # Dummy model training (you should replace it with your actual model training code)
    from sklearn.ensemble import RandomForestRegressor
    rf_model_grid = RandomForestRegressor()
    rf_model_bess = RandomForestRegressor()
    
    rf_model_grid.fit(grid_data, [1500, 1550, 1600, 1580, 1550])
    rf_model_bess.fit(bess_data, [500, 520, 530, 510, 500])
    
    # Save models after training
    joblib.dump(rf_model_grid, MODEL_PATH_GRID)
    joblib.dump(rf_model_bess, MODEL_PATH_BESS)

# Uncomment to train models on first startup (if models don't exist)
# train_models()

if __name__ == '__main__':
    app.run(debug=True)
