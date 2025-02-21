import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for all routes

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Define model paths
MODEL_PATH_GRID = 'models/rf_grid_model.pkl'
MODEL_PATH_BESS = 'models/rf_bess_model.pkl'

# Initialize model variables
rf_model_grid = None
rf_model_bess = None

# Label Encoder (used for encoding categorical variables)
label_encoder = LabelEncoder()

# Load the models if they exist
def load_models():
    global rf_model_grid, rf_model_bess
    if os.path.exists(MODEL_PATH_GRID):
        rf_model_grid = joblib.load(MODEL_PATH_GRID)
        print("Grid model loaded successfully.")
    else:
        print(f"Grid model not found at {MODEL_PATH_GRID}. Training the model now...")
        train_and_save_models()

    if os.path.exists(MODEL_PATH_BESS):
        rf_model_bess = joblib.load(MODEL_PATH_BESS)
        print("BESS model loaded successfully.")
    else:
        print(f"BESS model not found at {MODEL_PATH_BESS}. Training the model now...")
        train_and_save_models()

# Train the models (dummy data for now)
def train_and_save_models():
    """
    Trains models using dummy data for grid and BESS.
    Replace this with your real training code and data.
    """
    # Example dummy data for grid and BESS
    grid_data = pd.DataFrame({
        'previous_demand': [1500, 1550, 1600, 1580, 1550],
        'weather_conditions': [32, 30, 28, 31, 33],
        'time_of_day': [16, 16, 17, 16, 18],
        'special_event': ['No', 'No', 'Yes', 'No', 'No'],
        'economic_index': [5, 6, 4, 5, 3]
    })

    bess_data = pd.DataFrame({
        'previous_demand': [1500, 1550, 1600, 1580, 1550],
        'weather_conditions': [32, 30, 28, 31, 33],
        'time_of_day': [16, 16, 17, 16, 18],
        'special_event': ['No', 'No', 'Yes', 'No', 'No'],
        'economic_index': [5, 6, 4, 5, 3]
    })
    
    # Preprocessing: Convert categorical data into numeric data using label encoding
    grid_data['special_event'] = label_encoder.fit_transform(grid_data['special_event'])
    bess_data['special_event'] = label_encoder.fit_transform(bess_data['special_event'])

    # Define and train models
    global rf_model_grid, rf_model_bess
    rf_model_grid = RandomForestRegressor()
    rf_model_bess = RandomForestRegressor()

    # Dummy training process (replace with your actual logic)
    rf_model_grid.fit(grid_data.drop('previous_demand', axis=1), grid_data['previous_demand'])
    rf_model_bess.fit(bess_data.drop('previous_demand', axis=1), bess_data['previous_demand'])

    # Save the trained models (ensure directory exists)
    joblib.dump(rf_model_grid, MODEL_PATH_GRID)
    joblib.dump(rf_model_bess, MODEL_PATH_BESS)

    print("Models trained and saved successfully.")

# Load models when the app starts
load_models()

# Prediction route for grid demand
@app.route('/predict/grid', methods=['POST'])
def predict_grid():
    """
    Predicts the grid demand based on input data.
    The input data must include the necessary features.
    """
    try:
        data = request.get_json()

        # Ensure the data contains the required fields
        expected_keys = ['previous_demand', 'weather_conditions', 'time_of_day', 'special_event', 'economic_index']
        if not all(key in data for key in expected_keys):
            return jsonify({"error": "Missing required input parameters"}), 400
        
        # Convert the input data into a DataFrame for prediction
        input_data = pd.DataFrame([data])

        # Apply label encoding on the input data
        input_data['special_event'] = label_encoder.transform(input_data['special_event'])

        # Prediction using the loaded grid model
        prediction = rf_model_grid.predict(input_data.drop('previous_demand', axis=1))

        return jsonify({"predicted_demand": prediction[0]})

    except Exception as e:
        return jsonify({"error": f"Error during grid prediction: {str(e)}"}), 500

# Prediction route for BESS storage
@app.route('/predict/bess', methods=['POST'])
def predict_bess():
    """
    Predicts the battery storage requirement for the given input data.
    The input data must include the necessary features.
    """
    try:
        data = request.get_json()

        # Ensure the data contains the required fields
        expected_keys = ['previous_demand', 'weather_conditions', 'time_of_day', 'special_event', 'economic_index']
        if not all(key in data for key in expected_keys):
            return jsonify({"error": "Missing required input parameters"}), 400
        
        # Convert the input data into a DataFrame for prediction
        input_data = pd.DataFrame([data])

        # Apply label encoding on the input data
        input_data['special_event'] = label_encoder.transform(input_data['special_event'])

        # Prediction using the loaded BESS model
        prediction = rf_model_bess.predict(input_data.drop('previous_demand', axis=1))

        return jsonify({"predicted_storage": prediction[0]})

    except Exception as e:
        return jsonify({"error": f"Error during BESS prediction: {str(e)}"}), 500

# Start the Flask app (only for local development, remove if deploying)
if __name__ == '__main__':
    # Ensuring the app uses the correct port when deployed on Render.com
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
