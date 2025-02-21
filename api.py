from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import joblib

app = Flask(__name__)

# Model paths
MODEL_PATH_GRID = 'models/rf_grid_model.pkl'
MODEL_PATH_BESS = 'models/rf_bess_model.pkl'

# Ensure the models directory exists
def ensure_model_directory():
    if not os.path.exists('models'):
        os.makedirs('models')

# Check if models exist, if not, train them
def check_or_train_models():
    global rf_model_grid, rf_model_bess

    # If models are not already trained or not found, create and train them
    if not os.path.exists(MODEL_PATH_GRID) or not os.path.exists(MODEL_PATH_BESS):
        print("Training models...")

        # Create dummy data for training
        X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y_train_grid = np.array([100, 200, 300])  # Dummy target for grid
        y_train_bess = np.array([50, 150, 250])   # Dummy target for BESS

        # Initialize and train models
        rf_model_grid = RandomForestRegressor()
        rf_model_bess = RandomForestRegressor()

        rf_model_grid.fit(X_train, y_train_grid)
        rf_model_bess.fit(X_train, y_train_bess)

        # Save the models (for future use)
        joblib.dump(rf_model_grid, MODEL_PATH_GRID)
        joblib.dump(rf_model_bess, MODEL_PATH_BESS)

        print("Models trained and saved.")
    else:
        # Load pre-trained models
        rf_model_grid = joblib.load(MODEL_PATH_GRID)
        rf_model_bess = joblib.load(MODEL_PATH_BESS)
        print("Models loaded from disk.")

# Run the model training/loading check
ensure_model_directory()  # Ensure the model directory exists before loading/saving models
check_or_train_models()

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running and models are loaded"}), 200

# Grid prediction route
@app.route('/predict/grid', methods=['POST'])
def grid_optimization():
    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({"error": "Features are required"}), 400
        features = data['features']
        if not isinstance(features, list):
            return jsonify({"error": "Features should be a list of numerical values"}), 400
        features = np.array(features).reshape(1, -1)  # Reshape for prediction
        prediction = rf_model_grid.predict(features).tolist()
        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# BESS prediction route
@app.route('/predict/bess', methods=['POST'])
def bess_optimization():
    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({"error": "Features are required"}), 400
        features = data['features']
        if not isinstance(features, list):
            return jsonify({"error": "Features should be a list of numerical values"}), 400
        features = np.array(features).reshape(1, -1)  # Reshape for prediction
        prediction = rf_model_bess.predict(features).tolist()
        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Running the app
if __name__ == '__main__':
    app.run(debug=True)
