import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
CORS(app)

# Initialize Flask app and enable CORS
app = Flask(__name__)

# Load all models directly into the API (without .pkl files)
def initialize_models():
    """Initialize and train models directly in the API."""
    # Train RandomForest for power demand prediction
    rf_model = RandomForestRegressor(n_estimators=100)
    # Example random data for training
    X_train = pd.DataFrame(np.random.rand(100, 10))  # 100 samples, 10 features
    y_train = np.random.rand(100)
    rf_model.fit(X_train, y_train)

    # Train BESS model (simplified version)
    bess_model = RandomForestRegressor(n_estimators=100)
    bess_model.fit(X_train, y_train)  # Using same data for simplicity

    # Grid optimization model (simplified)
    grid_model = RandomForestRegressor(n_estimators=100)
    grid_model.fit(X_train, y_train)  # Again, using the same data for simplicity

    return rf_model, bess_model, grid_model

# Initialize models
rf_model, bess_model, grid_model = initialize_models()

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    """Health check for API."""
    if not rf_model or not bess_model or not grid_model:
        return jsonify({"error": "One or more models not loaded"}), 500
    return jsonify({"status": "API is running and models are loaded"}), 200

# Power demand prediction route
@app.route('/predict', methods=['POST'])
def predict_power_demand():
    """Predict power demand based on input features."""
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate that features are provided in the correct format
        if not data or "features" not in data:
            raise BadRequest("Missing 'features' in the request body")

        features = data["features"]

        # Check if the features are in the correct list format
        if not isinstance(features, list):
            raise BadRequest("'features' must be a list of numerical values")

        # Standardize features (important for many ML models)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform([features])

        # Predict power demand using the RF model
        prediction = rf_model.predict(features_scaled)

        return jsonify({"prediction": prediction.tolist()}), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# BESS optimization route
@app.route('/optimize_bess', methods=['POST'])
def optimize_bess():
    """Optimize BESS usage based on input features."""
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate that features are provided in the correct format
        if not data or "features" not in data:
            raise BadRequest("Missing 'features' in the request body")

        features = data["features"]

        # Check if the features are in the correct list format
        if not isinstance(features, list):
            raise BadRequest("'features' must be a list of numerical values")

        # Standardize features for BESS model
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform([features])

        # Predict BESS optimization using the BESS model
        prediction = bess_model.predict(features_scaled)

        return jsonify({"prediction": prediction.tolist()}), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Grid optimization route
@app.route('/optimize_grid', methods=['POST'])
def optimize_grid():
    """Optimize grid usage based on input features."""
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate that features are provided in the correct format
        if not data or "features" not in data:
            raise BadRequest("Missing 'features' in the request body")

        features = data["features"]

        # Check if the features are in the correct list format
        if not isinstance(features, list):
            raise BadRequest("'features' must be a list of numerical values")

        # Standardize features for grid optimization
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform([features])

        # Predict grid optimization using the grid model
        prediction = grid_model.predict(features_scaled)

        return jsonify({"prediction": prediction.tolist()}), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Custom 404 handler
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404

# Custom 500 handler
@app.errorhandler(500)
def internal_error(error):
    """Handle 500 internal server errors."""
    return jsonify({"error": "Internal server error"}), 500

# Main entry point
if __name__ == '__main__':
    # Set the port dynamically based on the environment variable, defaulting to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
