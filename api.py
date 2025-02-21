import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError

app = Flask(__name__)
CORS(app)

# Model path and loading logic
MODEL_PATH_RF = "rf_power_demand_model.pkl"
MODEL_PATH_BESS = "bess_model.pkl"
MODEL_PATH_GRID = "grid_optimization_model.pkl"

def load_model(model_path):
    """Load model from the given path."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

# Load models at startup
rf_model = load_model(MODEL_PATH_RF)
bess_model = load_model(MODEL_PATH_BESS)
grid_model = load_model(MODEL_PATH_GRID)

@app.route('/')
def home():
    return "API is live!"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for the API."""
    if not rf_model or not bess_model or not grid_model:
        return jsonify({"error": "One or more models not loaded"}), 500
    return jsonify({"status": "API is running and all models are loaded"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict power demand, BESS performance, and grid optimization."""
    if not rf_model or not bess_model or not grid_model:
        return jsonify({"error": "One or more models not loaded"}), 500
    
    data = request.get_json(force=True)
    if "features" not in data:
        return jsonify({"error": "Missing 'features' in the request body"}), 400

    features = data["features"]

    try:
        # Predict power demand using RF model
        rf_prediction = rf_model.predict([features])

        # Predict BESS performance
        bess_prediction = bess_model.predict([features])

        # Predict grid optimization
        grid_prediction = grid_model.predict([features])

        return jsonify({
            "rf_prediction": rf_prediction.tolist(),
            "bess_prediction": bess_prediction.tolist(),
            "grid_prediction": grid_prediction.tolist()
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
