import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import logging

# Setup logging for better debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Paths for the models (ensure these paths are correct)
MODEL_PATH = os.path.join(os.getcwd(), "rf_power_demand_model.pkl")
BESS_MODEL_PATH = os.path.join(os.getcwd(), "bess_model.pkl")
GRID_MODEL_PATH = os.path.join(os.getcwd(), "grid_optimization_model.pkl")

# Function to load models from disk
def load_model(model_path):
    """Load the model from disk."""
    try:
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            return joblib.load(model_path)
        else:
            logger.error(f"Model file {model_path} not found.")
            return None
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {str(e)}")
        return None

# Load models at the start
rf_model = load_model(MODEL_PATH)
bess_model = load_model(BESS_MODEL_PATH)
grid_model = load_model(GRID_MODEL_PATH)

@app.route('/')
def home():
    return "API is live and ready for testing!"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check route for API."""
    if rf_model is None or bess_model is None or grid_model is None:
        return jsonify({"error": "One or more models are not loaded"}), 500
    return jsonify({"status": "API is running and models are loaded"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict power demand based on input features."""
    if rf_model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate input data
        if not data or "features" not in data:
            raise BadRequest("Missing 'features' in the request body")

        features = data["features"]

        if not isinstance(features, list):
            raise BadRequest("'features' must be a list of numerical values")

        # Predict power demand using the model
        prediction = rf_model.predict([features])

        return jsonify({"prediction": prediction.tolist()}), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/grid_optimization', methods=['POST'])
def grid_optimization():
    """Optimize grid load using input features for grid management."""
    if grid_model is None:
        return jsonify({"error": "Grid optimization model not loaded"}), 500
    
    try:
        # Get the data from the request
        data = request.get_json()

        # Validate input data
        if not data or "features" not in data:
            raise BadRequest("Missing 'features' for grid optimization")

        features = data["features"]

        if not isinstance(features, list):
            raise BadRequest("'features' must be a list of numerical values")

        # Predict grid optimization result using the model
        optimized_output = grid_model.predict([features])

        return jsonify({"optimized_output": optimized_output.tolist()}), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/bess_optimization', methods=['POST'])
def bess_optimization():
    """Optimize Battery Energy Storage System (BESS) based on input features."""
    if bess_model is None:
        return jsonify({"error": "BESS optimization model not loaded"}), 500
    
    try:
        # Get the data from the request
        data = request.get_json()

        # Validate input data
        if not data or "features" not in data:
            raise BadRequest("Missing 'features' for BESS optimization")

        features = data["features"]

        if not isinstance(features, list):
            raise BadRequest("'features' must be a list of numerical values")

        # Predict BESS optimization result using the model
        bess_optimized_output = bess_model.predict([features])

        return jsonify({"bess_optimized_output": bess_optimized_output.tolist()}), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# List all available routes
@app.route('/routes', methods=['GET'])
def list_routes():
    """List all registered API routes."""
    routes = {rule.rule: list(rule.methods) for rule in app.url_map.iter_rules()}
    return jsonify(routes), 200

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

