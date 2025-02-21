import os
import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
CORS(app)

# Model path and loading logic
MODEL_PATH = "rf_power_demand_model.pkl"

def load_model():
    """Load the model from disk."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

# Load model at startup
rf_model = load_model()

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    """Health check for API."""
    if rf_model is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({"status": "API is running and model is loaded"}), 200

# Home route
@app.route('/')
def home():
    return "API is live!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """Predict power demand based on input features."""
    if rf_model is None:
        return jsonify({"error": "Model not loaded"}), 500

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

        # Predict power demand using the model
        prediction = rf_model.predict([features])

        return jsonify({"prediction": prediction.tolist()}), 200

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
