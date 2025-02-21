from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import traceback
import os
from werkzeug.exceptions import HTTPException

# ----------------------- Initial Setup -----------------------

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Enable Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Simulated Database for Model Versioning
MODEL_VERSIONS = []

# ----------------------- Error Handling -----------------------

def log_error(error_message):
    """ Logs errors with full stack trace for debugging """
    logging.error(f"Error: {error_message}\n{traceback.format_exc()}")

@app.errorhandler(404)
def not_found(error):
    log_error("404 - Endpoint Not Found")
    return jsonify({"error": "Endpoint not found", "code": 404}), 404

@app.errorhandler(500)
def internal_error(error):
    log_error("500 - Internal Server Error")
    return jsonify({"error": "Something went wrong!", "code": 500}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Invalid request", "code": 400}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({"error": "Unauthorized access", "code": 401}), 401

@app.errorhandler(403)
def forbidden(error):
    return jsonify({"error": "Forbidden access", "code": 403}), 403

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed", "code": 405}), 405

@app.errorhandler(408)
def request_timeout(error):
    return jsonify({"error": "Request timeout", "code": 408}), 408

@app.errorhandler(429)
def too_many_requests(error):
    return jsonify({"error": "Too many requests", "code": 429}), 429

@app.errorhandler(502)
def bad_gateway(error):
    return jsonify({"error": "Bad gateway", "code": 502}), 502

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({"error": "Service unavailable", "code": 503}), 503

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """ Catch any unexpected errors and prevent the API from crashing """
    log_error("Unexpected Server Error")
    return jsonify({"error": "An unexpected error occurred.", "code": 500}), 500

# ----------------------- API Routes -----------------------

@app.route('/health', methods=['GET'])
def health_check():
    """ Simple health check endpoint """
    return jsonify({"status": "API is running", "code": 200}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """ AI Prediction Endpoint """
    try:
        data = request.get_json()
        if not data or "hours" not in data:
            return bad_request("Missing 'hours' parameter")

        hours = data["hours"]
        if not isinstance(hours, int) or hours <= 0:
            return bad_request("Invalid 'hours' value. Must be a positive integer.")

        prediction = hours * 1.2  # Dummy logic, replace with actual model inference
        return jsonify({"prediction": prediction, "code": 200}), 200
    except Exception as e:
        log_error("Prediction Error")
        return internal_error(e)

@app.route('/battery_predict', methods=['POST'])
def battery_predict():
    """ Predict Battery Charge/Discharge Time """
    try:
        data = request.get_json()
        if not data or "battery_level" not in data:
            return bad_request("Missing 'battery_level' parameter")

        battery_level = data["battery_level"]
        if not isinstance(battery_level, (int, float)) or battery_level < 0 or battery_level > 100:
            return bad_request("Invalid 'battery_level'. Must be between 0-100.")

        charge_time = (100 - battery_level) * 0.5  # Dummy logic
        return jsonify({"charge_time_needed": charge_time, "code": 200}), 200
    except Exception as e:
        log_error("Battery Prediction Error")
        return internal_error(e)

@app.route('/model_versions', methods=['GET'])
def get_model_versions():
    """ Retrieve AI Model Versions """
    return jsonify({"models": MODEL_VERSIONS, "code": 200}), 200

@app.route('/retrain', methods=['POST'])
def retrain():
    """ Retrain AI Model """
    try:
        new_model_version = f"Model-{time.time()}"
        MODEL_VERSIONS.append(new_model_version)
        return jsonify({"message": "Model retrained", "version": new_model_version, "code": 200}), 200
    except Exception as e:
        log_error("Retraining Error")
        return internal_error(e)

@app.route('/config', methods=['GET'])
def get_config():
    """ Get Server Configuration (Example Usage) """
    config = {
        "server": os.getenv("SERVER_NAME", "Accepl AI"),
        "debug_mode": app.debug,
        "max_requests": 1000,
        "allowed_ips": ["0.0.0.0/0"],
    }
    return jsonify(config), 200

@app.route('/status', methods=['GET'])
def get_status():
    """ Get API Uptime & Statistics """
    uptime = time.time() - os.getenv("START_TIME", time.time())
    return jsonify({
        "status": "running",
        "uptime_seconds": uptime,
        "total_models_trained": len(MODEL_VERSIONS),
        "active_endpoints": ["/predict", "/battery_predict", "/retrain", "/model_versions"]
    }), 200

# ----------------------- Future-Proof Enhancements -----------------------

@app.route('/logs', methods=['GET'])
def get_logs():
    """ Returns the last 50 log entries (only works if logging is enabled) """
    try:
        with open("server.log", "r") as file:
            logs = file.readlines()[-50:]  # Get last 50 log lines
        return jsonify({"logs": logs}), 200
    except Exception as e:
        log_error("Error Fetching Logs")
        return internal_error(e)

# ----------------------- Main Entry -----------------------

if __name__ == '__main__':
    os.environ["START_TIME"] = str(time.time())  # Set start time for uptime tracking
    app.run(host='0.0.0.0', port=5000, debug=True)
