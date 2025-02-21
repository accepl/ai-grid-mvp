import os
import logging
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend/API calls

# Database Connection
DATABASE_URL = "sqlite:///predictions.db"
engine = create_engine(DATABASE_URL)

# Set up logging
logging.basicConfig(filename="api.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to load AI models safely
def load_model(filename):
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        logging.error(f"Error loading model {filename}: {e}")
        return None

# Load all models at startup
grid_model = load_model("grid_model.pkl")
battery_model = load_model("battery_model.pkl")
failure_model = load_model("failure_model.pkl")

# 🔥 Health Check Endpoint (For Deployment Debugging)
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running"}), 200

# 🔥 Grid Load Prediction
@app.route("/predict", methods=["GET"])
def predict():
    if grid_model is None:
        return jsonify({"error": "Grid model is not loaded"}), 500

    try:
        hours = int(request.args.get("hours", 1))
        future_timestamps = np.arange(100, 100 + hours).reshape(-1, 1)
        predictions = grid_model.predict(future_timestamps).tolist()
        logging.info(f"Grid prediction requested for {hours} hours")
        return jsonify({"predictions": predictions})
    except Exception as e:
        logging.error(f"Grid Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

# 🔥 Battery Charge/Discharge Prediction
@app.route("/battery_predict", methods=["GET"])
def battery_predict():
    if battery_model is None:
        return jsonify({"error": "Battery model is not loaded"}), 500

    try:
        demand = float(request.args.get("demand", 1000))
        charge = float(request.args.get("charge", 50))
        prediction = battery_model.predict(np.array([[demand, charge]]))[0]
        logging.info(f"Battery prediction requested: Demand={demand}, Charge={charge}")
        return jsonify({"predicted_discharge": round(prediction, 2)})
    except Exception as e:
        logging.error(f"Battery Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

# 🔥 Predictive Maintenance (Failure Detection)
@app.route("/failure_predict", methods=["GET"])
def failure_predict():
    if failure_model is None:
        return jsonify({"error": "Failure detection model is not loaded"}), 500

    try:
        temperature = float(request.args.get("temperature", 50))
        vibration = float(request.args.get("vibration", 5))
        prediction = failure_model.predict(np.array([[temperature, vibration]]))[0]
        failure_risk = "High" if prediction == -1 else "Low"
        logging.info(f"Failure prediction requested: Temp={temperature}, Vibration={vibration}, Risk={failure_risk}")
        return jsonify({"failure_risk": failure_risk})
    except Exception as e:
        logging.error(f"Failure Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

# 🔥 Retrain AI Models
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        import train_model  # Import dynamically for retraining
        train_model.train_grid_model()
        train_model.train_battery_model()
        train_model.train_failure_model()
        logging.info("Models retrained successfully")
        return jsonify({"message": "Models retrained successfully"}), 200
    except Exception as e:
        logging.error(f"Retraining Error: {e}")
        return jsonify({"error": str(e)}), 500

# 🔥 Fetch Historical Predictions
@app.route("/history", methods=["GET"])
def history():
    try:
        df = pd.read_sql("SELECT * FROM grid_predictions", con=engine)
        return df.to_json(orient="records")
    except Exception as e:
        logging.error(f"History Fetch Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/battery_history", methods=["GET"])
def battery_history():
    try:
        df = pd.read_sql("SELECT * FROM battery_predictions", con=engine)
        return df.to_json(orient="records")
    except Exception as e:
        logging.error(f"Battery History Fetch Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/failure_history", methods=["GET"])
def failure_history():
    try:
        df = pd.read_sql("SELECT * FROM failure_predictions", con=engine)
        return df.to_json(orient="records")
    except Exception as e:
        logging.error(f"Failure History Fetch Error: {e}")
        return jsonify({"error": str(e)}), 500

# 🔥 Start Flask with Render-Compatible Port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
