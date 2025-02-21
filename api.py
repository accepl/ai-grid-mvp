from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import train_model

app = Flask(__name__)
DATABASE_URL = "sqlite:///predictions.db"
engine = create_engine(DATABASE_URL)

# Load models
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

@app.route("/predict", methods=["GET"])
def predict():
    model = load_model("grid_model.pkl")
    hours = int(request.args.get("hours", 1))
    future_timestamps = np.arange(100, 100 + hours).reshape(-1, 1)
    predictions = model.predict(future_timestamps)
    return jsonify({"predictions": predictions.tolist()})

@app.route("/battery_predict", methods=["GET"])
def battery_predict():
    model = load_model("battery_model.pkl")
    demand = float(request.args.get("demand", 1000))
    charge = float(request.args.get("charge", 50))
    prediction = model.predict(np.array([[demand, charge]]))[0]
    return jsonify({"predicted_discharge": round(prediction, 2)})

@app.route("/failure_predict", methods=["GET"])
def failure_predict():
    model = load_model("failure_model.pkl")
    temperature = float(request.args.get("temperature", 50))
    vibration = float(request.args.get("vibration", 5))
    prediction = model.predict(np.array([[temperature, vibration]]))[0]
    return jsonify({"failure_risk": "High" if prediction == -1 else "Low"})

@app.route("/retrain", methods=["POST"])
def retrain():
    train_model.train_grid_model()
    train_model.train_battery_model()
    train_model.train_failure_model()
    return jsonify({"message": "Models retrained successfully"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
