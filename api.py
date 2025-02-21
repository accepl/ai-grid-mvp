from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import train_model

app = Flask(__name__)
DATABASE_URL = "sqlite:///grid_predictions.db"
engine = create_engine(DATABASE_URL)

# Load trained model
def load_model():
    with open("grid_model.pkl", "rb") as file:
        return pickle.load(file)

@app.route("/predict", methods=["GET"])
def predict():
    model = load_model()
    hours = int(request.args.get("hours", 1))
    future_timestamps = np.arange(100, 100 + hours).reshape(-1, 1)
    predictions = model.predict(future_timestamps)

    return jsonify({"predictions": predictions.tolist()})

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        train_model.train_model()
        return jsonify({"message": "Model retrained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def history():
    df = pd.read_sql("SELECT * FROM predictions", con=engine)
    return df.to_json(orient="records")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

