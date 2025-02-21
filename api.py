from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import redis
import logging
from sqlalchemy import create_engine
import train_model

app = Flask(__name__)

# Database Setup
DATABASE_URL = "sqlite:///grid_predictions.db"
engine = create_engine(DATABASE_URL)

# Redis Setup for Caching
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Logging Setup
logging.basicConfig(filename="api_logs.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load trained model
def load_model():
    with open("grid_model.pkl", "rb") as file:
        return pickle.load(file)

@app.route("/predict", methods=["GET"])
def predict():
    model = load_model()
    hours = int(request.args.get("hours", 1))

    # Check Redis Cache First
    cache_key = f"prediction_{hours}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        logging.info(f"Cache hit for {cache_key}")
        return jsonify({"predictions": eval(cached_result.decode("utf-8"))})

    # Compute Prediction
    future_timestamps = np.arange(100, 100 + hours).reshape(-1, 1)
    predictions = model.predict(future_timestamps).tolist()

    # Store in Redis with TTL (Expires in 60 seconds)
    redis_client.setex(cache_key, 60, str(predictions))

    logging.info(f"Cache miss - computed predictions for {cache_key}")
    return jsonify({"predictions": predictions})

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        train_model.train_model()
        redis_client.flushdb()  # Clear cache after retraining
        logging.info("Model retrained successfully")
        return jsonify({"message": "Model retrained successfully"})
    except Exception as e:
        logging.error(f"Error retraining model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def history():
    df = pd.read_sql("SELECT * FROM predictions", con=engine)
    return df.to_json(orient="records")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
