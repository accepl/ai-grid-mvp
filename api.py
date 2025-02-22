import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Define paths
MODEL_DIR = "models"
MODEL_PATH_GRID = os.path.join(MODEL_DIR, "rf_grid_model.pkl")
MODEL_PATH_BESS = os.path.join(MODEL_DIR, "rf_bess_model.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Label Encoder (used for encoding categorical variables)
label_encoder = LabelEncoder()

# Function to load models lazily
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# Function to train & save models if they don’t exist
def train_and_save_models():
    """Trains dummy models for Grid and BESS if no trained model is found."""
    dummy_data = pd.DataFrame({
        "previous_demand": [1500, 1550, 1600, 1580, 1550],
        "weather_conditions": [32, 30, 28, 31, 33],
        "time_of_day": [16, 16, 17, 16, 18],
        "special_event": ["No", "No", "Yes", "No", "No"],
        "economic_index": [5, 6, 4, 5, 3]
    })

    # Encode categorical features
    dummy_data["special_event"] = label_encoder.fit_transform(dummy_data["special_event"])

    # Train models
    rf_grid = RandomForestRegressor().fit(dummy_data.drop("previous_demand", axis=1), dummy_data["previous_demand"])
    rf_bess = RandomForestRegressor().fit(dummy_data.drop("previous_demand", axis=1), dummy_data["previous_demand"])

    # Save models
    joblib.dump(rf_grid, MODEL_PATH_GRID)
    joblib.dump(rf_bess, MODEL_PATH_BESS)
    print("✅ Models trained & saved successfully.")

# Ensure models exist
if not os.path.exists(MODEL_PATH_GRID) or not os.path.exists(MODEL_PATH_BESS):
    print("⚠️ Models not found. Training now...")
    train_and_save_models()

# Prediction function
def predict(model_path, input_data):
    """Loads model if necessary and makes a prediction."""
    model = load_model(model_path)
    if model is None:
        return {"error": "Model not found or failed to load."}, 500

    try:
        input_df = pd.DataFrame([input_data])
        input_df["special_event"] = label_encoder.transform(input_df["special_event"])
        prediction = model.predict(input_df.drop("previous_demand", axis=1))
        return {"prediction": prediction[0]}
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}, 500

# API route for grid demand prediction
@app.route("/predict/grid", methods=["POST"])
def predict_grid():
    data = request.get_json()
    return jsonify(predict(MODEL_PATH_GRID, data))

# API route for BESS prediction
@app.route("/predict/bess", methods=["POST"])
def predict_bess():
    data = request.get_json()
    return jsonify(predict(MODEL_PATH_BESS, data))

# API route to retrain the model
@app.route("/retrain", methods=["POST"])
def retrain_model():
    try:
        train_and_save_models()
        return jsonify({"message": "Model retrained successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Retraining error: {str(e)}"}), 500

# Start Flask app (for local dev only, not needed in production)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
