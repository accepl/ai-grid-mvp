import os
import pandas as pd
import numpy as np
import pickle
import logging
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set Random Seed Globally
np.random.seed(42)

# Database Setup
DATABASE_URL = "sqlite:///predictions.db"
engine = create_engine(DATABASE_URL)

# Model Storage Directory
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Generate Synthetic Data for Grid Load Forecasting
def generate_grid_data():
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="H")
    demand = np.random.randint(500, 2000, size=100)
    df = pd.DataFrame({"timestamp": timestamps, "demand": demand})
    
    # Feature Engineering: Extract useful time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    
    return df

# Generate Synthetic Data for Battery Charge/Discharge
def generate_battery_data():
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="H")
    demand = np.random.randint(500, 2000, size=100)
    charge = np.random.randint(20, 80, size=100)
    discharge = np.random.randint(10, 60, size=100)
    
    df = pd.DataFrame({"timestamp": timestamps, "demand": demand, "charge": charge, "discharge": discharge})
    return df

# Generate Synthetic Data for Predictive Maintenance
def generate_failure_data():
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="H")
    temp = np.random.randint(20, 100, size=100)  # Temperature sensor
    vibration = np.random.randint(1, 10, size=100)  # Vibration sensor
    failure = np.random.choice([0, 1], size=100, p=[0.9, 0.1])  # Failure labels
    
    df = pd.DataFrame({"timestamp": timestamps, "temperature": temp, "vibration": vibration, "failure": failure})
    return df

# Train Grid Load Model
def train_grid_model():
    logging.info("🔄 Training Grid Load Model...")
    data = generate_grid_data()
    
    X = data[["hour", "day_of_year"]]  # Use time-based features
    y = data["demand"]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, "grid_model.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    data.to_sql("grid_predictions", con=engine, if_exists="replace", index=False)
    logging.info(f"✅ Grid Load Model trained & saved at {model_path}")
    return model

# Train Battery Model
def train_battery_model():
    logging.info("🔄 Training Battery Storage Model...")
    data = generate_battery_data()
    
    X = data[["demand", "charge"]]
    y = data["discharge"]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, "battery_model.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    data.to_sql("battery_predictions", con=engine, if_exists="replace", index=False)
    logging.info(f"✅ Battery Storage Model trained & saved at {model_path}")
    return model

# Train Predictive Maintenance Model
def train_failure_model():
    logging.info("🔄 Training Predictive Maintenance Model...")
    data = generate_failure_data()
    
    X = data[["temperature", "vibration"]]

    model = IsolationForest(contamination=0.1)
    model.fit(X)
    
    # Compute anomaly scores
    data["anomaly_score"] = model.decision_function(X)
    
    model_path = os.path.join(MODEL_DIR, "failure_model.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    data.to_sql("failure_predictions", con=engine, if_exists="replace", index=False)
    logging.info(f"✅ Predictive Maintenance Model trained & saved at {model_path}")
    return model

# Train all models
if __name__ == "__main__":
    train_grid_model()
    train_battery_model()
    train_failure_model()
    logging.info("🚀 All models trained successfully!")
