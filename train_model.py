import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

# Database Setup
DATABASE_URL = "sqlite:///predictions.db"
engine = create_engine(DATABASE_URL)

# Generate Synthetic Data for Grid Load Forecasting
def generate_grid_data():
    np.random.seed(42)
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="H")
    demand = np.random.randint(500, 2000, size=100)
    df = pd.DataFrame({"timestamp": timestamps, "demand": demand})
    return df

# Generate Synthetic Data for Battery Charge/Discharge
def generate_battery_data():
    np.random.seed(42)
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="H")
    demand = np.random.randint(500, 2000, size=100)
    charge = np.random.randint(20, 80, size=100)
    discharge = np.random.randint(10, 60, size=100)
    df = pd.DataFrame({"timestamp": timestamps, "demand": demand, "charge": charge, "discharge": discharge})
    return df

# Generate Synthetic Data for Predictive Maintenance
def generate_failure_data():
    np.random.seed(42)
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="H")
    temp = np.random.randint(20, 100, size=100)  # Temperature sensor
    vibration = np.random.randint(1, 10, size=100)  # Vibration sensor
    failure = np.random.choice([0, 1], size=100, p=[0.9, 0.1])  # Failure labels
    df = pd.DataFrame({"timestamp": timestamps, "temperature": temp, "vibration": vibration, "failure": failure})
    return df

# Train Grid Load Model
def train_grid_model():
    data = generate_grid_data()
    X = data.index.values.reshape(-1, 1)
    y = data["demand"]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    with open("grid_model.pkl", "wb") as file:
        pickle.dump(model, file)

    data.to_sql("grid_predictions", con=engine, if_exists="replace", index=False)
    return model

# Train Battery Model
def train_battery_model():
    data = generate_battery_data()
    X = data[["demand", "charge"]]
    y = data["discharge"]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    with open("battery_model.pkl", "wb") as file:
        pickle.dump(model, file)

    data.to_sql("battery_predictions", con=engine, if_exists="replace", index=False)
    return model

# Train Predictive Maintenance Model
def train_failure_model():
    data = generate_failure_data()
    X = data[["temperature", "vibration"]]
    y = data["failure"]

    model = IsolationForest(contamination=0.1)
    model.fit(X)

    with open("failure_model.pkl", "wb") as file:
        pickle.dump(model, file)

    data.to_sql("failure_predictions", con=engine, if_exists="replace", index=False)
    return model

# Train all models
if __name__ == "__main__":
    train_grid_model()
    train_battery_model()
    train_failure_model()
    print("All models trained successfully!")
