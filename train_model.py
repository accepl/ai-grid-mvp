import pandas as pd
import numpy as np
import requests
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sqlalchemy import create_engine

# Set up database
DATABASE_URL = "sqlite:///grid_predictions.db"
engine = create_engine(DATABASE_URL)

# Function to fetch real-time data (if available)
def fetch_real_time_data():
    try:
        url = "https://api.entsoe.eu/rest"  # Example API (update with actual)
        params = {"key": "your_api_key"}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            return df
    except Exception as e:
        print(f"Error fetching real-time data: {e}")
    return None

# Function to simulate grid load data if API is unavailable
def generate_synthetic_data():
    np.random.seed(42)
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="H")
    demand = np.random.randint(500, 2000, size=100)
    df = pd.DataFrame({"timestamp": timestamps, "demand": demand})
    return df

# Train AI model
def train_model():
    data = fetch_real_time_data()
    if data is None:
        data = generate_synthetic_data()

    X = data.index.values.reshape(-1, 1)
    y = data["demand"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f"Model trained with MSE: {mse}")

    # Save model
    with open("grid_model.pkl", "wb") as file:
        pickle.dump(model, file)

    # Save predictions to database
    data["predicted_demand"] = model.predict(X)
    data.to_sql("predictions", con=engine, if_exists="replace", index=False)

    return model

if __name__ == "__main__":
    train_model()
