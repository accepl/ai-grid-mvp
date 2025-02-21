import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

# Database Setup
DATABASE_URL = "sqlite:///grid_predictions.db"
engine = create_engine(DATABASE_URL)

# Simulate battery charge/discharge data
def generate_battery_data():
    np.random.seed(42)
    timestamps = pd.date_range(start="2025-01-01", periods=100, freq="H")
    demand = np.random.randint(500, 2000, size=100)
    charge = np.random.randint(20, 80, size=100)  # Battery charge level %
    discharge = np.random.randint(10, 60, size=100)  # Battery discharge level %
    df = pd.DataFrame({"timestamp": timestamps, "demand": demand, "charge": charge, "discharge": discharge})
    return df

# Train AI model for battery optimization
def train_battery_model():
    data = generate_battery_data()
    
    X = data[["demand", "charge"]]
    y = data["discharge"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Save Model
    with open("battery_model.pkl", "wb") as file:
        pickle.dump(model, file)

    # Save battery predictions to database
    data["predicted_discharge"] = model.predict(X)
    data.to_sql("battery_predictions", con=engine, if_exists="replace", index=False)

    print("Battery model trained successfully.")

if __name__ == "__main__":
    train_battery_model()

