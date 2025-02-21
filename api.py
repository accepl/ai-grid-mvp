from fastapi import FastAPI
import uvicorn
import numpy as np
import os
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = FastAPI()

# Define request model for power demand prediction
class PredictionRequest(BaseModel):
    temperature: float
    day_of_week: int
    holiday: int

# Define request model for BESS optimization
class BESSRequest(BaseModel):
    current_battery_level: float  # Percentage (0-100%)
    predicted_power_demand: float  # MW
    grid_surplus: float  # MW (negative if grid deficit)
    max_battery_capacity: float  # MWh

# Function to train power demand model if not found
def train_and_save_model():
    print("Training new power demand model...")
    timestamps = pd.date_range(start="2024-01-01", periods=24*365, freq='h')
    base_demand = 500 + 100 * np.sin(np.linspace(0, 12 * np.pi, len(timestamps)))
    random_fluctuation = np.random.normal(0, 50, len(timestamps))
    power_demand = base_demand + random_fluctuation

    temperature = 25 + 10 * np.sin(np.linspace(0, 4 * np.pi, len(timestamps))) + np.random.normal(0, 2, len(timestamps))
    day_of_week = [ts.weekday() for ts in timestamps]
    holiday_indicator = [1 if ts.weekday() in [5, 6] else 0 for ts in timestamps]

    df = pd.DataFrame({
        'Temperature_C': temperature,
        'Day_of_Week': day_of_week,
        'Holiday': holiday_indicator,
        'Power_Demand_MW': power_demand
    })

    X = df[['Temperature_C', 'Day_of_Week', 'Holiday']]
    y = df['Power_Demand_MW']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "rf_power_demand_model.pkl")
    print("Power demand model trained and saved!")

# Train power demand model if it doesn't exist
if not os.path.exists("rf_power_demand_model.pkl"):
    train_and_save_model()

rf_model = joblib.load("rf_power_demand_model.pkl")

@app.get("/")
def home():
    return {"message": "AI Grid Load Balancing API is live!"}

@app.post("/predict/")
def predict_power_demand(request: PredictionRequest):
    input_data = np.array([[request.temperature, request.day_of_week, request.holiday]])
    prediction = rf_model.predict(input_data)[0]
    return {"predicted_power_demand_mw": prediction}

@app.post("/bess-optimize/")
def optimize_bess(request: BESSRequest):
    # Decision logic for battery storage optimization
    battery_action = "hold"
    charge_power = 0  # MW

    if request.grid_surplus > 50 and request.current_battery_level < 90:
        battery_action = "charge"
        charge_power = min(request.grid_surplus, request.max_battery_capacity * 0.1)  # Charge up to 10% of capacity

    elif request.grid_surplus < -50 and request.current_battery_level > 20:
        battery_action = "discharge"
        charge_power = min(abs(request.grid_surplus), request.max_battery_capacity * 0.1)  # Discharge up to 10% of capacity

    return {
        "battery_action": battery_action,
        "charge_power_mw": charge_power
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

