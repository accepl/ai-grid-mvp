from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import os
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import asyncio

app = FastAPI()

# Enable CORS to allow API access from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Define request model
class PredictionRequest(BaseModel):
    temperature: float
    day_of_week: int
    holiday: int

# Function to train model if not found
def train_and_save_model():
    print("🚀 Training new model...")
    
    timestamps = pd.date_range(start="2024-01-01", periods=24 * 365, freq='h')
    
    # Base demand with fluctuations
    base_demand = 500 + 100 * np.sin(np.linspace(0, 12 * np.pi, len(timestamps)))
    random_fluctuation = np.random.normal(0, 50, len(timestamps))
    power_demand = base_demand + random_fluctuation

    # Temperature variation
    temperature = 25 + 10 * np.sin(np.linspace(0, 4 * np.pi, len(timestamps))) + np.random.normal(0, 2, len(timestamps))
    
    # Weekday and holiday indicators
    day_of_week = [ts.weekday() for ts in timestamps]
    holiday_indicator = [1 if ts.weekday() in [5, 6] else 0 for ts in timestamps]

    df = pd.DataFrame({
        'Temperature_C': temperature,
        'Day_of_Week': day_of_week,
        'Holiday': holiday_indicator,
        'Power_Demand_MW': power_demand
    })

    # Splitting data
    X = df[['Temperature_C', 'Day_of_Week', 'Holiday']]
    y = df['Power_Demand_MW']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model with lower complexity (faster execution)
    model = RandomForestRegressor(n_estimators=50, random_state=42)  
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "rf_power_demand_model.pkl")
    print("✅ Model trained and saved!")

# Check if model exists, otherwise train it
if not os.path.exists("rf_power_demand_model.pkl"):
    train_and_save_model()

# Load the trained model
rf_model = joblib.load("rf_power_demand_model.pkl")

@app.get("/")
def home():
    return {"message": "✅ AI Grid Load Balancing API is live and ready!"}

@app.post("/predict/")
async def predict_power_demand(request: PredictionRequest):
    try:
        # Convert input to array
        input_data = np.array([[request.temperature, request.day_of_week, request.holiday]])
        
        # Run prediction with a 5-second timeout
        prediction = await asyncio.wait_for(
            asyncio.to_thread(rf_model.predict, input_data),
            timeout=5  
        )

        return {"predicted_power_demand_mw": prediction[0]}

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="⏳ Prediction took too long. Try reducing input complexity.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️ Internal Server Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
