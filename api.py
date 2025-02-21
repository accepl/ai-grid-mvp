from fastapi import FastAPI
import uvicorn
import numpy as np
import joblib
from pydantic import BaseModel

# Load trained model
rf_model = joblib.load("rf_power_demand_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request model
class PredictionRequest(BaseModel):
    temperature: float
    day_of_week: int
    holiday: int

# API endpoint for power demand prediction
@app.post("/predict/")
def predict_power_demand(request: PredictionRequest):
    input_data = np.array([[request.temperature, request.day_of_week, request.holiday]])
    prediction = rf_model.predict(input_data)[0]
    return {"predicted_power_demand_mw": prediction}

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
