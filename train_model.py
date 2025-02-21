import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate timestamps for 1 year (hourly data)
timestamps = pd.date_range(start="2024-01-01", periods=24*365, freq='H')

# Simulated power demand data (MW)
base_demand = 500 + 100 * np.sin(np.linspace(0, 12 * np.pi, len(timestamps)))
random_fluctuation = np.random.normal(0, 50, len(timestamps))
power_demand = base_demand + random_fluctuation

# Simulated temperature data (°C)
temperature = 25 + 10 * np.sin(np.linspace(0, 4 * np.pi, len(timestamps))) + np.random.normal(0, 2, len(timestamps))

# Day of the week (0 = Monday, 6 = Sunday)
day_of_week = [ts.weekday() for ts in timestamps]

# Holiday indicator (1 for weekend, 0 otherwise)
holiday_indicator = [1 if ts.weekday() in [5, 6] else 0 for ts in timestamps]

# Create DataFrame
df = pd.DataFrame({
    'Temperature_C': temperature,
    'Day_of_Week': day_of_week,
    'Holiday': holiday_indicator,
    'Power_Demand_MW': power_demand
})

# Prepare data for model training
X = df[['Temperature_C', 'Day_of_Week', 'Holiday']]
y = df['Power_Demand_MW']

# Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "rf_power_demand_model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model trained successfully! RMSE: {rmse:.2f} MW")
