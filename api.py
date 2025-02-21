from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Example models, using RandomForestRegressor for both features
# Replace with your actual trained models if available
rf_model_grid = RandomForestRegressor()
rf_model_bess = RandomForestRegressor()

# Dummy prediction functions for grid optimization and BESS
def predict_grid(features):
    return rf_model_grid.predict([features]).tolist()

def predict_bess(features):
    return rf_model_bess.predict([features]).tolist()

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running and models are loaded"}), 200

# Grid prediction route
@app.route('/predict/grid', methods=['POST'])
def grid_optimization():
    try:
        data = request.get_json()
        features = data['features']
        prediction = predict_grid(features)
        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# BESS prediction route
@app.route('/predict/bess', methods=['POST'])
def bess_optimization():
    try:
        data = request.get_json()
        features = data['features']
        prediction = predict_bess(features)
        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Running the app
if __name__ == '__main__':
    app.run(debug=True)
