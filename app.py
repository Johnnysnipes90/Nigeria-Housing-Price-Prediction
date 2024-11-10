
import os
import json
from flask import Flask, request, jsonify
import pandas as pd
import joblib

from src.preprocess import Preprocessor

# Set paths to the model and config files
MODEL_PATH = os.getenv("MODEL_PATH", default="model/best_rf_model.pkl")

# Load the machine learning model
loaded_model = joblib.load(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

# Define a home route that returns a simple JSON response
@app.route("/")
def home():
    return jsonify({"message": "Welcome to Nigeria House Price Prediction!"})

# Define the /predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve the JSON payload from the request
        payload = request.get_json()

        # Convert JSON payload to a DataFrame
        input_data = pd.DataFrame(payload)

        # Initialize preprocessor and preprocess the input data
        preprocessor = Preprocessor()
        processed_data = preprocessor.preprocess(input_data)

        # Use the loaded model to make predictions
        prediction = loaded_model.predict(processed_data)

        # Return the prediction as a JSON response
        return jsonify({"predictions": prediction.tolist()})
    
    except Exception as e:
        # Return error message if there is any exception
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000, debug=True)
