import os
import json
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
from waitress import serve

from src.preprocess import Preprocessor

# Set paths to the model file
MODEL_PATH = os.getenv("MODEL_PATH", default="model/best_rf_model.pkl")
# Load the model
loaded_model = joblib.load(MODEL_PATH)

app = Flask(__name__)

# Define a home route that returns a simple json response
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the House prediction API!"})

@app.route("/predict", methods=["POST"])
def predict():
    # Get the JSON payload from the request
    payload = request.get_json()

    # Convert the JSON payload to a DataFrame
    data = pd.DataFrame(payload)

    # Initialize the preprocessor and preprocess the data
    preprocessor = Preprocessor()

    try:
        # Preprocess the incoming data using the Preprocessor class
        preprocessed_data = preprocessor.preprocess(data)

        # Use the loaded model to make a prediction
        prediction = loaded_model.predict(preprocessed_data)

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        # Handle errors during preprocessing or prediction
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == "__main__":
    serve(app, host = '127.0.0.1', port = 5000)