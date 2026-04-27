"""
AI Diabetes Risk Predictor — Flask Backend
============================================
Exposes a POST /predict endpoint that accepts patient health metrics
and returns a diabetes risk prediction with probability.

Run locally:
    python app.py

Production (Render):
    gunicorn app:app
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ──────────────────────────────────────────────
# Load the trained model + scaler bundle
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    print("[OK] Model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"[ERROR] model.pkl not found at {MODEL_PATH}. Run train_model.py first.")
    model = None
    scaler = None

# ──────────────────────────────────────────────
# Expected input fields and their valid ranges
# ──────────────────────────────────────────────
FIELDS = [
    {"key": "pregnancies", "min": 0, "max": 20},
    {"key": "glucose",     "min": 0, "max": 300},
    {"key": "bp",          "min": 0, "max": 200},
    {"key": "skin",        "min": 0, "max": 100},
    {"key": "insulin",     "min": 0, "max": 900},
    {"key": "bmi",         "min": 0, "max": 80},
    {"key": "dpf",         "min": 0, "max": 3},
    {"key": "age",         "min": 1, "max": 120},
]


# ──────────────────────────────────────────────
# Health-check endpoint
# ──────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    """Simple health-check so deployment platforms can verify the service."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


# ──────────────────────────────────────────────
# Prediction endpoint
# ──────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept patient metrics and return diabetes prediction.

    Request JSON:
        {
            "pregnancies": 6,
            "glucose": 148,
            "bp": 72,
            "skin": 35,
            "insulin": 0,
            "bmi": 33.6,
            "dpf": 0.627,
            "age": 50
        }

    Response JSON:
        {
            "prediction": 1,
            "probability": 0.73
        }
    """
    # Guard: model must be loaded
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Contact the administrator."}), 503

    # Guard: request must be JSON
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # Validate & extract each field
    values = []
    errors = []
    for field in FIELDS:
        key = field["key"]
        val = data.get(key)

        if val is None:
            errors.append(f"Missing field: '{key}'")
            continue

        # Attempt numeric conversion
        try:
            val = float(val)
        except (ValueError, TypeError):
            errors.append(f"Invalid value for '{key}': must be a number.")
            continue

        # Range check
        if val < field["min"] or val > field["max"]:
            errors.append(
                f"'{key}' out of range ({field['min']}–{field['max']}). Got {val}."
            )
            continue

        values.append(val)

    if errors:
        return jsonify({"error": "Validation failed.", "details": errors}), 422

    # Convert to numpy array and scale
    input_array = np.asarray(values).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    # Predict
    prediction = int(model.predict(scaled_input)[0])
    probability = float(model.predict_proba(scaled_input)[0][1])  # P(diabetic)

    return jsonify({
        "prediction": prediction,
        "probability": round(probability, 4),
    })


# ──────────────────────────────────────────────
# Run the dev server
# ──────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
