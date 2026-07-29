"""
Fraud Detection Flask App
--------------------------
Loads the trained sklearn pipeline (fraud_pipeline.pkl) and serves:
  - GET  /            -> simple HTML form (templates/index.html)
  - POST /predict     -> JSON API: takes a transaction, returns fraud verdict + probability

Run locally:
    source venv/bin/activate
    python3 app.py
Then open http://127.0.0.1:5000
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained pipeline once at startup (not per-request — much faster)
# ---------------------------------------------------------------------------
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "fraud_pipeline.pkl")
pipeline = joblib.load(PIPELINE_PATH)

# The exact raw columns the pipeline was trained on (pre-transaction only)
EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFlaggedFraud",
]

VALID_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


def validate_and_build_row(payload: dict) -> pd.DataFrame:
    """Validate incoming JSON and build a single-row DataFrame matching the
    pipeline's expected input schema. Raises ValueError on bad input."""
    missing = [c for c in EXPECTED_COLUMNS if c not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    if payload["type"] not in VALID_TYPES:
        raise ValueError(f"'type' must be one of {sorted(VALID_TYPES)}")

    try:
        row = {
            "step": int(payload["step"]),
            "type": str(payload["type"]),
            "amount": float(payload["amount"]),
            "oldbalanceOrg": float(payload["oldbalanceOrg"]),
            "oldbalanceDest": float(payload["oldbalanceDest"]),
            "isFlaggedFraud": int(payload["isFlaggedFraud"]),
        }
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid field type: {e}")

    return pd.DataFrame([row], columns=EXPECTED_COLUMNS)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    payload = request.get_json()

    try:
        input_df = validate_and_build_row(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prediction = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    return jsonify({
        "isFraud": prediction,
        "fraud_probability": round(probability, 4),
        "verdict": "FRAUD" if prediction == 1 else "LEGITIMATE",
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # debug=True is fine locally; Render will use gunicorn instead (see Procfile)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
