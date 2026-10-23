import io
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
# Allow cross-origin requests from the frontend dev server
CORS(app)

# 1. Load Model
MODEL_PATH = "fraud_detection_model.pkl"
model = joblib.load(MODEL_PATH)
EXPECTED_FEATURES = list(model.feature_names_in_)

# 2. Load Demo Dataset for Presets & Fallback
DATASET_PATH = "dataset/demo_transactions.csv"
if os.path.exists(DATASET_PATH):
    demo_df = pd.read_csv(DATASET_PATH)
else:
    demo_df = pd.DataFrame(columns=EXPECTED_FEATURES)


def calculate_risk(prob: float) -> str:
    """Classifies risk level based on probability thresholds."""
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def extract_reasons(row: dict, prob: float) -> list:
    """Provides human-readable explainability flags for anomalies."""
    reasons = []
    if float(row.get("V14", 0)) < -4.0:
        reasons.append("Severe negative shift in V14 transaction pattern")
    if float(row.get("V12", 0)) < -4.0:
        reasons.append("Anomalous V12 deviation detected")
    if float(row.get("V4", 0)) > 3.0:
        reasons.append("Unusually high V4 velocity indicator")
    if float(row.get("Amount", 0)) > 1000.0:
        reasons.append("High transaction amount")
    if prob >= 0.50 and not reasons:
        reasons.append("Composite statistical anomaly across latent features")
    return reasons


@app.route("/api/health", methods=["GET"])
def health():
    """System health and model info endpoint."""
    return jsonify({
        "status": "ok",
        "model_type": type(model).__name__,
        "features_count": len(EXPECTED_FEATURES),
        "features": EXPECTED_FEATURES,
        "sample_rows_available": len(demo_df)
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """
    Returns sample transactions (normal and fraudulent)
    so the frontend can populate 'Quick Test' presets.
    """
    if demo_df.empty:
        return jsonify({"samples": []})

    samples = []
    if "Class" in demo_df.columns:
        fraud_sample = demo_df[demo_df["Class"] == 1].head(3)
        normal_sample = demo_df[demo_df["Class"] == 0].head(3)
        combined = pd.concat([fraud_sample, normal_sample])
    else:
        combined = demo_df.head(6)

    for idx, row in combined.iterrows():
        sample_item = {col: float(row[col]) for col in EXPECTED_FEATURES if col in row}
        sample_item["id"] = int(row.get("id", idx))
        sample_item["label"] = "Fraud Example" if row.get("Class") == 1 else "Normal Example"
        samples.append(sample_item)

    return jsonify({"samples": samples})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Evaluates a single transaction.
    Expects JSON payload with 'Amount' and 'V1' through 'V28' (and optional 'id').
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        row_dict = {}
        for feat in EXPECTED_FEATURES:
            if feat == "id":
                row_dict[feat] = data.get("id", 1)
            else:
                row_dict[feat] = float(data.get(feat, 0.0))

        input_df = pd.DataFrame([row_dict])[EXPECTED_FEATURES]

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        fraud_prob = float(probabilities[classes.index(1)]) if 1 in classes else float(prediction)
        risk_level = calculate_risk(fraud_prob)
        reasons = extract_reasons(row_dict, fraud_prob)

        return jsonify({
            "success": True,
            "id": row_dict["id"],
            "amount": row_dict.get("Amount", 0.0),
            "is_fraud": bool(prediction == 1 or fraud_prob >= 0.50),
            "fraud_probability": round(fraud_prob, 4),
            "risk_level": risk_level,
            "flags": reasons
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """
    Evaluates an uploaded CSV file containing transactions.
    """
    if "file" not in request.files:
        return jsonify({"error": "No CSV file provided under key 'file'"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Uploaded file must be a CSV"}), 400

    try:
        batch_df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))

        for feat in EXPECTED_FEATURES:
            if feat not in batch_df.columns:
                batch_df[feat] = 0.0

        X = batch_df[EXPECTED_FEATURES]
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds

        results = []
        flagged_count = 0

        for i in range(len(batch_df)):
            is_fraud = bool(preds[i] == 1 or probs[i] >= 0.50)
            if is_fraud:
                flagged_count += 1

            results.append({
                "row_index": i,
                "id": int(batch_df.iloc[i].get("id", i + 1)),
                "amount": float(batch_df.iloc[i].get("Amount", 0.0)),
                "is_fraud": is_fraud,
                "fraud_probability": round(float(probs[i]), 4),
                "risk_level": calculate_risk(float(probs[i]))
            })

        return jsonify({
            "success": True,
            "total_records": len(batch_df),
            "flagged_fraud_count": flagged_count,
            "fraud_rate": round(flagged_count / len(batch_df), 4) if len(batch_df) > 0 else 0,
            "results": results
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
