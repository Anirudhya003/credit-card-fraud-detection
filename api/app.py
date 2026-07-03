"""
Flask Web Application for Credit Card Fraud Detection
Provides a web UI and REST API for real-time fraud prediction using trained ML models.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------- Globals loaded at startup ----------
MODEL = None
MODEL_NAME = None
SCALER = None
FEATURE_NAMES = None
CAT_MAPPINGS = None
THRESHOLD = 0.5
FRAUD_ALERTS = []


def load_artifacts():
    """Load the best trained model, scaler, encoders, and threshold."""
    global MODEL, MODEL_NAME, SCALER, FEATURE_NAMES, CAT_MAPPINGS, THRESHOLD

    # Load XGBoost (best model) or fall back to Random Forest
    for name, fname in [("XGBoost", "xgboost.pkl"), ("Random Forest", "random_forest.pkl")]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            MODEL = joblib.load(path)
            MODEL_NAME = name
            break

    if MODEL is None:
        print("ERROR: No trained model found.")
        return False

    # Scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)

    # Feature names (exact order used during training)
    fn_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    if os.path.exists(fn_path):
        FEATURE_NAMES = joblib.load(fn_path)

    # Categorical mappings
    cm_path = os.path.join(MODELS_DIR, "categorical_mappings.pkl")
    if os.path.exists(cm_path):
        CAT_MAPPINGS = joblib.load(cm_path)

    # Optimized threshold
    th_path = os.path.join(MODELS_DIR, "best_threshold.pkl")
    if os.path.exists(th_path):
        THRESHOLD = joblib.load(th_path)

    print(f"  Model    : {MODEL_NAME}")
    print(f"  Features : {len(FEATURE_NAMES) if FEATURE_NAMES else '?'}")
    print(f"  Threshold: {THRESHOLD:.4f}")
    return True


def build_feature_row(data):
    """
    Convert raw input dict to a single-row DataFrame matching training features.
    Applies encoding + feature engineering + scaling.
    """
    row = {
        "User_ID":                   float(data.get("user_id", 1)),
        "Merchant_ID":               float(data.get("merchant_id", 1)),
        "Transaction_Amount":        float(data.get("transaction_amount", 0)),
        "Transaction_Hour":          int(data.get("transaction_hour", 12)),
        "Transaction_Day":           int(data.get("transaction_day", 15)),
        "Transaction_Month":         int(data.get("transaction_month", 6)),
        "Location_Lat":              float(data.get("location_lat", 28.6)),   # Default: Delhi
        "Location_Lon":              float(data.get("location_lon", 77.2)),
        "Device_Type":               data.get("device_type", "Mobile"),
        "Transaction_Type":          data.get("transaction_type", "Online"),
        "Previous_Transaction_Days": int(data.get("previous_transaction_days", 5)),
        "Days_Since_Card_Issued":    int(data.get("days_since_card_issued", 365)),
    }

    # Encode categoricals
    if CAT_MAPPINGS:
        for col in ("Device_Type", "Transaction_Type"):
            mapping = CAT_MAPPINGS.get(col, {})
            val = row[col]
            row[col] = mapping.get(val, 0)

    df = pd.DataFrame([row])

    # Feature engineering (must match preprocessing)
    df["Log_Amount"]   = np.log1p(df["Transaction_Amount"])
    df["Is_Night"]     = (df["Transaction_Hour"].between(0, 5)).astype(int)
    df["Is_New_Card"]  = (df["Days_Since_Card_Issued"] < 30).astype(int)
    df["Is_Dormant"]   = (df["Previous_Transaction_Days"] >= 30).astype(int)
    df["Amount_Zscore"] = 0.0  # will be overridden by scaler
    df["Night_NewCard"] = df["Is_Night"] * df["Is_New_Card"]
    df["Abs_Lat"]       = df["Location_Lat"].abs()

    # Reorder to match training feature order
    if FEATURE_NAMES:
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURE_NAMES]

    # Scale
    if SCALER is not None:
        df = pd.DataFrame(SCALER.transform(df), columns=FEATURE_NAMES)

    return df


def predict_transaction(data):
    """Run the ML model on a single transaction and return result dict."""
    df = build_feature_row(data)

    if hasattr(MODEL, "predict_proba"):
        prob = float(MODEL.predict_proba(df)[0][1])
    else:
        score = float(MODEL.decision_function(df)[0])
        prob = 1 / (1 + np.exp(-score))

    is_fraud = prob >= THRESHOLD
    if is_fraud:
        if prob >= 0.9:
            action = "BLOCKED"
        elif prob >= 0.7:
            action = "BLOCKED - VERIFY IDENTITY"
        else:
            action = "FLAGGED - REVIEW REQUIRED"
    else:
        action = "APPROVED"

    return {
        "is_fraud":          is_fraud,
        "fraud_probability": round(prob, 4),
        "threshold":         round(THRESHOLD, 4),
        "action":            action,
        "model":             MODEL_NAME,
    }


# ==================== HTML PAGES ====================

@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/batch")
def batch_page():
    return render_template("batch.html")

@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


# ==================== API ENDPOINTS ====================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/predict", methods=["POST"])
def predict_api():
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON body"}), 400

    result = predict_transaction(data)
    result["status"] = "success"

    if result["is_fraud"]:
        alert = {
            "alert_id":  f"ALERT_{len(FRAUD_ALERTS)+1:05d}",
            "timestamp": datetime.now().isoformat(),
            "amount":    data.get("transaction_amount", 0),
            "probability": result["fraud_probability"],
            "action":    result["action"],
        }
        FRAUD_ALERTS.append(alert)

    return jsonify(result)


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    transactions = data.get("transactions", []) if data else []
    if not transactions:
        return jsonify({"status": "error", "message": "No transactions"}), 400

    results = []
    for idx, txn in enumerate(transactions):
        res = predict_transaction(txn)
        res["transaction_index"] = idx
        results.append(res)
        if res["is_fraud"]:
            FRAUD_ALERTS.append({
                "alert_id":  f"ALERT_{len(FRAUD_ALERTS)+1:05d}",
                "timestamp": datetime.now().isoformat(),
                "amount":    txn.get("transaction_amount", 0),
                "probability": res["fraud_probability"],
                "action":    res["action"],
            })

    fraud_count = sum(1 for r in results if r["is_fraud"])
    return jsonify({
        "status": "success",
        "total": len(results),
        "fraud_detected": fraud_count,
        "results": results,
    })


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    return jsonify({
        "status": "success",
        "total": len(FRAUD_ALERTS),
        "alerts": FRAUD_ALERTS[-100:],
    })


@app.route("/api/alerts/clear", methods=["POST"])
def clear_alerts():
    global FRAUD_ALERTS
    FRAUD_ALERTS = []
    return jsonify({"status": "success", "message": "Alerts cleared"})


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "features": FEATURE_NAMES or [],
        "total_features": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
    })


# ==================== MAIN ====================

# Load the model when the application starts (works for Gunicorn and Flask)
if not load_artifacts():
    raise RuntimeError("Failed to load model. Run 'python run_pipeline.py' first.")

if __name__ == "__main__":
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION - WEB APPLICATION")
    print("=" * 60)
    print("\nStarting server at http://localhost:5000")
    print("=" * 60)

    app.run(debug=False, host="0.0.0.0", port=5000)
