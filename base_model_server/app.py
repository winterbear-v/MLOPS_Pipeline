import os
import json
import time
import mlflow.sklearn
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Gauge, Histogram

app = Flask(__name__)
metrics = PrometheusMetrics(app)

MODEL_NAME = os.environ.get("MODEL_NAME", "unknown_model")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH", "")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
INPUT_SCHEMA = json.loads(os.environ.get("INPUT_SCHEMA", "{}"))
BASELINE_STATS = json.loads(os.environ.get("BASELINE_STATS", "{}"))

# Custom Prometheus metrics for MLOps monitoring & data drift (Unit-VI)
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total number of predictions served by this model container",
    ["model_name", "version"],
)
PREDICTION_VALUE_GAUGE = Gauge(
    "model_last_prediction_value",
    "Most recent numeric prediction output value",
    ["model_name", "version"],
)
FEATURE_DRIFT_GAUGE = Gauge(
    "model_feature_drift_zscore",
    "Instantaneous Z-score data drift across numeric input features",
    ["model_name", "version"],
)
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Time spent running sklearn pipeline inference",
    ["model_name", "version"],
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
loaded_model = None


def load_registered_model():
    global loaded_model
    if loaded_model is not None:
        return loaded_model
    if ARTIFACT_PATH and os.path.exists(ARTIFACT_PATH):
        loaded_model = mlflow.sklearn.load_model(ARTIFACT_PATH)
    else:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        loaded_model = mlflow.sklearn.load_model(model_uri)
    return loaded_model


@app.route("/health", methods=["GET"])
def health():
    return (
        jsonify(
            {
                "status": "healthy",
                "model_name": MODEL_NAME,
                "version": MODEL_VERSION,
                "expected_features": INPUT_SCHEMA,
            }
        ),
        200,
    )


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    payload = request.get_json(force=True)

    # Accept {"features": {...}} or direct {...}
    features_dict = payload.get("features", payload)

    if not isinstance(features_dict, dict):
        return (
            jsonify(
                {
                    "error": "Input payload must be a JSON object of feature key-value pairs."
                }
            ),
            400,
        )

    missing_cols = [col for col in INPUT_SCHEMA.keys() if col not in features_dict]
    if missing_cols:
        return (
            jsonify(
                {
                    "error": "Missing required input features.",
                    "missing_features": missing_cols,
                    "expected_schema": INPUT_SCHEMA,
                }
            ),
            400,
        )

    # Type conversion and DataFrame assembly
    row_data = {}
    z_scores = []

    for col, col_type in INPUT_SCHEMA.items():
        val = features_dict.get(col)

        if col_type == "float":
            try:
                num_val = float(val)
            except (TypeError, ValueError):
                return (
                    jsonify(
                        {
                            "error": f"Feature '{col}' expects a numeric float value, got '{val}'."
                        }
                    ),
                    400,
                )

            row_data[col] = num_val

            # Compute Z-score drift against Pair A's training baseline (Unit-VI)
            if col in BASELINE_STATS:
                mean = BASELINE_STATS[col]["mean"]
                std = BASELINE_STATS[col]["std"]
                z_scores.append(abs(num_val - mean) / std)
        else:
            row_data[col] = str(val)

    # Run inference (after all features have been collected)
    input_df = pd.DataFrame([row_data])
    model = load_registered_model()
    preds = model.predict(input_df)
    pred_val = preds[0].item() if hasattr(preds[0], "item") else preds[0]

    latency = time.time() - start_time
    INFERENCE_LATENCY.labels(model_name=MODEL_NAME, version=MODEL_VERSION).observe(
        latency
    )
    PREDICTION_COUNTER.labels(model_name=MODEL_NAME, version=MODEL_VERSION).inc()

    if isinstance(pred_val, (int, float)):
        PREDICTION_VALUE_GAUGE.labels(
            model_name=MODEL_NAME, version=MODEL_VERSION
        ).set(float(pred_val))

    mean_drift = float(np.mean(z_scores)) if z_scores else 0.0
    FEATURE_DRIFT_GAUGE.labels(model_name=MODEL_NAME, version=MODEL_VERSION).set(
        mean_drift
    )

    return (
        jsonify(
            {
                "model_name": MODEL_NAME,
                "version": MODEL_VERSION,
                "prediction": pred_val,
                "drift_zscore": round(mean_drift, 4),
                "drift_alert": bool(mean_drift > 3.0),
                "latency_ms": round(latency * 1000, 2),
            }
        ),
        200,
    )


if __name__ == "__main__":
    load_registered_model()
    app.run(host="0.0.0.0", port=8000)
