import json
import os
import threading
import uuid

from flask import Flask, jsonify, request

from trainer import run_training_job
from validator import validate_and_version_csv

app = Flask(__name__)

DATASETS_DIR = os.environ.get("DATASETS_DIR", "/app/shared_data/datasets")
JOBS_DIR = os.environ.get("JOBS_DIR", "/app/shared_data/jobs")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"service": "pair_a_service", "status": "healthy"}), 200


@app.route("/api/v1/validate", methods=["POST"])
def validate_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No CSV file uploaded under 'file' key."}), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename submitted."}), 400

    try:
        metadata = validate_and_version_csv(uploaded_file, DATASETS_DIR)
        return jsonify(metadata), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal validation error: {str(e)}"}), 500


@app.route("/api/v1/train", methods=["POST"])
def train_endpoint():
    payload = request.get_json(force=True) or {}
    dataset_id = payload.get("dataset_id")
    target_column = payload.get("target_column")
    problem_type = (payload.get("problem_type", "classification") or "classification").lower()

    if not dataset_id or not target_column:
        return jsonify({"error": "Both 'dataset_id' and 'target_column' are required."}), 400

    if problem_type not in ["classification", "regression"]:
        return jsonify({"error": "'problem_type' must be 'classification' or 'regression'."}), 400

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")

    initial_state = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "status": "RUNNING",
        "problem_type": problem_type,
        "target_column": target_column,
    }

    with open(job_file, "w", encoding="utf-8") as f:
        json.dump(initial_state, f, indent=2)

    worker = threading.Thread(
        target=run_training_job,
        args=(
            job_id,
            dataset_id,
            target_column,
            problem_type,
            DATASETS_DIR,
            JOBS_DIR,
            MLFLOW_TRACKING_URI,
        ),
        daemon=True,
    )
    worker.start()

    return jsonify({"job_id": job_id, "status": "RUNNING"}), 202


@app.route("/api/v1/jobs/<job_id>/summary", methods=["GET"])
def job_summary_endpoint(job_id):
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    if not os.path.exists(job_file):
        return jsonify({"error": f"Job {job_id} not found."}), 404

    with open(job_file, "r", encoding="utf-8") as f:
        job_data = json.load(f)

    return jsonify({
        "job_id": job_data.get("job_id"),
        "dataset_id": job_data.get("dataset_id"),
        "status": job_data.get("status"),
        "problem_type": job_data.get("problem_type"),
        "target_column": job_data.get("target_column"),
        "leaderboard": job_data.get("leaderboard", []),
        "error": job_data.get("error"),
    }), 200


@app.route("/api/v1/jobs/<job_id>/best-model", methods=["GET"])
def best_model_endpoint(job_id):
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    if not os.path.exists(job_file):
        return jsonify({"error": f"Job {job_id} not found."}), 404

    with open(job_file, "r", encoding="utf-8") as f:
        job_data = json.load(f)

    if job_data.get("status") != "COMPLETED":
        return jsonify({"error": "Job has not completed yet.", "status": job_data.get("status")}), 400

    return jsonify(job_data.get("best_model")), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

