import os

import time

import json

import threading

import docker

from flask import Flask, request, jsonify

from cleanup import cleanup_old_deployments, update_prometheus_targets

app = Flask(__name__)

DOCKER_NETWORK = os.environ.get("DOCKER_NETWORK", "model_factory_net")

BASE_SERVER_IMAGE = os.environ.get("BASE_SERVER_IMAGE", "model-factory-base-server:latest")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

TARGETS_FILE = os.environ.get("TARGETS_FILE", "/app/monitoring/prometheus/targets.json")

HOST_SHARED_DATA_VOLUME = os.environ.get("SHARED_VOLUME_NAME", "model_factory_shared_data")

MAX_ACTIVE_MODELS = int(os.environ.get("MAX_ACTIVE_MODELS", "3"))

CONTAINER_TTL_SECONDS = int(os.environ.get("CONTAINER_TTL_SECONDS", "3600"))


def get_docker_client():

    return docker.from_env()


def find_available_port(client, start_port=8001, end_port=8099) -> int:

    used_ports = set()

    for c in client.containers.list():

        ports = c.attrs.get("NetworkSettings", {}).get("Ports", {})

        for _, bindings in ports.items():

            if bindings:

                for b in bindings:

                    used_ports.add(int(b["HostPort"]))

    for port in range(start_port, end_port + 1):

        if port not in used_ports:

            return port

    raise RuntimeError("No free host ports available in range 8001-8099.")


def background_cleanup_loop():

    while True:

        time.sleep(60)

        try:

            client = get_docker_client()

            cleanup_old_deployments(

                client,

                TARGETS_FILE,

                max_active_containers=MAX_ACTIVE_MODELS,

                max_age_seconds=CONTAINER_TTL_SECONDS,

            )

        except Exception:

            pass


@app.route("/health", methods=["GET"])

def health():

    return jsonify({"service": "pair_b_controller", "status": "healthy"}), 200


@app.route("/api/v1/deploy", methods=["POST"])

def deploy_model():

    payload = request.get_json(force=True)

    model_name = payload.get("registered_model_name")

    version = str(payload.get("version", "1"))

    artifact_path = payload.get("artifact_path", "")

    input_schema = payload.get("input_schema", {})

    baseline_stats = payload.get("baseline_stats", {})

    if not model_name:

        return jsonify({"error": "'registered_model_name' is required."}), 400

    try:

        client = get_docker_client()

        container_name = f"served_{model_name}_v{version}".lower().replace("-", "_")

        # Remove existing container with the same name if redeploying

        try:

            existing = client.containers.get(container_name)

            existing.remove(force=True)

        except docker.errors.NotFound:

            pass

        host_port = find_available_port(client)

        now_ts = str(time.time())

        container = client.containers.run(

            image=BASE_SERVER_IMAGE,

            name=container_name,

            detach=True,

            network=DOCKER_NETWORK,

            ports={"8000/tcp": host_port},

            volumes={HOST_SHARED_DATA_VOLUME: {"bind": "/app/shared_data", "mode": "ro"}},

            environment={

                "MODEL_NAME": model_name,

                "MODEL_VERSION": version,

                "ARTIFACT_PATH": artifact_path,

                "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,

                "INPUT_SCHEMA": json.dumps(input_schema),

                "BASELINE_STATS": json.dumps(baseline_stats),

            },

            labels={

                "app": "model-factory-deployment",

                "model_name": model_name,

                "model_version": version,

                "host_port": str(host_port),

                "created_at_ts": now_ts,

            },

        )

        # Enforce cleanup policies & refresh Prometheus scrape targets

        removed = cleanup_old_deployments(

            client,

            TARGETS_FILE,

            max_active_containers=MAX_ACTIVE_MODELS,

            max_age_seconds=CONTAINER_TTL_SECONDS,

        )

        return jsonify({

            "status": "DEPLOYED",

            "container_id": container.short_id,

            "container_name": container_name,

            "model_name": model_name,

            "version": version,

            "host_port": host_port,

            "predict_url_external": f"http://localhost:{host_port}/predict",

            "predict_url_internal": f"http://{container_name}:8000/predict",

            "metrics_url": f"http://localhost:{host_port}/metrics",

            "grafana_dashboard_url": f"http://localhost:3000/d/model-factory-mon/model-factory-monitoring?var-model_name={model_name}",

            "cleaned_up_containers": removed,

        }), 201

    except Exception as e:

        return jsonify({"error": f"Deployment failed: {str(e)}"}), 500


@app.route("/api/v1/deployments", methods=["GET"])

def list_deployments():

    try:

        client = get_docker_client()

        containers = client.containers.list(

            filters={"label": "app=model-factory-deployment"}

        )

        deployments = []

        for c in containers:

            port = c.labels.get("host_port", "unknown")

            m_name = c.labels.get("model_name", "unknown")

            deployments.append({

                "container_id": c.short_id,

                "container_name": c.name,

                "model_name": m_name,

                "version": c.labels.get("model_version", "1"),

                "status": c.status,

                "host_port": port,

                "created_at_ts": float(c.labels.get("created_at_ts", "0")),

                "predict_url_external": f"http://localhost:{port}/predict",

                "predict_url_internal": f"http://{c.name}:8000/predict",

                "grafana_url": f"http://localhost:3000/d/model-factory-mon/model-factory-monitoring?var-model_name={m_name}",

            })

        return jsonify({"active_deployments": deployments}), 200

    except Exception as e:

        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/cleanup", methods=["POST"])

def trigger_manual_cleanup():

    payload = request.get_json(silent=True) or {}

    max_keep = int(payload.get("max_keep", 1))

    max_age = int(payload.get("max_age_seconds", 3600))

    try:

        client = get_docker_client()

        removed = cleanup_old_deployments(

            client, TARGETS_FILE, max_active_containers=max_keep, max_age_seconds=max_age

        )

        return jsonify({"status": "CLEANUP_EXECUTED", "removed_containers": removed}), 200

    except Exception as e:

        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":

    cleaner = threading.Thread(target=background_cleanup_loop, daemon=True)

    cleaner.start()

    app.run(host="0.0.0.0", port=5002)
