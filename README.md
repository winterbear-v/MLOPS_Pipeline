# Model Factory: An End-to-End Automated MLOps Platform

Model Factory converts a tabular CSV dataset into a trained, evaluated,
registered, and deployable machine-learning model with minimal manual
training or serving code. It validates and versions data, compares
scikit-learn models, tracks experiments with MLflow, deploys the winning
model in a Docker container, and monitors latency and feature drift
using Prometheus and Grafana.

> **Scope:** Academic/laboratory platform for tabular classification and
> regression. It is not a hardened production deployment.

## Contents

-   [Features](#features)
-   [Architecture](#architecture)
-   [Technology Stack](#technology-stack)
-   [Workflow](#workflow)
-   [Model Selection](#model-selection)
-   [Services and Ports](#services-and-ports)
-   [API Overview](#api-overview)
-   [Getting Started](#getting-started)
-   [Testing](#testing)
-   [Project Structure](#project-structure)
-   [Limitations and Security](#limitations-and-security)

## Features

-   **CSV validation and privacy filtering:** Rejects empty or very
    small datasets (fewer than 15 rows), removes empty columns, and
    drops likely PII columns based on column-name patterns and
    email/phone value patterns.
-   **Dataset lineage:** Computes a SHA-256 hash and attempts DVC
    tracking. If DVC is unavailable, writes a `.dvc.json` manifest
    containing dataset ID, filename, hash, and size.
-   **Automated preprocessing:** Numeric columns use median imputation
    and `StandardScaler`; categorical columns use most-frequent-value
    imputation and one-hot encoding with unknown categories ignored.
-   **Model comparison:** Trains a fixed set of candidate scikit-learn
    models and selects a winner using a primary metric.
-   **Experiment tracking and registry:** Logs parameters, metrics,
    training time, and artifacts to MLflow, then registers the winning
    model.
-   **Containerized inference:** Starts a dedicated Docker container for
    each deployed model, validates input features, and exposes a JSON
    prediction endpoint.
-   **Monitoring:** Exposes prediction count, inference latency, and
    feature-drift metrics for Prometheus and Grafana.
-   **Resource governance:** Applies an active-container limit and
    time-to-live cleanup policy.
-   **CI and tests:** Uses pytest and a Jenkins pipeline for testing,
    image builds, Compose startup, and health checks.

## Architecture

The platform is divided into two application pairs integrated through
REST APIs and a shared Docker volume.

``` text
User / Browser
     |
Streamlit UI (:8501)
     |--------------------------|
     v                          v
Pair A API (:5001)        Pair B Controller (:5002)
     |                          |
     +--> MLflow (:5000)        +--> Docker Engine
     |                          |       |
     +--> Shared data volume    |   Model-serving containers
                                |       |
                                +--> Prometheus (:9090)
                                         |
                                     Grafana (:3000)

Network: model_factory_net
Volume:  model_factory_shared_data
```

### Pair A --- Data, Training, and Registry

Pair A validates and cleans datasets, creates lineage metadata, trains
candidate models, logs runs to MLflow, and saves the winning model and
deployment metadata. It does not manage Docker deployments.

### Pair B --- Deploy, Serve, Monitor, and Cleanup

Pair B consumes the winning-model metadata, starts serving containers,
maintains Prometheus's dynamic target list, lists deployments, and
removes expired or excess containers. It does not retrain models.

The integration contract is a job/model JSON payload containing fields
such as registered model name, version, artifact path, input schema, and
baseline statistics.

## Technology Stack

  -----------------------------------------------------------------------
  Area                                Technologies
  ----------------------------------- -----------------------------------
  UI                                  Streamlit

  REST APIs                           Flask

  ML and preprocessing                pandas, scikit-learn

  Experiment tracking and registry    MLflow

  Dataset lineage                     SHA-256, DVC when available, JSON
                                      manifest fallback

  Containerization                    Docker, Docker Compose, Docker SDK

  Monitoring                          Prometheus, Grafana

  Testing and CI                      pytest, Jenkins
  -----------------------------------------------------------------------

## Workflow

1.  Upload a CSV and validate it. The service checks the dataset,
    removes likely PII columns, and returns dataset metadata and a
    SHA-256 hash.
2.  Select the target column and choose classification or regression.
3.  Start training. The API returns HTTP `202 Accepted` and a job ID
    while training runs asynchronously.
4.  Poll the job-summary endpoint until the job completes or fails.
5.  Compare candidate models and log parameters, metrics, and artifacts
    to MLflow.
6.  Register the winning model and save its portable artifact and
    metadata.
7.  Deploy the winner. Pair B starts a dedicated container and returns
    the prediction URL and host port.
8.  Submit feature values to the prediction endpoint and receive a
    prediction, latency, drift score, and drift-alert status.
9.  Inspect metrics in Grafana and clean up old deployments.

## Model Selection

  -----------------------------------------------------------------------
  Task              Candidate models  Primary metric    Secondary metric
  ----------------- ----------------- ----------------- -----------------
  Classification    Logistic          Weighted F1       Accuracy
                    Regression,                         
                    Random Forest                       
                    Classifier,                         
                    Gradient Boosting                   
                    Classifier                          

  Regression        Ridge, Random     R²                RMSE
                    Forest Regressor,                   
                    Gradient Boosting                   
                    Regressor                           
  -----------------------------------------------------------------------

The model with the highest primary metric is selected. This is a bounded
AutoML-style comparison, not a large hyperparameter search.

The numeric and categorical preprocessing branches are combined with
`ColumnTransformer` and attached to the estimator in a scikit-learn
`Pipeline`. Saving the full pipeline allows the serving service to
process raw feature values without reimplementing preprocessing.

## Monitoring and Resource Management

The serving service exposes `/metrics` for Prometheus. The report
describes a prediction Counter, inference-latency Histogram, and
feature-drift Z-score Gauge. Prometheus discovers model servers through
a `targets.json` file updated after deployment or cleanup.

For numeric features, the system records training means and standard
deviations. At inference it computes absolute Z-scores and averages
available values. A mean score greater than `3.0` sets `drift_alert` to
true. This is a lightweight educational signal, not a comprehensive
statistical drift test.

The cleanup controller uses: - `MAX_ACTIVE_MODELS=3` -
`CONTAINER_TTL_SECONDS=3600`

These settings limit active model containers and expire old deployments.
Confirm the actual values in the project's configuration before changing
them.

## Services and Ports

  ------------------------------------------------------------------------
  Service                                       Port Purpose
  --------------------- ---------------------------- ---------------------
  Streamlit frontend                          `8501` Main user workflow

  Pair A API                                  `5001` Validation, training,
                                                     job status,
                                                     best-model metadata

  Pair B controller                           `5002` Deployments and
                                                     cleanup

  MLflow                                      `5000` Experiment tracking
                                                     and model registry

  Prometheus                                  `9090` Metrics collection

  Grafana                                     `3000` Monitoring dashboards

  Model server               `8000` inside container Prediction and
                                                     metrics

  Deployed model host                    `8001–8099` Access individual
  ports                                              model deployments
  ------------------------------------------------------------------------

Use the actual configuration in the project's Compose file if it differs
from this report.

## API Overview

  --------------------------------------------------------------------------------------
  Method            Endpoint                         Service           Purpose
  ----------------- -------------------------------- ----------------- -----------------
  `GET`             `/health`                        Pair A and Pair B Health check

  `POST`            `/api/v1/validate`               Pair A            Upload CSV using
                                                                       multipart field
                                                                       `file`

  `POST`            `/api/v1/train`                  Pair A            Start training

  `GET`             `/api/v1/jobs/<id>/summary`      Pair A            Job status and
                                                                       leaderboard

  `GET`             `/api/v1/jobs/<id>/best-model`   Pair A            Retrieve winner
                                                                       metadata

  `POST`            `/api/v1/deploy`                 Pair B            Start a serving
                                                                       container

  `GET`             `/api/v1/deployments`            Pair B            List active
                                                                       deployments

  `POST`            `/api/v1/cleanup`                Pair B            Trigger cleanup

  `POST`            `/predict`                       Model server      Return prediction
                                                                       and monitoring
                                                                       fields

  `GET`             `/metrics`                       Model server      Expose Prometheus
                                                                       metrics
  --------------------------------------------------------------------------------------

A prediction request can use a `features` object:

``` json
{
  "features": {
    "feature_a": 12.5,
    "feature_b": "category_value"
  }
}
```

Replace the example fields with those expected by the selected model's
input schema. The response includes the prediction, mean drift Z-score,
drift-alert flag, and inference latency in milliseconds.

## Getting Started

### Prerequisites

-   Docker Engine and Docker Compose (or Docker Desktop with Compose).
-   Git, if cloning the repository.
-   A tabular CSV dataset for classification or regression.
-   A browser.

### Start the platform

From the repository root, where the Compose file is located:

``` bash
docker compose up --build -d
docker compose ps
```

Open the Streamlit UI at `http://localhost:8501`.

Other endpoints described in the report: - Pair A health:
`http://localhost:5001/health` - Pair B health:
`http://localhost:5002/health` - MLflow: `http://localhost:5000` -
Prometheus: `http://localhost:9090` - Grafana: `http://localhost:3000`

To inspect logs:

``` bash
docker compose logs -f
```

To stop services:

``` bash
docker compose down
```

The named shared volume stores datasets, job status, MLflow artifacts,
and model folders. `docker compose down` normally preserves named
volumes. Avoid `docker compose down -v` unless you intend to delete
stored volume data.

The report documents `admin / admin` as the default Grafana laboratory
login. Treat this as demonstration-only and change it before any shared
or production use.

## Testing

The report's Jenkins pipeline runs tests with:

``` bash
pytest tests/ -v
```

The described tests cover PII detection, preprocessing pipeline
behavior, baseline statistics, Pair B request validation, and health
checks. The Jenkins pipeline also builds the base serving image,
builds/starts Compose services, and checks service health endpoints.

## Project Structure

The report identifies these main modules; confirm exact filenames in the
repository.

``` text
.
├── pair_a_service/
│   ├── app.py              # Validation, training, job and best-model APIs
│   ├── validator.py        # CSV checks, PII removal, hash and lineage
│   ├── pipeline.py         # Preprocessing and sklearn pipeline
│   ├── trainer.py          # Background training and MLflow tracking
│   └── tests/test_pair_a.py
├── pair_b_controller/
│   ├── app.py              # Deploy, list and cleanup APIs
│   └── cleanup.py          # TTL and active-container limits
├── base_model_server/
│   └── app.py              # Model loading, prediction, metrics and drift
├── pair_b_frontend/
│   └── app.py              # Streamlit UI
├── tests/test_pair_b.py
└── Jenkinsfile
```

## Limitations and Security

The report identifies the following limitations: - A fixed set of three
candidate models is evaluated per task. - Fairness checks use one proxy
subgroup attribute and a simple performance gap; they are not a legal or
comprehensive audit. - Drift uses a per-request Z-score, not a windowed
population test. - Training uses in-process background threads and JSON
files rather than a distributed task queue. - Mounting
`/var/run/docker.sock` grants powerful control over the Docker host and
is intended for a controlled lab. - Default Grafana credentials are for
demonstration only. - Kubernetes, autoscaling clusters, deep learning,
NLP, computer vision, and full production cloud deployment are out of
scope.

**Do not expose this setup as-is to untrusted users or the public
internet.** Production use would require a security review covering
authentication, authorization, secrets, network exposure, artifact
trust, container isolation, resource quotas, and monitoring.

## Learning Outcomes

This project demonstrates data validation and lineage, reproducible
preprocessing, automated model comparison, experiment tracking, model
registration, containerized inference, REST integration, latency and
drift monitoring, resource governance, and CI checks.

------------------------------------------------------------------------

**Model Factory --- Data in. Model selected. Service deployed.
Performance monitored.**
