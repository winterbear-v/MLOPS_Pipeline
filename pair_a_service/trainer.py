import os
import time

import json

import numpy as np

import pandas as pd

import mlflow

import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    f1_score,

    accuracy_score,

    mean_squared_error,

    r2_score,

)

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.ensemble import (

    RandomForestClassifier,

    GradientBoostingClassifier,

    RandomForestRegressor,

    GradientBoostingRegressor,

)

from pipeline import build_feature_pipeline, compute_baseline_statistics


def evaluate_subgroup_fairness(X_test: pd.DataFrame, y_test, y_pred, problem_type: str) -> float:

    """

    Unit-VIII Fairness & Bias Evaluation:

    Measures performance parity across groups of the first categorical column

    (or median split of the first numerical column if no categorical exists).

    Returns the absolute performance gap (closer to 0.0 means fairer).

    """

    try:

        cat_cols = X_test.select_dtypes(exclude=[np.number]).columns.tolist()

        if cat_cols:

            sensitive_series = X_test[cat_cols[0]].astype(str)

        else:

            num_col = X_test.columns[0]

            median_val = X_test[num_col].median()

            sensitive_series = (X_test[num_col] >= median_val).astype(str)

        group_scores = []

        for group_val in sensitive_series.unique():

            mask = (sensitive_series == group_val).values

            if mask.sum() < 3:

                continue

            y_t_sub = np.array(y_test)[mask]

            y_p_sub = np.array(y_pred)[mask]

            if problem_type == "classification":

                score = accuracy_score(y_t_sub, y_p_sub)

            else:

                score = r2_score(y_t_sub, y_p_sub) if len(np.unique(y_t_sub)) > 1 else 0.0

            group_scores.append(score)

        if len(group_scores) >= 2:

            return round(float(max(group_scores) - min(group_scores)), 4)

        return 0.0

    except Exception:

        return 0.0


def run_training_job(

    job_id: str,

    dataset_id: str,

    target_column: str,

    problem_type: str,

    datasets_dir: str,

    jobs_dir: str,

    mlflow_tracking_uri: str,

):

    """Background worker function that trains models, logs to MLflow, and registers the winner."""

    job_file = os.path.join(jobs_dir, f"{job_id}.json")

    try:

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        experiment_name = f"experiment_{dataset_id}"

        mlflow.set_experiment(experiment_name)

        csv_path = os.path.join(datasets_dir, f"{dataset_id}.csv")

        df = pd.read_csv(csv_path)

        if target_column not in df.columns:

            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        df = df.dropna(subset=[target_column])

        X = df.drop(columns=[target_column])

        y = df[target_column]

        # Encode string target labels cleanly if classification target is text

        target_mapping = None

        if problem_type == "classification" and y.dtype == object:

            unique_classes = sorted(y.astype(str).unique().tolist())

            target_mapping = {label: idx for idx, label in enumerate(unique_classes)}

            y = y.astype(str).map(target_mapping)

        X_train, X_test, y_train, y_test = train_test_split(

            X, y, test_size=0.2, random_state=42

        )

        # Schema & drift baseline preparation

        input_schema = {

            col: ("float" if col in X.select_dtypes(include=[np.number]).columns else "string")

            for col in X.columns

        }

        baseline_stats = compute_baseline_statistics(X_train)

        if problem_type == "classification":

            candidates = {

                "LogisticRegression": LogisticRegression(max_iter=500),

                "RandomForestClassifier": RandomForestClassifier(n_estimators=80, random_state=42),

                "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=80, random_state=42),

            }

            primary_metric_name = "f1_weighted"

        else:

            candidates = {

                "RidgeRegression": Ridge(alpha=1.0),

                "RandomForestRegressor": RandomForestRegressor(n_estimators=80, random_state=42),

                "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=80, random_state=42),

            }

            primary_metric_name = "r2_score"

        leaderboard = []

        best_score = -float("inf")

        best_model_info = None

        best_pipeline = None

        for model_name, estimator in candidates.items():

            start_t = time.time()

            with mlflow.start_run(run_name=f"{model_name}_{job_id}") as run:

                pipeline = build_feature_pipeline(X_train, estimator)

                pipeline.fit(X_train, y_train)

                preds = pipeline.predict(X_test)

                duration = round(time.time() - start_t, 3)

                fairness_gap = evaluate_subgroup_fairness(X_test, y_test, preds, problem_type)

                if problem_type == "classification":

                    primary_score = round(float(f1_score(y_test, preds, average="weighted")), 4)

                    secondary_score = round(float(accuracy_score(y_test, preds)), 4)

                    mlflow.log_metric("f1_weighted", primary_score)

                    mlflow.log_metric("accuracy", secondary_score)

                else:

                    primary_score = round(float(r2_score(y_test, preds)), 4)

                    secondary_score = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)

                    mlflow.log_metric("r2_score", primary_score)

                    mlflow.log_metric("rmse", secondary_score)

                mlflow.log_metric("fairness_disparity_gap", fairness_gap)

                mlflow.log_metric("training_time_sec", duration)

                mlflow.log_param("problem_type", problem_type)

                mlflow.log_param("target_column", target_column)

                mlflow.log_param("dataset_id", dataset_id)

                signature = infer_signature(X_train, preds)

                mlflow.sklearn.log_model(

                    sk_model=pipeline,

                    artifact_path="model",

                    signature=signature,

                )

                entry = {

                    "algorithm": model_name,

                    "run_id": run.info.run_id,

                    "primary_metric": primary_metric_name,

                    "primary_score": primary_score,

                    "secondary_score": secondary_score,

                    "fairness_disparity_gap": fairness_gap,

                    "training_time_sec": duration,

                }

                leaderboard.append(entry)

                if primary_score > best_score:

                    best_score = primary_score

                    best_pipeline = pipeline

                    best_model_info = entry

        # Register the winning model in MLflow Model Registry

        registered_model_name = f"model_{dataset_id}"

        best_run_id = best_model_info["run_id"]

        model_uri = f"runs:/{best_run_id}/model"

        reg_result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

        # Also save a direct portable artifact path for guaranteed container loading

        direct_model_dir = os.path.join(datasets_dir, f"registered_{registered_model_name}_v{reg_result.version}")

        mlflow.sklearn.save_model(best_pipeline, direct_model_dir)

        job_state = {

            "job_id": job_id,

            "dataset_id": dataset_id,

            "status": "COMPLETED",

            "problem_type": problem_type,

            "target_column": target_column,

            "target_mapping": target_mapping,

            "leaderboard": sorted(leaderboard, key=lambda x: x["primary_score"], reverse=True),

            "best_model": {

                "registered_model_name": registered_model_name,

                "version": str(reg_result.version),

                "run_id": best_run_id,

                "mlflow_uri": f"models:/{registered_model_name}/{reg_result.version}",

                "artifact_path": direct_model_dir,

                "algorithm": best_model_info["algorithm"],

                "primary_metric": primary_metric_name,

                "primary_score": best_model_info["primary_score"],

                "fairness_disparity_gap": best_model_info["fairness_disparity_gap"],

                "input_schema": input_schema,

                "baseline_stats": baseline_stats,

            },

        }

        with open(job_file, "w", encoding="utf-8") as f:

            json.dump(job_state, f, indent=2)

    except Exception as e:

        error_state = {

            "job_id": job_id,

            "dataset_id": dataset_id,

            "status": "FAILED",

            "error": str(e),

        }

        with open(job_file, "w", encoding="utf-8") as f:

            json.dump(error_state, f, indent=2)
