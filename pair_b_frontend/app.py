import os
import time
import requests
import pandas as pd
import streamlit as st

PAIR_A_URL = os.environ.get("PAIR_A_URL", "http://pair-a-service:5001")
PAIR_B_URL = os.environ.get("PAIR_B_URL", "http://pair-b-controller:5002")

st.set_page_config(page_title="Model Factory MLOps Platform", layout="wide")

st.title("🏭 Model Factory: End-to-End Automated MLOps Platform")
st.markdown(
    "Upload any tabular CSV dataset to automatically validate data, strip PII, "
    "train & compare models, register the winner in **MLflow**, and deploy a "
    "monitored **Docker API container**."
)

if "validation_data" not in st.session_state:
    st.session_state["validation_data"] = None

if "job_id" not in st.session_state:
    st.session_state["job_id"] = None

if "best_model" not in st.session_state:
    st.session_state["best_model"] = None

if "last_deployment" not in st.session_state:
    st.session_state["last_deployment"] = None

tab1, tab2, tab3 = st.tabs([
    "1️⃣ Upload, Validate & Train (Pair A)",
    "2️⃣ Dynamic Container Deploy & Live Inference (Pair B)",
    "3️⃣ Active Containers, Grafana Monitoring & Cleanup",
])

with tab1:
    st.subheader("Step 1: Upload Tabular Dataset (CSV)")

    uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

    if uploaded_file is not None and st.button("🔍 Validate CSV & Run Privacy/DVC Gate"):
        with st.spinner("Validating schema, dropping PII columns, and versioning with DVC..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv",
                )
            }

            resp = requests.post(
                f"{PAIR_A_URL}/api/v1/validate",
                files=files,
                timeout=30,
            )

            if resp.status_code == 200:
                st.session_state["validation_data"] = resp.json()
                st.success(
                    f"Dataset `{st.session_state['validation_data']['dataset_id']}` validated!"
                )
            else:
                st.error(resp.json().get("error", "Validation failed"))

    val_data = st.session_state["validation_data"]

    if val_data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset ID", val_data["dataset_id"])
        col2.metric("Cleaned Rows", val_data["rows"])
        col3.metric(
            "Dropped PII Columns (Unit-VIII)",
            len(val_data["dropped_pii_columns"]),
        )

        if val_data["dropped_pii_columns"]:
            st.warning(
                "🔒 Privacy Protection: Automatically stripped PII columns: "
                f"`{val_data['dropped_pii_columns']}`"
            )

        st.caption(
            "DVC / SHA-256 Data Lineage Hash: "
            f"`{val_data['dataset_version_hash']}`"
        )

        st.divider()
        st.subheader("Step 2: Configure Automated Model Training")

        c1, c2 = st.columns(2)
        target_col = c1.selectbox(
            "Select Target Column to Predict",
            options=val_data["columns"],
        )
        prob_type = c2.selectbox(
            "Select Problem Type",
            options=["classification", "regression"],
        )

        if st.button("🚀 Start Multi-Model AutoML Pipeline"):
            payload = {
                "dataset_id": val_data["dataset_id"],
                "target_column": target_col,
                "problem_type": prob_type,
            }

            resp = requests.post(
                f"{PAIR_A_URL}/api/v1/train",
                json=payload,
                timeout=15,
            )

            if resp.status_code == 202:
                st.session_state["job_id"] = resp.json()["job_id"]
                st.session_state["best_model"] = None
            else:
                st.error(
                    resp.json().get(
                        "error",
                        "Failed to start training",
                    )
                )

    if st.session_state["job_id"]:
        st.divider()
        st.subheader(
            f"Step 3: Experiment Leaderboard "
            f"(`{st.session_state['job_id']}`)"
        )

        status_placeholder = st.empty()

        while True:
            sum_resp = requests.get(
                f"{PAIR_A_URL}/api/v1/jobs/"
                f"{st.session_state['job_id']}/summary",
                timeout=10,
            )

            summary = sum_resp.json()
            status = summary.get("status")

            if status == "RUNNING":
                status_placeholder.info(
                    "⏳ Training Scikit-Learn Pipelines & logging to MLflow... "
                    "Polling every 2s."
                )
                time.sleep(2)

            elif status == "COMPLETED":
                status_placeholder.success(
                    "✅ Training Complete & Best Model Registered in MLflow!"
                )

                df_lead = pd.DataFrame(summary["leaderboard"])
                st.dataframe(df_lead, use_container_width=True)

                best_resp = requests.get(
                    f"{PAIR_A_URL}/api/v1/jobs/"
                    f"{st.session_state['job_id']}/best-model",
                    timeout=10,
                )

                if best_resp.status_code == 200:
                    st.session_state["best_model"] = best_resp.json()

                break

            else:
                status_placeholder.error(
                    f"❌ Training Failed: {summary.get('error')}"
                )
                break

with tab2:
    best = st.session_state["best_model"]

    if not best:
        st.info(
            "Complete model training in Tab 1 first to unlock "
            "dynamic container deployment."
        )
    else:
        st.subheader("Winning Model Registered in MLflow")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Registered Name", best["registered_model_name"])
        m2.metric("Algorithm", best["algorithm"])
        m3.metric(best["primary_metric"], best["primary_score"])
        m4.metric("Fairness Gap", best["fairness_disparity_gap"])

        if st.button("🐳 Build & Spin Up Dedicated Model Container"):
            with st.spinner(
                "Spawning isolated Docker container and registering "
                "with Prometheus..."
            ):
                dep_resp = requests.post(
                    f"{PAIR_B_URL}/api/v1/deploy",
                    json=best,
                    timeout=30,
                )

                if dep_resp.status_code == 201:
                    st.session_state["last_deployment"] = dep_resp.json()
                    st.success("🎉 Dedicated Model API Container is Live!")
                else:
                    st.error(
                        dep_resp.json().get(
                            "error",
                            "Deployment failed",
                        )
                    )

        dep = st.session_state["last_deployment"]

        if dep:
            st.code(
                f"POST {dep['predict_url_external']}",
                language="http",
            )

            st.markdown(
                f"📊 **[Open Dedicated Grafana Dashboard for "
                f"`{dep['model_name']}`]({dep['grafana_dashboard_url']})**"
            )

            st.divider()
            st.subheader("Live Prediction & Data Drift Tester")

            input_schema = best["input_schema"]
            baselines = best.get("baseline_stats", {})
            form_inputs = {}

            with st.form("prediction_form"):
                cols = st.columns(3)

                for idx, (feat, ftype) in enumerate(input_schema.items()):
                    with cols[idx % 3]:
                        if ftype == "float":
                            default_val = float(
                                baselines.get(feat, {}).get("mean", 0.0)
                            )
                            form_inputs[feat] = st.number_input(
                                f"{feat} (numeric)",
                                value=default_val,
                            )
                        else:
                            form_inputs[feat] = st.text_input(
                                f"{feat} (categorical)",
                                value="sample",
                            )

                submitted = st.form_submit_button(
                    "⚡ Send Live Prediction Request"
                )

            if submitted:
                time.sleep(0.5)

                pred_resp = requests.post(
                    dep["predict_url_internal"],
                    json={"features": form_inputs},
                    timeout=10,
                )

                if pred_resp.status_code == 200:
                    res = pred_resp.json()

                    r1, r2, r3 = st.columns(3)
                    r1.metric(
                        "Predicted Output",
                        str(res["prediction"]),
                    )
                    r2.metric(
                        "Feature Drift Z-Score",
                        res["drift_zscore"],
                    )
                    r3.metric(
                        "Inference Latency",
                        f"{res['latency_ms']} ms",
                    )

                    if res["drift_alert"]:
                        st.error(
                            "🚨 Data Drift Alert: Input features deviate "
                            "> 3 standard deviations from training baseline!"
                        )
                else:
                    st.error(pred_resp.text)

with tab3:
    st.subheader("Active Deployed Model Containers")

    if st.button("🔄 Refresh Active Deployments"):
        pass

    list_resp = requests.get(
        f"{PAIR_B_URL}/api/v1/deployments",
        timeout=10,
    )

    if list_resp.status_code == 200:
        deployments = list_resp.json().get(
            "active_deployments",
            [],
        )

        if deployments:
            st.dataframe(
                pd.DataFrame(deployments),
                use_container_width=True,
            )
        else:
            st.info("No deployed model containers are currently running.")

    st.divider()
    st.subheader("Automated & Manual Cleanup of Old Deployments")

    keep_n = st.slider(
        "Keep only the newest N containers (delete older ones)",
        min_value=0,
        max_value=5,
        value=1,
    )

    if st.button("🧹 Execute Container Cleanup Now"):
        clean_resp = requests.post(
            f"{PAIR_B_URL}/api/v1/cleanup",
            json={
                "max_keep": keep_n,
                "max_age_seconds": 3600,
            },
            timeout=15,
        )

        st.json(clean_resp.json())
