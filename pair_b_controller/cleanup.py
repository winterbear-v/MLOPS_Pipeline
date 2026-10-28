import time
import json

import os

import docker


def update_prometheus_targets(docker_client, targets_file_path: str):
    """Rebuilds Prometheus file_sd_configs targets.json from currently active model containers."""

    try:

        containers = docker_client.containers.list(
            filters={"label": "app=model-factory-deployment", "status": "running"}
        )

        target_entries = []

        for c in containers:

            model_name = c.labels.get("model_name", c.name)

            version = c.labels.get("model_version", "1")

            target_entries.append(
                {
                    "targets": [f"{c.name}:8000"],
                    "labels": {
                        "model_name": model_name,
                        "version": version,
                        "container_name": c.name,
                    },
                }
            )

        os.makedirs(os.path.dirname(targets_file_path), exist_ok=True)

        with open(targets_file_path, "w", encoding="utf-8") as f:

            json.dump(target_entries, f, indent=2)

    except Exception as e:

        print(f"[Cleanup] Error updating Prometheus targets: {e}")


def cleanup_old_deployments(
    docker_client,
    targets_file_path: str,
    max_active_containers: int = 3,
    max_age_seconds: int = 3600,
) -> list:
    """

    Stops and removes deployed model containers that either:

    1. Exceed the TTL (max_age_seconds), OR

    2. Exceed the maximum allowed active model containers (keeps newest N).

    """

    removed = []

    try:

        containers = docker_client.containers.list(
            all=True, filters={"label": "app=model-factory-deployment"}
        )

        now = time.time()

        # Sort newest first based on created_at label

        containers_sorted = sorted(
            containers,
            key=lambda c: float(c.labels.get("created_at_ts", "0")),
            reverse=True,
        )

        for idx, container in enumerate(containers_sorted):

            created_ts = float(container.labels.get("created_at_ts", str(now)))

            age = now - created_ts

            if idx >= max_active_containers or age > max_age_seconds:

                cname = container.name

                container.remove(force=True)

                removed.append(cname)

        update_prometheus_targets(docker_client, targets_file_path)

    except Exception as e:

        print(f"[Cleanup] Error cleaning containers: {e}")

    return removed
