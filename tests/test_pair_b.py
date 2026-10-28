import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../pair_b_controller",
        )
    ),
)

from app import app


def test_pair_b_controller_health_endpoint():
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_pair_b_deploy_requires_model_name():
    client = app.test_client()
    response = client.post("/api/v1/deploy", json={})

    assert response.status_code == 400
