from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


MODEL_ARTIFACT_PATH = Path("artifacts/models/matrix_factorization.pkl")


def test_health_check():
    if not MODEL_ARTIFACT_PATH.exists():
        pytest.skip("Model artifact not available in CI environment")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint_returns_recommendations():
    if not MODEL_ARTIFACT_PATH.exists():
        pytest.skip("Model artifact not available in CI environment")

    with TestClient(app) as client:
        response = client.post(
            "/recommend",
            json={"user_id": 1, "k": 5},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 1
    assert data["model_name"] == "matrix_factorization"
    assert len(data["recommendations"]) == 5


def test_model_info_endpoint_returns_metadata():
    if not MODEL_ARTIFACT_PATH.exists():
        pytest.skip("Model artifact not available in CI environment")

    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "matrix_factorization"
    assert data["model_path"] == "artifacts/models/matrix_factorization.pkl"
    assert data["num_users"] > 0
    assert data["num_items"] > 0
    assert data["num_factors"] == 20

def test_metrics_endpoint_returns_response():
    if not MODEL_ARTIFACT_PATH.exists():
        pytest.skip("Model artifact not available in CI environment")

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)