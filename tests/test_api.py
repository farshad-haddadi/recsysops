from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


MATRIX_MODEL_PATH = Path("artifacts/models/matrix_factorization.pkl")
TWO_TOWER_MODEL_PATH = Path("artifacts/models/two_tower.pt")


def test_health_check():
    if not MATRIX_MODEL_PATH.exists() or not TWO_TOWER_MODEL_PATH.exists():
        pytest.skip("Model artifacts not available in CI environment")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint_returns_matrix_factorization_recommendations():
    if not MATRIX_MODEL_PATH.exists() or not TWO_TOWER_MODEL_PATH.exists():
        pytest.skip("Model artifacts not available in CI environment")

    with TestClient(app) as client:
        response = client.post(
            "/recommend",
            json={
                "user_id": 1,
                "k": 5,
                "model_name": "matrix_factorization",
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 1
    assert data["model_name"] == "matrix_factorization"
    assert len(data["recommendations"]) == 5


def test_recommend_endpoint_returns_two_tower_recommendations():
    if not MATRIX_MODEL_PATH.exists() or not TWO_TOWER_MODEL_PATH.exists():
        pytest.skip("Model artifacts not available in CI environment")

    with TestClient(app) as client:
        response = client.post(
            "/recommend",
            json={
                "user_id": 1,
                "k": 5,
                "model_name": "two_tower",
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == 1
    assert data["model_name"] == "two_tower"
    assert len(data["recommendations"]) == 5


def test_recommend_endpoint_rejects_unknown_model():
    if not MATRIX_MODEL_PATH.exists() or not TWO_TOWER_MODEL_PATH.exists():
        pytest.skip("Model artifacts not available in CI environment")

    with TestClient(app) as client:
        response = client.post(
            "/recommend",
            json={
                "user_id": 1,
                "k": 5,
                "model_name": "unknown_model",
            },
        )

    assert response.status_code == 400


def test_model_info_endpoint_returns_metadata():
    if not MATRIX_MODEL_PATH.exists() or not TWO_TOWER_MODEL_PATH.exists():
        pytest.skip("Model artifacts not available in CI environment")

    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["default_model_name"] == "matrix_factorization"
    assert "matrix_factorization" in data["available_models"]
    assert "two_tower" in data["available_models"]


def test_metrics_endpoint_returns_response():
    if not MATRIX_MODEL_PATH.exists() or not TWO_TOWER_MODEL_PATH.exists():
        pytest.skip("Model artifacts not available in CI environment")

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)