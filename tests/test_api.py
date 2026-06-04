from fastapi.testclient import TestClient

from app.main import app


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint_returns_recommendations():
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