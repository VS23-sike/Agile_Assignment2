from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == (
        "Student Performance Prediction API is running"
    )


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "evaluation_metrics" in response.json()


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={
            "study_hours": 6,
            "attendance": 80,
            "previous_score": 65,
        },
    )
    assert response.status_code == 200
    assert 0 <= response.json()["predicted_score"] <= 100


def test_prediction_rejects_invalid_input():
    response = client.post(
        "/predict",
        json={
            "study_hours": 30,
            "attendance": 80,
            "previous_score": 65,
        },
    )
    assert response.status_code == 422