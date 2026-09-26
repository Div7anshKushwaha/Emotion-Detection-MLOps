from flask_app.app import app


def test_home():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200


def test_health():
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"
    assert data["model"] == "EmotionDetectionModel"


def test_predict():
    client = app.test_client()

    response = client.post(
        "/predict",
        json={"text": "I am very happy today"},
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "emotion" in data
    assert data["emotion"] in ["Happiness", "Sadness"]


def test_predict_without_text():
    client = app.test_client()

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Text is required"