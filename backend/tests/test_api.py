from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_models_endpoint(monkeypatch):
    def mock_list_models():
        return ["qwen3.5-4b", "text-embedding-nomic-embed-text-v1.5"]

    monkeypatch.setattr("app.main.list_models", mock_list_models)

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "qwen3.5-4b" in response.json()["models"]


def test_chat_endpoint(monkeypatch):
    def mock_generate_response(message):
        return "Hello! I'm LocalMind."

    monkeypatch.setattr(
        "app.main.generate_response",
        mock_generate_response,
    )

    response = client.post(
        "/chat",
        json={"message": "Hello LocalMind!"},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Hello! I'm LocalMind."


def test_empty_chat_message():
    response = client.post(
        "/chat",
        json={"message": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Message cannot be empty."


def test_chat_model_failure(monkeypatch):
    def mock_generate_response(message):
        raise RuntimeError("Cannot connect to LM Studio.")

    monkeypatch.setattr(
        "app.main.generate_response",
        mock_generate_response,
    )

    response = client.post(
        "/chat",
        json={"message": "Hello"},
    )

    assert response.status_code == 503