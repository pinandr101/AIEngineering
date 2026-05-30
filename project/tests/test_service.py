import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.service import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid_category():
    history = [200.0] * 30
    response = client.post("/predict", json={"category": "Groceries", "history": history})
    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert len(data["forecast"]) == 7

def test_predict_invalid_category():
    response = client.post("/predict", json={"category": "Unknown", "history": [100.0]*30})
    assert response.status_code == 400

def test_predict_short_history():
    response = client.post("/predict", json={"category": "Groceries", "history": [100.0]*10})
    assert response.status_code == 400
