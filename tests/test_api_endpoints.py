import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_get_crops_api():
    response = client.get("/api/crops")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "crops" in data
    assert isinstance(data["crops"], list)
    assert len(data["crops"]) > 0

def test_get_similarity_api():
    response = client.get("/api/similarity", params={"lat": 28.6139, "lon": 77.2090, "crop": "wheat"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "crop_comparisons" in data
    assert "schedule" in data

def test_run_pipeline_api():
    response = client.post("/api/run-pipeline", json={"crop": "rice", "lat": 28.6139, "lon": 77.2090})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "logs" in data
