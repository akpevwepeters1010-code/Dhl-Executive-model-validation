from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_validation_rejects_invalid_weight():
    response = client.post("/predict", json={
        "hub_location": "HUB_US",
        "service_level": "Express_Worldwide",
        "transit_duration": 18,
        "weather_severity": 2,
        "traffic_density": 4,
        "route_distance": 500,
        "package_weight": 0
    })
    assert response.status_code == 422
