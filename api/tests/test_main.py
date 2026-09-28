"""
Experiment 7 — CI unit tests for the Experiment 6 FastAPI app.

Deliberately does NOT require edge_efficiency_best_model.pkl /
infra_efficiency_best_model.pkl to be present. Those files aren't (and
shouldn't be) committed to git, so this suite verifies the API's *graceful
degradation* path (a clean 503, not a crash) when a model isn't loaded, in
addition to its input validation. If real models are present (e.g. in a
fuller pipeline that pulls them via DVC before testing), the same tests
accept a successful 200 prediction too — see the "if model files exist"
branches below.

Run from the api/ directory:
    pytest tests -v
"""

import pytest
from fastapi.testclient import TestClient

from main import app, EDGE_MODEL_PATH, INFRA_MODEL_PATH

EDGE_VALID_PAYLOAD = {
    "Temperature": 78.5,
    "Pressure": 101.2,
    "Vibration": 2.3,
    "Network_Latency": 15.4,
    "Edge_Processing_Time": 8.1,
    "Fuzzy_PID_Output": 0.62,
    "Maintenance_Status": "Normal",
    "Predicted_Failure": 0,
}

EDGE_INVALID_PAYLOAD = {**EDGE_VALID_PAYLOAD, "Maintenance_Status": "NotARealStatus"}

INFRA_VALID_PAYLOAD = {
    "CPU (%)": 45.0,
    "Energy (watts)": 120.5,
    "MEM (%)": 60.2,
    "fs (%)": 35.0,
    "rx (B/sec)": 1200.0,
    "tx (B/sec)": 980.0,
    "MEM (B)": 0.0,
    "Telemetry_Type": "Node",
    "Pod_Status": "Pods On",
    "Scenario": "Node_Pods_On",
}

INFRA_INVALID_PAYLOAD = {**INFRA_VALID_PAYLOAD, "Telemetry_Type": "NotAType"}


@pytest.fixture
def client():
    # Using the context-manager form ensures FastAPI's startup event (model
    # loading) actually fires before the tests run.
    with TestClient(app) as c:
        yield c


def test_root_lists_endpoints(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "endpoints" in body
    assert "/predict/edge" in body["endpoints"]
    assert "/predict/infra" in body["endpoints"]


def test_health_reports_model_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "edge_model_loaded" in body
    assert "infra_model_loaded" in body
    # These should reflect reality, whatever it is in this environment
    assert body["edge_model_loaded"] == EDGE_MODEL_PATH.exists()
    assert body["infra_model_loaded"] == INFRA_MODEL_PATH.exists()


def test_predict_edge_invalid_enum_returns_422(client):
    response = client.post("/predict/edge", json=EDGE_INVALID_PAYLOAD)
    assert response.status_code == 422


def test_predict_edge_missing_fields_returns_422(client):
    response = client.post("/predict/edge", json={})
    assert response.status_code == 422


def test_predict_edge_valid_payload(client):
    response = client.post("/predict/edge", json=EDGE_VALID_PAYLOAD)
    if EDGE_MODEL_PATH.exists():
        assert response.status_code == 200
        body = response.json()
        assert body["track"] == "edge"
        assert isinstance(body["predicted_score"], float)
    else:
        # No model file in this environment -> API must fail gracefully,
        # not crash. This is the behavior CI actually exercises by default.
        assert response.status_code == 503


def test_predict_infra_invalid_enum_returns_422(client):
    response = client.post("/predict/infra", json=INFRA_INVALID_PAYLOAD)
    assert response.status_code == 422


def test_predict_infra_valid_payload(client):
    response = client.post("/predict/infra", json=INFRA_VALID_PAYLOAD)
    if INFRA_MODEL_PATH.exists():
        assert response.status_code == 200
        body = response.json()
        assert body["track"] == "infra"
        assert isinstance(body["predicted_score"], float)
    else:
        assert response.status_code == 503
