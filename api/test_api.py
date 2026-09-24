"""
Experiment 6 — local test evidence script.

Run the API first (either directly or via Docker — see README.md), then run:
    python test_api.py

This hits every endpoint with sample JSON payloads and prints the responses,
so the printed output of this script IS your "local test evidence" deliverable —
capture it (screenshot or redirect to a file) for the report.
"""

import json
import requests

BASE_URL = "http://localhost:8000"


def show(label, response):
    print(f"\n--- {label} ---")
    print("Status code:", response.status_code)
    print("Response body:", json.dumps(response.json(), indent=2))


def main():
    # 1. Health check
    show("GET /health", requests.get(f"{BASE_URL}/health"))

    # 2. Edge track — valid sample prediction
    edge_payload = {
        "Temperature": 78.5,
        "Pressure": 101.2,
        "Vibration": 2.3,
        "Network_Latency": 15.4,
        "Edge_Processing_Time": 8.1,
        "Fuzzy_PID_Output": 0.62,
        "Maintenance_Status": "Normal",
        "Predicted_Failure": 0,
    }
    show("POST /predict/edge (valid)", requests.post(f"{BASE_URL}/predict/edge", json=edge_payload))

    # 3. Edge track — a second sample representing worse machine conditions,
    #    to sanity-check the prediction actually moves in a sensible direction
    edge_payload_bad = {
        "Temperature": 95.0,
        "Pressure": 130.0,
        "Vibration": 9.8,
        "Network_Latency": 80.0,
        "Edge_Processing_Time": 40.0,
        "Fuzzy_PID_Output": 0.15,
        "Maintenance_Status": "Failure",
        "Predicted_Failure": 1,
    }
    show("POST /predict/edge (degraded conditions)", requests.post(f"{BASE_URL}/predict/edge", json=edge_payload_bad))

    # 4. Infrastructure track — valid sample prediction (Node telemetry)
    infra_payload_node = {
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
    show("POST /predict/infra (Node telemetry)", requests.post(f"{BASE_URL}/predict/infra", json=infra_payload_node))

    # 5. Infrastructure track — Pod telemetry sample (different structural NaN pattern)
    infra_payload_pod = {
        "CPU (%)": 30.0,
        "Energy (watts)": 55.0,
        "MEM (%)": 0.0,
        "fs (%)": 0.0,
        "rx (B/sec)": 0.0,
        "tx (B/sec)": 0.0,
        "MEM (B)": 52428800,
        "Telemetry_Type": "Pod",
        "Pod_Status": "Pods On",
        "Scenario": "Pod_Pods_On",
    }
    show("POST /predict/infra (Pod telemetry)", requests.post(f"{BASE_URL}/predict/infra", json=infra_payload_pod))

    # 6. Invalid request — bad enum value, should return a 422 validation error.
    #    Demonstrates the API's input validation, not just the happy path.
    bad_payload = dict(edge_payload)
    bad_payload["Maintenance_Status"] = "NotARealStatus"
    show("POST /predict/edge (invalid Maintenance_Status -> expect 422)",
         requests.post(f"{BASE_URL}/predict/edge", json=bad_payload))


if __name__ == "__main__":
    main()
