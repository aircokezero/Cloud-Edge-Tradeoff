# Experiment 6 — Containerization & API Deployment

Serves the two regression models trained and saved in Experiment 4
(`edge_efficiency_best_model.pkl`, `infra_efficiency_best_model.pkl`) behind
a FastAPI `/predict` interface, packaged into a Docker image.

This document assumes you've installed docker on your device already. If not, follow docker's own documentation for your respective device.

## 1. Put the Experiment 4 models in place

```
api/
├── main.py
├── requirements.txt
├── Dockerfile
├── test_api.py
├── README.md
└── models/
    ├── edge_efficiency_best_model.pkl
    └── infra_efficiency_best_model.pkl
```

## 2. Run locally without Docker (fastest way to iterate)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for interactive Swagger UI, or in another
terminal:

```bash
python test_api.py
```

## 3. Build and run the Docker image

```bash
docker build -t cloud-edge-api .
docker run -d -p 8000:8000 --name cloud-edge-api cloud-edge-api
```

Check it started cleanly:

```bash
docker ps
docker logs cloud-edge-api
```

You should see `Loaded edge model from ...` and `Loaded infra model from ...`
in the logs — if either is missing, that model file wasn't found under
`models/` when the image was built (rebuild after placing the `.pkl` files).

## 4. Test the containerized API

```bash
python test_api.py
```

Or with `curl` directly:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict/edge \
  -H "Content-Type: application/json" \
  -d '{
        "Temperature": 78.5,
        "Pressure": 101.2,
        "Vibration": 2.3,
        "Network_Latency": 15.4,
        "Edge_Processing_Time": 8.1,
        "Fuzzy_PID_Output": 0.62,
        "Maintenance_Status": "Normal",
        "Predicted_Failure": 0
      }'

curl -X POST http://localhost:8000/predict/infra \
  -H "Content-Type: application/json" \
  -d '{
        "CPU (%)": 45.0,
        "Energy (watts)": 120.5,
        "MEM (%)": 60.2,
        "fs (%)": 35.0,
        "rx (B/sec)": 1200.0,
        "tx (B/sec)": 980.0,
        "MEM (B)": 0.0,
        "Telemetry_Type": "Node",
        "Pod_Status": "Pods On",
        "Scenario": "Node_Pods_On"
      }'
```

## 5. Stop and clean up

```bash
docker stop cloud-edge-api
docker rm cloud-edge-api
```

## Capturing "local test evidence" for the report

Redirect `test_api.py`'s output to a file:

```bash
python test_api.py > test_evidence.txt
```