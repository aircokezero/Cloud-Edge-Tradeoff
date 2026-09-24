"""
Experiment 6 — Containerization & API Deployment
Cloud-Edge Tradeoff: Adaptive Workload Placement

Serves the two regression models trained and saved in Experiment 4:
  - models/edge_efficiency_best_model.pkl   -> predicts Edge_Efficiency_Score
  - models/infra_efficiency_best_model.pkl  -> predicts Infrastructure_Efficiency_Score

Run locally (without Docker):
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000/docs for interactive Swagger UI, or see
test_api.py for scripted sample requests.
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_DIR = Path(__file__).parent / "models"
EDGE_MODEL_PATH = MODEL_DIR / "edge_efficiency_best_model.pkl"
INFRA_MODEL_PATH = MODEL_DIR / "infra_efficiency_best_model.pkl"

app = FastAPI(
    title="Cloud-Edge Tradeoff — Efficiency Prediction API",
    description="Serves the Experiment 4 models that predict Edge_Efficiency_Score "
                "and Infrastructure_Efficiency_Score from raw telemetry.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Load models once at startup (not per-request) so predictions stay fast.
# If a model file is missing, the app still starts — that endpoint will just
# return a 503 until the file is placed in models/ and the container restarted.
# ---------------------------------------------------------------------------
edge_model = None
infra_model = None


@app.on_event("startup")
def load_models() -> None:
    global edge_model, infra_model
    if EDGE_MODEL_PATH.exists():
        edge_model = joblib.load(EDGE_MODEL_PATH)
        print(f"Loaded edge model from {EDGE_MODEL_PATH}")
    else:
        print(f"WARNING: edge model not found at {EDGE_MODEL_PATH}")

    if INFRA_MODEL_PATH.exists():
        infra_model = joblib.load(INFRA_MODEL_PATH)
        print(f"Loaded infra model from {INFRA_MODEL_PATH}")
    else:
        print(f"WARNING: infra model not found at {INFRA_MODEL_PATH}")


# ---------------------------------------------------------------------------
# Request schemas — field names/aliases match the exact training columns
# from Experiment 4, so the incoming JSON converts directly into a DataFrame
# the saved sklearn Pipeline (ColumnTransformer + model) can consume as-is.
# ---------------------------------------------------------------------------
class EdgeFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Temperature: float = Field(..., examples=[78.5])
    Pressure: float = Field(..., examples=[101.2])
    Vibration: float = Field(..., examples=[2.3])
    Network_Latency: float = Field(..., examples=[15.4])
    Edge_Processing_Time: float = Field(..., examples=[8.1])
    Fuzzy_PID_Output: float = Field(..., examples=[0.62])
    Maintenance_Status: Literal["Normal", "Warning", "Failure"] = Field(..., examples=["Normal"])
    Predicted_Failure: int = Field(..., ge=0, le=1, examples=[0])


class InfraFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cpu_pct: float = Field(..., alias="CPU (%)", examples=[45.0])
    energy_watts: float = Field(..., alias="Energy (watts)", examples=[120.5])
    mem_pct: float = Field(0.0, alias="MEM (%)", examples=[60.2])
    fs_pct: float = Field(0.0, alias="fs (%)", examples=[35.0])
    rx_bps: float = Field(0.0, alias="rx (B/sec)", examples=[1200.0])
    tx_bps: float = Field(0.0, alias="tx (B/sec)", examples=[980.0])
    mem_b: float = Field(0.0, alias="MEM (B)", examples=[0.0])
    Telemetry_Type: Literal["Node", "Pod"] = Field(..., examples=["Node"])
    Pod_Status: Literal["Pods On", "Pods Off"] = Field(..., examples=["Pods On"])
    Scenario: Literal[
        "Node_Pods_On", "Node_Pods_Off", "Pod_Pods_On", "Pod_Pods_Off"
    ] = Field(..., examples=["Node_Pods_On"])


class PredictionResponse(BaseModel):
    track: str
    predicted_score: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Cloud-Edge Tradeoff efficiency prediction API",
        "endpoints": ["/health", "/predict/edge", "/predict/infra", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "edge_model_loaded": edge_model is not None,
        "infra_model_loaded": infra_model is not None,
    }


@app.post("/predict/edge", response_model=PredictionResponse)
def predict_edge(features: EdgeFeatures):
    if edge_model is None:
        raise HTTPException(status_code=503, detail="Edge model is not loaded on this server.")

    row = pd.DataFrame([features.model_dump(by_alias=True)])
    try:
        prediction = edge_model.predict(row)[0]
    except Exception as exc:  # surfaces a clean 400 instead of a raw 500 traceback
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return PredictionResponse(track="edge", predicted_score=float(prediction))


@app.post("/predict/infra", response_model=PredictionResponse)
def predict_infra(features: InfraFeatures):
    if infra_model is None:
        raise HTTPException(status_code=503, detail="Infrastructure model is not loaded on this server.")

    row = pd.DataFrame([features.model_dump(by_alias=True)])
    try:
        prediction = infra_model.predict(row)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return PredictionResponse(track="infra", predicted_score=float(prediction))
