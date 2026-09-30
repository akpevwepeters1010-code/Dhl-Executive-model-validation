"""FastAPI service for shipment SLA risk prediction."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/xgboost_pipeline.joblib")
app = FastAPI(title="DHL Logistics SLA Risk API", version="1.0.0")

class Shipment(BaseModel):
    hub_location: str
    service_level: str
    transit_duration: float = Field(gt=0)
    weather_severity: float = Field(ge=0)
    traffic_density: float = Field(ge=0)
    route_distance: float = Field(gt=0)
    package_weight: float = Field(gt=0)

def risk(probability: float):
    if probability >= 0.75:
        return "CRITICAL", "Escalate to priority courier and consider dynamic rerouting."
    if probability >= 0.40:
        return "ELEVATED", "Monitor closely and prioritize loading."
    return "LOW", "Continue standard routing."

@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}

@app.post("/predict")
def predict(shipment: Shipment):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run src/train.py first."
        )

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([shipment.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    category, action = risk(probability)

    return {
        "delay_probability": round(probability, 4),
        "prediction": "Delayed" if probability >= 0.40 else "On-Time",
        "risk_category": category,
        "recommended_action": action,
    }
