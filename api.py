"""
Heart Attack Risk Prediction API
FastAPI server for real-time inference.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import Optional

# ── Setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

# Load model, scaler, and metadata
model = load_model(os.path.join(MODEL_DIR, "heart_attack_model.keras"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Attack Risk Predictor",
    description="Binary classification API for heart attack risk prediction using a neural network.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────
class PatientData(BaseModel):
    """Patient medical features for prediction."""
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
                "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
                "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
            }
        }


class PredictionResponse(BaseModel):
    risk: str
    probability: float
    confidence: float
    label: int


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Heart Attack Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict heart attack risk",
            "/health": "GET - Health check",
            "/features": "GET - List expected features",
            "/docs": "GET - Interactive API docs",
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.get("/features")
def features():
    return {"features": FEATURES, "count": len(FEATURES)}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    try:
        # Build feature array in correct order
        input_data = np.array([[
            patient.age, patient.sex, patient.cp, patient.trestbps,
            patient.chol, patient.fbs, patient.restecg, patient.thalach,
            patient.exang, patient.oldpeak, patient.slope, patient.ca,
            patient.thal
        ]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        probability = float(model.predict(input_scaled, verbose=0)[0][0])
        label = 1 if probability > 0.5 else 0
        confidence = probability if label == 1 else 1 - probability

        return PredictionResponse(
            risk="High Risk" if label == 1 else "Low Risk",
            probability=round(probability, 4),
            confidence=round(confidence, 4),
            label=label
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting Heart Attack Risk Prediction API...")
    print(f"   Model features: {FEATURES}")
    print(f"   Docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
