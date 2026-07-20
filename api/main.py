"""
Hospital Readmission Risk — FastAPI Prediction Endpoint
POST /predict → risk score, tier, top SHAP drivers
GET  /health  → model metadata
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import uvicorn

app = FastAPI(
    title="Hospital Readmission Risk API",
    description="Predicts 30-day readmission risk with SHAP explainability.",
    version="1.0.0",
)

# ── Load artifacts at startup ─────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

model = None
engineer = None
explainer = None
feature_names = None
threshold = 0.35
test_metrics = {}


@app.on_event("startup")
def load_artifacts():
    global model, engineer, explainer, feature_names, threshold, test_metrics
    try:
        model        = joblib.load(os.path.join(MODEL_DIR, 'readmission_model.pkl'))
        engineer     = joblib.load(os.path.join(MODEL_DIR, 'feature_engineer.pkl'))
        explainer    = joblib.load(os.path.join(MODEL_DIR, 'shap_explainer.pkl'))
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
        threshold    = joblib.load(os.path.join(MODEL_DIR, 'threshold.pkl'))
        test_metrics = joblib.load(os.path.join(MODEL_DIR, 'test_metrics.pkl'))
        print(f"✓ Model loaded | AUC-ROC: {test_metrics.get('auc_roc', 'N/A')}")
    except Exception as e:
        print(f"⚠ Model load error: {e}")


# ── Request / Response schemas ────────────────────────────────────────────────
class PatientInput(BaseModel):
    age: str = Field("[60-70)", description="Age bracket e.g. '[60-70)'")
    gender: str = Field("Female", description="Male or Female")
    race: str = Field("Caucasian")
    time_in_hospital: int = Field(5, ge=1, le=14, description="Days in hospital")
    num_lab_procedures: int = Field(43, ge=0, le=132)
    num_procedures: int = Field(1, ge=0, le=6)
    num_medications: int = Field(16, ge=1, le=81)
    number_outpatient: int = Field(0, ge=0)
    number_emergency: int = Field(0, ge=0)
    number_inpatient: int = Field(0, ge=0)
    number_diagnoses: int = Field(7, ge=1, le=16)
    diag_1: str = Field("250.00", description="Primary ICD-9 diagnosis")
    diag_2: str = Field("?")
    diag_3: str = Field("?")
    admission_type_id: int = Field(1, description="1=Emergency, 2=Urgent, 3=Elective")
    discharge_disposition_id: int = Field(1)
    admission_source_id: int = Field(4)
    A1Cresult: str = Field("None", description=">8, >7, Norm, or None")
    change: str = Field("No", description="Medication change: Ch or No")
    diabetesMed: str = Field("Yes")
    insulin: str = Field("No", description="No/Steady/Up/Down")
    metformin: str = Field("No")
    repaglinide: str = Field("No")
    nateglinide: str = Field("No")
    glimepiride: str = Field("No")
    glipizide: str = Field("No")
    glyburide: str = Field("No")
    weight: str = Field("?")
    payer_code: str = Field("MC")
    medical_specialty: str = Field("InternalMedicine")


class RiskDriver(BaseModel):
    feature: str
    impact: float
    direction: str   # "increases" or "decreases"


class PredictionResponse(BaseModel):
    readmission_risk_score: float
    readmission_risk_pct: str
    risk_tier: str
    risk_tier_description: str
    top_risk_drivers: List[RiskDriver]
    avoided_cost_if_intervened: str
    recommendation: str


# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build DataFrame matching training schema
    row = patient.dict()
    df = pd.DataFrame([row])

    # Feature engineering
    try:
        X = engineer.transform(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering error: {e}")

    # Predict
    prob = float(model.predict_proba(X.values)[0][1])

    # SHAP drivers
    drivers = []
    if explainer is not None:
        try:
            sv = explainer.shap_values(X.values)[0]
            top_idx = np.argsort(np.abs(sv))[::-1][:5]
            for i in top_idx:
                drivers.append(RiskDriver(
                    feature=feature_names[i].replace('_', ' '),
                    impact=round(float(sv[i]), 4),
                    direction="increases" if sv[i] > 0 else "decreases",
                ))
        except Exception:
            pass

    # Risk tier
    if prob >= 0.65:
        tier = "HIGH"
        tier_desc = "Immediate care coordination recommended"
        rec = "Schedule 48h post-discharge follow-up call. Assign care manager. Medication reconciliation."
    elif prob >= 0.40:
        tier = "MODERATE"
        tier_desc = "Enhanced monitoring recommended"
        rec = "Schedule 7-day post-discharge clinic visit. Medication review. Patient education."
    else:
        tier = "LOW"
        tier_desc = "Standard discharge protocol"
        rec = "Standard discharge instructions. 30-day follow-up appointment."

    avoided = f"${int(prob * 15200 * 0.40):,}" if prob > 0.30 else "$0"

    return PredictionResponse(
        readmission_risk_score=round(prob, 4),
        readmission_risk_pct=f"{prob*100:.1f}%",
        risk_tier=tier,
        risk_tier_description=tier_desc,
        top_risk_drivers=drivers,
        avoided_cost_if_intervened=avoided,
        recommendation=rec,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "metrics": test_metrics,
        "threshold": float(threshold),
    }


@app.get("/")
def root():
    return {"message": "Hospital Readmission Risk API. POST /predict with patient data. GET /docs for Swagger UI."}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
