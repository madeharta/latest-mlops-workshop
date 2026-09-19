"""
Sesi 5 -- Serving model dengan FastAPI.

Strategi pemuatan model (lihat README untuk penjelasan lengkap):
  1. Jika api/models/model.pkl ada (hasil mlflow/export_model.py), pakai itu.
     Ini jalur yang dipakai saat kontainerisasi (Sesi 6) -- image mandiri,
     tidak bergantung pada tracking store developer.
  2. Jika tidak ada, muat langsung dari MLflow Model Registry -- jalur ini
     yang dipakai saat development lokal di Sesi 5, sebelum kontainerisasi.

Jalankan (dari root proyek):
    uvicorn api.app:app --reload
    # buka http://127.0.0.1:8000/docs
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Insurance Approval API",
    description="Model serving endpoint untuk prediksi persetujuan polis asuransi",
    version="1.0.0",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    MODEL_SOURCE = "local file (api/models/model.pkl)"
else:
    import mlflow.sklearn
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model("models:/insurance-approval-model@production")
    MODEL_SOURCE = "MLflow Model Registry"


class ApplicantData(BaseModel):
    age: float = Field(..., example=34, description="Usia pemohon (tahun)")
    income: float = Field(..., example=78000, description="Pendapatan tahunan (Rp)")
    bmi: float = Field(..., example=24.5, description="Body Mass Index")
    tenure_months: float = Field(..., example=18, description="Lama menjadi nasabah (bulan)")
    num_claims: float = Field(..., example=1, description="Jumlah klaim sebelumnya")
    credit_score: float = Field(..., example=720, description="Skor kredit")
    policy_value: float = Field(..., example=150000, description="Nilai polis yang diajukan")
    risk_score: float = Field(..., example=42.5, description="Skor risiko internal")


class PredictionResponse(BaseModel):
    approved: bool
    probability: float
    model_version: str


@app.get("/", tags=["Health"])
def health_check():
    """Health-check endpoint untuk memastikan service berjalan."""
    return {"status": "ok", "service": "insurance-approval-api", "model_source": MODEL_SOURCE}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(data: ApplicantData):
    """Memprediksi apakah pengajuan polis disetujui berdasarkan data pemohon."""
    df = pd.DataFrame([data.dict()])
    proba = model.predict_proba(df)[0][1]
    return PredictionResponse(
        approved=bool(proba >= 0.5),
        probability=round(float(proba), 4),
        model_version="RandomForest-v1",
    )
