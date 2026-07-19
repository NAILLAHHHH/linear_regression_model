# FastAPI for life expectancy predictions

import io
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from summative.linear_regression.training import FEATURES, train_and_save

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "summative/linear_regression/artifacts/best_model.joblib"

app = FastAPI(title="Life Expectancy API")

# CORS: only allow my local frontend origins (not *)
# methods/headers limited to what the app needs
# credentials false since we dont use cookies
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5000",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class PredictIn(BaseModel):
    status: int = Field(..., ge=0, le=1)  # 0 developing, 1 developed
    adult_mortality: float = Field(..., ge=0, le=800)
    alcohol: float = Field(..., ge=0, le=20)
    bmi: float = Field(..., ge=1, le=90)
    polio: float = Field(..., ge=0, le=100)
    hiv_aids: float = Field(..., ge=0, le=60)
    gdp: float = Field(..., ge=0, le=150000)
    income_composition: float = Field(..., ge=0, le=1)
    schooling: float = Field(..., ge=0, le=25)


class PredictOut(BaseModel):
    predicted_life_expectancy: float
    model_name: str


@app.get("/")
def home():
    return {"msg": "life expectancy api", "docs": "/docs"}


@app.get("/health")
def health():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="model not found, run the notebook first")
    art = joblib.load(MODEL_PATH)
    return {"status": "ok", "model_name": art["model_name"]}


@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="model not found")

    row = pd.DataFrame(
        [
            {
                "status": data.status,
                "adult_mortality": data.adult_mortality,
                "alcohol": data.alcohol,
                "bmi": data.bmi,
                "polio": data.polio,
                "hiv_aids": data.hiv_aids,
                "gdp": data.gdp,
                "income_composition": data.income_composition,
                "schooling": data.schooling,
            }
        ],
        columns=FEATURES,
    )
    art = joblib.load(MODEL_PATH)
    pred = float(art["model"].predict(row)[0])
    return PredictOut(
        predicted_life_expectancy=round(pred, 2),
        model_name=art["model_name"],
    )


@app.post("/retrain")
def retrain(file: UploadFile = File(...)):
    # upload new csv -> retrain and replace best model
    content = file.file.read()
    try:
        pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=422, detail="could not read csv")

    try:
        _, metrics = train_and_save(io.BytesIO(content), MODEL_PATH)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="retraining failed")

    best = min(metrics, key=lambda n: metrics[n]["test_rmse"])
    return {
        "message": "model updated",
        "selected_model": best,
        "test_rmse": round(metrics[best]["test_rmse"], 3),
    }
