"""FastAPI сервис для прогноза продаж по категории на 7 дней (Ridge)."""
import logging
import yaml
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path

from src.features import build_features

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Sales Forecast API by Category", version="2.1")

ARTIFACTS_DIR = "artifacts"
CONFIG_PATH = f"{ARTIFACTS_DIR}/service_config.yaml"

# Глобальные переменные
service_cfg = {}
ridge_models = {}
feature_cols = None
lags = [1, 7, 14]
rolling_windows = [7]

@app.on_event("startup")
def load_all_models():
    global service_cfg, ridge_models, feature_cols, lags, rolling_windows
    try:
        with open(CONFIG_PATH) as f:
            service_cfg = yaml.safe_load(f)
        categories = service_cfg["categories"]
        feature_cols = service_cfg["feature_cols"]
        lags = service_cfg.get("lags", [1, 7, 14])
        rolling_windows = service_cfg.get("rolling_windows", [7])
        logger.info(f"Loaded service config: categories={categories}, features={feature_cols}")
    except Exception as e:
        logger.error(f"Failed to load service config: {e}")
        raise RuntimeError("Service configuration missing or corrupt")

    for cat in categories:
        model_path = f"{ARTIFACTS_DIR}/best_ridge_{cat}.pkl"
        try:
            ridge_models[cat] = joblib.load(model_path)
            logger.info(f"Loaded Ridge model for '{cat}'")
        except Exception as e:
            logger.error(f"Could not load Ridge model for '{cat}': {e}")
            # Если модели нет, категория будет недоступна

    if not ridge_models:
        raise RuntimeError("No Ridge models loaded. Service cannot operate.")

class PredictRequest(BaseModel):
    category: str
    history: List[float]

class PredictResponse(BaseModel):
    forecast: List[float]

@app.get("/health")
def health():
    return {"status": "ok"}

def ridge_iterative_forecast(category: str, history: List[float], steps: int = 7) -> List[float]:
    """Итеративный прогноз Ridge с пересчётом признаков."""
    model = ridge_models[category]
    # Создаём временный ряд с датами (даты не важны для признаков)
    base_date = pd.Timestamp("2023-01-01")
    dates = [base_date + pd.Timedelta(days=i) for i in range(len(history))]
    sales_series = pd.Series(history, name="sales")
    df = pd.DataFrame({"date": dates, "sales": sales_series})

    forecast = []
    for _ in range(steps):
        # Пересчитываем признаки
        feat_df = build_features(df, "sales", lags, rolling_windows)
        # Берём последнюю строку
        last_features = feat_df.iloc[-1:][feature_cols]
        pred = model.predict(last_features)[0]
        forecast.append(float(pred))
        # Добавляем прогноз к истории
        new_date = df["date"].iloc[-1] + pd.Timedelta(days=1)
        new_row = pd.DataFrame({"date": [new_date], "sales": [pred]})
        df = pd.concat([df, new_row], ignore_index=True)
    return forecast

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    cat = request.category
    if cat not in ridge_models:
        raise HTTPException(400, f"Unknown category '{cat}'. Available: {list(ridge_models.keys())}")
    if len(request.history) < 30:  # нужно хотя бы 30 точек для надёжных лагов
        raise HTTPException(400, "History must have at least 30 values for reliable features")
    forecast = ridge_iterative_forecast(cat, request.history, steps=7)
    logger.info(f"Category '{cat}' forecast: {forecast}")
    return PredictResponse(forecast=forecast)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
