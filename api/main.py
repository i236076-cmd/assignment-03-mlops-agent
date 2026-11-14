"""Churn prediction service. Settings come from environment variables
(set through a ConfigMap when running on Kubernetes)."""
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = os.getenv("MODEL_DIR", "model")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

FEATURES = ["tenure", "monthly_charges", "support_calls", "usage_hours", "contract_type"]

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("churn-api")

state = {"model": None, "info": {}, "started": time.time()}


@asynccontextmanager
async def lifespan(app):
    # if the model cannot be loaded the app keeps running but reports not ready
    try:
        state["model"] = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
        with open(os.path.join(MODEL_DIR, "model_info.json")) as f:
            state["info"] = json.load(f)
        log.info("model loaded from %s", MODEL_DIR)
    except Exception as exc:
        log.error("could not load model from %s: %s", MODEL_DIR, exc)
    yield


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


class Customer(BaseModel):
    tenure: int = Field(ge=0, le=100)
    monthly_charges: float = Field(ge=0)
    support_calls: int = Field(ge=0)
    usage_hours: float = Field(ge=0)
    contract_type: int = Field(ge=0, le=2)


@app.get("/health")
def health():
    # liveness: the process is up
    return {"status": "ok"}


@app.get("/ready")
def ready():
    # readiness: only true when the model is loaded
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/version")
def version():
    return {
        "app_version": APP_VERSION,
        "model": state["info"],
        "uptime_seconds": round(time.time() - state["started"]),
    }


@app.post("/predict")
def predict(customer: Customer):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    row = pd.DataFrame([customer.model_dump()])[FEATURES]
    prob = float(state["model"].predict_proba(row)[0][1])
    log.info("prediction made probability=%.3f", prob)
    return {"churn": prob >= 0.5, "churn_probability": round(prob, 4)}
