import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features.json")

app = FastAPI()

# Chargé une seule fois au démarrage
model = None
features = None


def load_model() -> None:
    global model, features
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model.joblib introuvable: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise RuntimeError(f"features.json introuvable: {FEATURES_PATH}")
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)


@app.on_event("startup")
def startup() -> None:
    load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict) -> dict:
    missing = [f for f in features if f not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Features manquantes: {missing}")

    values = pd.DataFrame([[payload[f] for f in features]], columns=features)
    variation_pct = float(model.predict(values)[0])

    return {
        "predicted_variation_pct": round(variation_pct, 4),
        "direction": "UP" if variation_pct > 0 else "DOWN",
    }
