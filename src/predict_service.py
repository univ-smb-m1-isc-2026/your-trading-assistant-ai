import json
import os
import re
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features.json")
TEST_LOG_PATH = os.path.join(ARTIFACTS_DIR, "test.log")

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


def _parse_ratio(value: str) -> tuple[int, int]:
    parts = value.strip().split("/")
    if len(parts) != 2:
        return 0, 0
    return int(parts[0].strip()), int(parts[1].strip())


def parse_test_report() -> dict[str, Any]:
    if not os.path.exists(TEST_LOG_PATH):
        raise HTTPException(status_code=404, detail="test.log introuvable. Lance d'abord python src/test_model.py")

    with open(TEST_LOG_PATH, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    summary: dict[str, Any] = {}
    precision_table: list[dict[str, Any]] = []
    worst_examples: list[dict[str, Any]] = []
    best_examples: list[dict[str, Any]] = []

    section = ""
    row_pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2})\s+(\S+)\s+([+-]\d+\.\d+)%\s+([+-]\d+\.\d+)%\s+(\d+\.\d+)pp$"
    )

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Lignes de test :"):
            summary["test_rows"] = int(stripped.split(":")[1].strip())
            continue

        if "MAE  :" in stripped and "RMSE" in stripped and "R²" in stripped:
            m = re.search(r"MAE\s*:\s*±([0-9.]+)%\s+RMSE\s*:\s*([0-9.]+)%\s+R²\s*:\s*([-0-9.]+)", stripped)
            if m:
                summary["mae_pct"] = float(m.group(1))
                summary["rmse_pct"] = float(m.group(2))
                summary["r2"] = float(m.group(3))
            continue

        if "Direction correcte" in line and "│" in line:
            cols = [c.strip() for c in line.split("│") if c.strip()]
            if len(cols) >= 3:
                success, total = _parse_ratio(cols[1])
                rate_pct = float(cols[2].split("%")[0].strip())
                summary["direction_success"] = success
                summary["direction_total"] = total
                summary["direction_accuracy_pct"] = rate_pct
            continue

        if stripped.startswith("│   ±") and "pp" in line and "│" in line:
            cols = [c.strip() for c in line.split("│") if c.strip()]
            if len(cols) >= 3:
                margin_pp = cols[0].replace("±", "").replace("pp", "").strip()
                success, total = _parse_ratio(cols[1])
                rate_pct = float(cols[2].split("%")[0].strip())
                precision_table.append({
                    "margin_pp": margin_pp,
                    "success": success,
                    "total": total,
                    "rate_pct": rate_pct,
                })
            continue

        if stripped.startswith("Pires prédictions"):
            section = "worst"
            m = re.search(r":\s*(\d+)\s+lignes$", stripped)
            if m:
                summary["worst_count"] = int(m.group(1))
            continue

        if stripped.startswith("Meilleures prédictions"):
            section = "best"
            m = re.search(r":\s*(\d+)\s+lignes$", stripped)
            if m:
                summary["best_count"] = int(m.group(1))
            continue

        m = row_pattern.match(stripped)
        if m and section in {"worst", "best"}:
            item = {
                "date": m.group(1),
                "ticker": m.group(2),
                "predicted_pct": float(m.group(3)),
                "actual_pct": float(m.group(4)),
                "abs_error_pp": float(m.group(5)),
            }
            if section == "worst":
                worst_examples.append(item)
            else:
                best_examples.append(item)

    return {
        "summary": summary,
        "precision_table": precision_table,
        "worst_examples": worst_examples,
        "best_examples": best_examples,
    }


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


@app.get("/test-report")
def test_report() -> dict[str, Any]:
    return parse_test_report()
