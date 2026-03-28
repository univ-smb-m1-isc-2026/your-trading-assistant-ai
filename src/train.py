import json
import logging
import math
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = [
    # Returns
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    # Moving averages
    "close_vs_ma5",
    "close_vs_ma10",
    "close_vs_ma20",
    "close_vs_ma50",
    # Volatility
    "volatility_5",
    "volatility_10",
    "volatility_20",
    # Volume
    "volume_ratio_5",
    "volume_ratio_20",
    # Price action
    "high_low_range",
    "open_gap",
    # RSI
    "rsi_14",
    # MACD
    "macd_signal_diff",
    # Bollinger
    "bollinger_pos",
    # ATR
    "atr_14_pct",
    # Calendar
    "day_of_week",
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
LOG_PATH = os.path.join(ARTIFACTS_DIR, "train.log")


def setup_logging() -> logging.Logger:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    unique_dates = sorted(df.index.unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    train_df = df[df.index < split_date].copy()
    test_df = df[df.index >= split_date].copy()

    return train_df, test_df, str(split_date)


def main() -> None:
    log = setup_logging()
    log.info("=" * 50)
    log.info("TRAINING PIPELINE — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 50)

    if not os.path.exists(DATA_PATH):
        log.error("Dataset introuvable: %s", DATA_PATH)
        log.error("Lance d'abord: python src/build_dataset.py")
        return

    log.info("[1/5] Chargement du dataset...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    log.info("  → %d lignes, %d tickers", len(df), df["ticker"].nunique())
    log.info("  → Période: %s → %s", df.index.min().date(), df.index.max().date())

    log.info("[2/5] Split temporel 80/20...")
    train_df, test_df, split_date = temporal_split(df)
    log.info("  → Train: %d lignes | Test: %d lignes", len(train_df), len(test_df))
    log.info("  → Date de split: %s", split_date)

    X_train = train_df[FEATURES]
    y_train = train_df["target"]
    X_test = test_df[FEATURES]
    y_test = test_df["target"]

    log.info("  → Target train — mean=%.4f%% std=%.4f%%", y_train.mean(), y_train.std())

    log.info("[3/5] Entraînement HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42,
    )
    model.fit(X_train, y_train)
    log.info("  → Entraînement terminé.")

    log.info("[4/5] Évaluation sur le set de test...")
    preds = model.predict(X_test)
    y_actual = y_test.to_numpy()

    mae = mean_absolute_error(y_actual, preds)
    rmse = math.sqrt(mean_squared_error(y_actual, preds))
    r2 = r2_score(y_actual, preds)

    # Direction accuracy : le modèle prédit-il le bon sens (hausse/baisse) ?
    direction_ok = np.sign(preds) == np.sign(y_actual)
    direction_acc = direction_ok.mean()

    # Biais moyen : le modèle surévalue ou sous-évalue ?
    mean_error = (preds - y_actual).mean()

    log.info("")
    log.info("=== RÉSULTATS ===")
    log.info("Erreur moyenne absolue (MAE): %.2f%%  (en moyenne, le modèle se trompe de ±%.2f points de %%)", mae, mae)
    log.info("RMSE: %.2f%%", rmse)
    log.info("R²:   %.4f  (1.0 = parfait, 0.0 = pas mieux que la moyenne)", r2)
    log.info("")
    log.info("Direction accuracy: %.1f%%  (%d/%d prédictions dans le bon sens)",
             direction_acc * 100, int(direction_ok.sum()), len(y_actual))
    log.info("Biais moyen: %+.4f%%  (%s)",
             mean_error, "surévalue" if mean_error > 0 else "sous-évalue")

    # Exemples concrets
    log.info("")
    log.info("Exemples de prédictions (10 lignes au hasard du set de test):")
    log.info("  %-12s %-6s %10s %10s %6s", "Date", "Ticker", "Prédit", "Réel", "Sens")
    indices = np.linspace(0, len(y_actual) - 1, 10, dtype=int)
    for i in indices:
        date = X_test.index[i]
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        ticker = test_df.iloc[i]["ticker"]
        ok = "✓" if direction_ok[i] else "✗"
        log.info("  %-12s %-6s %+9.2f%% %+9.2f%% %4s", date_str, ticker, preds[i], y_actual[i], ok)

    log.info("[5/5] Sauvegarde des artefacts...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    joblib.dump(model, model_path)
    log.info("  → model.joblib sauvegardé (%d KB)", os.path.getsize(model_path) // 1024)

    features_path = os.path.join(ARTIFACTS_DIR, "features.json")
    with open(features_path, "w") as f:
        json.dump(FEATURES, f, indent=2)
    log.info("  → features.json sauvegardé")

    metrics = {
        "mae_pct": round(mae, 4),
        "rmse_pct": round(rmse, 4),
        "r2": round(r2, 6),
        "direction_accuracy_pct": round(direction_acc * 100, 2),
        "mean_bias_pct": round(mean_error, 4),
        "split_date": split_date,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("  → metrics.json sauvegardé")

    test_dataset_path = os.path.join(ARTIFACTS_DIR, "test_dataset.csv")
    test_export = test_df[["ticker", *FEATURES, "target"]].copy()
    test_export.index.name = "date"
    test_export.to_csv(test_dataset_path)
    log.info("  → test_dataset.csv sauvegardé (%d lignes)", len(test_export))

    log.info("")
    log.info("Terminé. Log complet: %s", LOG_PATH)


if __name__ == "__main__":
    main()
