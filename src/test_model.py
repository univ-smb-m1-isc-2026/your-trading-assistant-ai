import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
TEST_DATASET_PATH = os.path.join(ARTIFACTS_DIR, "test_dataset.csv")

TEST_LOG_PATH = os.path.join(ARTIFACTS_DIR, "test.log")

MARGINS = [50, 30, 20, 10, 5, 2, 1, 0.5, 0.25, 0.1]


class TeeWriter:
    """Écrit simultanément sur la console et dans un fichier."""

    def __init__(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self._file = open(file_path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text: str) -> None:
        self._stdout.write(text)
        self._file.write(text)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def main() -> None:
    for path, label in [
        (MODEL_PATH, "model.joblib"),
        (FEATURES_PATH, "features.json"),
        (TEST_DATASET_PATH, "test_dataset.csv"),
    ]:
        if not os.path.exists(path):
            print(f"{label} introuvable: {path}")
            print("Lance d'abord: python src/train.py")
            sys.exit(1)

    tee = TeeWriter(TEST_LOG_PATH)
    sys.stdout = tee

    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATASET_PATH, index_col=0, parse_dates=True)

    x_test = pd.DataFrame(test_df[features].values, columns=features)
    y_test = test_df["target"].to_numpy()
    preds = model.predict(x_test)

    n = len(y_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    same_sign = np.sign(preds) == np.sign(y_test)

    # Écart absolu entre prédit et réel (en points de %)
    abs_error = np.abs(preds - y_test)

    print("=" * 62)
    print("  TEST DU MODÈLE — SPLIT TEMPOREL")
    print("=" * 62)
    print(f"  Lignes de test : {n}")
    print(f"  MAE  : ±{mae:.2f}%    RMSE : {rmse:.2f}%    R² : {r2:.4f}")
    print()

    # --- Tableau de précision ---
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │              TABLEAU DE PRÉCISION                       │")
    print("  ├────────────────────────────┬──────────┬─────────────────┤")
    print("  │ Critère                    │ Réussite │ Taux            │")
    print("  ├────────────────────────────┼──────────┼─────────────────┤")

    # Direction
    dir_count = int(same_sign.sum())
    dir_pct = dir_count / n * 100
    print(f"  │ Direction correcte         │ {dir_count:>5}/{n:<4}│ {dir_pct:>6.1f}%          │")
    print("  ├────────────────────────────┼──────────┼─────────────────┤")

    # Marges : écart absolu en points de % (même échelle que les variations)
    print(f"  │ Valeur à ± X pp            │ sur {n:<5}│ (écart en pp)   │")

    for margin in MARGINS:
        mask = abs_error <= margin
        count = int(mask.sum())
        pct = count / n * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        label = f"{margin:>5g}"
        print(f"  │   ± {label} pp                │ {count:>5}/{n:<4}│ {pct:>6.1f}% {bar} │")

    print("  └────────────────────────────┴──────────┴─────────────────┘")
    print()

    # --- Exemples : pires prédictions (écart > 20pp) ---
    worst_mask = abs_error > 20
    worst_indices = np.where(worst_mask)[0]
    print(f"  Pires prédictions (écart > 20 pp) : {len(worst_indices)} lignes")
    print(f"  {'Date':<12} {'Ticker':<7} {'Prédit':>8} {'Réel':>8} {'Écart':>8}")
    print(f"  {'─'*12} {'─'*7} {'─'*8} {'─'*8} {'─'*8}")
    for i in worst_indices[:15]:
        date = test_df.index[i]
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        ticker = test_df.iloc[i]["ticker"]
        print(f"  {date_str:<12} {ticker:<7} {preds[i]:+7.2f}% {y_test[i]:+7.2f}% {abs_error[i]:>7.2f}pp")

    # --- Exemples : meilleures prédictions (écart < 0.1pp) ---
    best_mask = abs_error < 0.1
    best_indices = np.where(best_mask)[0]
    print(f"\n  Meilleures prédictions (écart < 0.1 pp) : {len(best_indices)} lignes")
    print(f"  {'Date':<12} {'Ticker':<7} {'Prédit':>8} {'Réel':>8} {'Écart':>8}")
    print(f"  {'─'*12} {'─'*7} {'─'*8} {'─'*8} {'─'*8}")
    for i in best_indices[:15]:
        date = test_df.index[i]
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        ticker = test_df.iloc[i]["ticker"]
        print(f"  {date_str:<12} {ticker:<7} {preds[i]:+7.2f}% {y_test[i]:+7.2f}% {abs_error[i]:>7.2f}pp")

    # --- Vérification metrics.json ---
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)

        print()
        all_ok = True
        for key, current in [("mae_pct", mae), ("rmse_pct", rmse), ("direction_accuracy_pct", dir_pct)]:
            expected = saved.get(key)
            if expected is None:
                continue
            delta = abs(current - float(expected))
            ok = delta < 0.01
            if not ok:
                all_ok = False
        print(f"  metrics.json : {'PASS' if all_ok else 'FAIL'}")

    print(f"\n  Log sauvegardé : {TEST_LOG_PATH}")

    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()
