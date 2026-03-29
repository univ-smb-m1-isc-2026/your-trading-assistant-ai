import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests


HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"

# Mêmes tickers que ProdDataInitializer.java
TICKERS = [
    "BTC", "ETH", "ATOM", "DYDX", "SOL", "AVAX", "BNB", "APE", "OP", "LTC",
    "ARB", "DOGE", "INJ", "SUI", "kPEPE", "CRV", "LDO", "LINK", "STX", "CFX",
    "GMX", "SNX", "XRP", "BCH", "APT", "AAVE", "COMP", "WLD", "YGG", "TRX",
    "kSHIB", "UNI", "SEI", "RUNE", "ZRO", "DOT", "BANANA", "TRB", "FTT", "ARK",
    "BIGTIME", "KAS", "BLUR", "TIA", "BSV", "ADA", "TON", "MINA", "POLYX", "GAS",
]

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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset.csv")


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def fetch_candles(ticker: str, days: int = 365 * 5) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    resp = requests.post(HYPERLIQUID_URL, json={
        "type": "candleSnapshot",
        "req": {
            "coin": ticker,
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        raise RuntimeError(f"Aucune donnée pour {ticker}")

    rows = []
    for c in raw:
        rows.append({
            "date": pd.Timestamp.utcfromtimestamp(c["t"] / 1000).normalize(),
            "Open": float(c["o"]),
            "High": float(c["h"]),
            "Low": float(c["l"]),
            "Close": float(c["c"]),
            "Volume": float(c["v"]),
        })

    df = pd.DataFrame(rows).set_index("date").sort_index()
    # dédupliquer au cas où Hyperliquid renvoie des doublons
    df = df[~df.index.duplicated(keep="last")]
    return df


def compute_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    opn = df["Open"]
    volume = df["Volume"]
    prev_close = close.shift(1)
    daily_return = close.pct_change(1)

    # Moving averages
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    # MACD (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    bb_mid = ma20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_safe = bb_width.replace(0, 1e-10)

    # ATR 14
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()

    result = pd.DataFrame({
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "ticker": ticker,
        # Returns
        "return_1d": daily_return,
        "return_2d": close.pct_change(2),
        "return_3d": close.pct_change(3),
        "return_5d": close.pct_change(5),
        "return_10d": close.pct_change(10),
        "return_20d": close.pct_change(20),
        # Moving averages
        "close_vs_ma5": close / ma5 - 1,
        "close_vs_ma10": close / ma10 - 1,
        "close_vs_ma20": close / ma20 - 1,
        "close_vs_ma50": close / ma50 - 1,
        # Volatility
        "volatility_5": daily_return.rolling(5).std(),
        "volatility_10": daily_return.rolling(10).std(),
        "volatility_20": daily_return.rolling(20).std(),
        # Volume
        "volume_ratio_5": volume / volume.rolling(5).mean(),
        "volume_ratio_20": volume / volume.rolling(20).mean(),
        # Price action
        "high_low_range": (high - low) / close,
        "open_gap": (opn - prev_close) / prev_close,
        # RSI
        "rsi_14": compute_rsi(close, 14),
        # MACD
        "macd_signal_diff": (macd_line - macd_signal) / close,
        # Bollinger position (0 = bas, 1 = haut)
        "bollinger_pos": (close - bb_lower) / bb_width_safe,
        # ATR en % du prix
        "atr_14_pct": atr_14 / close,
        # Calendar
        "day_of_week": df.index.dayofweek.astype(float),
        # Target
        "target": (close.shift(-1) - close) / close * 100,
    }, index=df.index)

    return result.dropna()


def build_dataset() -> pd.DataFrame:
    parts = []
    failed = []

    for i, ticker in enumerate(TICKERS):
        try:
            raw = fetch_candles(ticker)
            part = compute_features(raw, ticker)
            parts.append(part)
            print(f"[OK] {ticker}: {len(part)} lignes  ({i + 1}/{len(TICKERS)})")
        except Exception as e:
            failed.append((ticker, str(e)))
            print(f"[FAIL] {ticker}: {e}  ({i + 1}/{len(TICKERS)})")
        if i < len(TICKERS) - 1:
            time.sleep(1)

    if not parts:
        raise RuntimeError("Aucune donnée exploitable téléchargée.")

    dataset = pd.concat(parts).sort_index()

    print(f"\nDataset final: {len(dataset)} lignes")
    print(f"Tickers utilisés: {dataset['ticker'].nunique()} / {len(TICKERS)}")

    if failed:
        print("\nTickers en échec:")
        for ticker, err in failed:
            print(f"  - {ticker}: {err}")

    return dataset


def main() -> None:
    dataset = build_dataset()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH)
    print(f"\nDataset sauvegardé dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
