from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "target_log_ret_1h"

FEATURE_COLUMNS = [
    "log_ret_1h",
    "log_ret_3h",
    "log_ret_6h",
    "log_ret_12h",
    "log_ret_24h",
    "vol_24h",
    "rv_24h",
    "volume_chg_1h",
    "volume_over_ma_24h",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width_20",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("open_time").reset_index(drop=True)
    data["open_time"] = pd.to_datetime(data["open_time"], utc=True)

    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "trades"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    close = data["close"]
    data["log_ret_1h"] = np.log(close / close.shift(1))
    data["log_ret_3h"] = np.log(close / close.shift(3))
    data["log_ret_6h"] = np.log(close / close.shift(6))
    data["log_ret_12h"] = np.log(close / close.shift(12))
    data["log_ret_24h"] = np.log(close / close.shift(24))

    data["vol_24h"] = data["log_ret_1h"].rolling(24, min_periods=24).std()
    data["rv_24h"] = np.sqrt((data["log_ret_1h"] ** 2).rolling(24, min_periods=24).sum())

    data["volume_chg_1h"] = data["volume"].pct_change(1)
    vol_ma_24 = data["volume"].rolling(24, min_periods=24).mean()
    data["volume_over_ma_24h"] = data["volume"] / vol_ma_24 - 1.0

    data["rsi_14"] = _rsi(close, window=14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    ma_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std()
    upper = ma_20 + 2 * std_20
    lower = ma_20 - 2 * std_20
    data["bb_width_20"] = (upper - lower) / ma_20.replace(0.0, np.nan)

    data[TARGET_COL] = np.log(close.shift(-1) / close)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COL]).reset_index(drop=True)
    return data


def build_panel_feature_dataset(panel_df: pd.DataFrame) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    for symbol, g in panel_df.groupby("symbol", sort=False):
        feat = build_feature_dataset(g)
        feat["symbol"] = symbol
        out.append(feat)
    if not out:
        raise RuntimeError("No panel features generated")
    return pd.concat(out, ignore_index=True).sort_values(["open_time", "symbol"]).reset_index(drop=True)
