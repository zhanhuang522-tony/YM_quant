from __future__ import annotations

import json
from datetime import datetime, timezone
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .types import OHLCV_COLUMNS

_NUMERIC_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "trades",
    "taker_base_vol",
    "taker_quote_vol",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["open_time"] = pd.to_datetime(data["open_time"], unit="ms", utc=True, errors="coerce")
    data["close_time"] = pd.to_datetime(data["close_time"], unit="ms", utc=True, errors="coerce")

    for col in _NUMERIC_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["open_time", "close_time", "open", "high", "low", "close"])
    data = data.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    data = data[OHLCV_COLUMNS].reset_index(drop=True)
    return data


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)
    logger.info("Saved %s rows to %s", len(df), path)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    return pd.read_parquet(path)


def build_quality_report(df: pd.DataFrame, start_utc: str, end_utc: str) -> dict:
    ts = pd.to_datetime(df["open_time"], utc=True)
    expected_index = pd.date_range(start=start_utc, end=end_utc, freq="1min", tz="UTC")
    observed_index = pd.DatetimeIndex(ts)

    missing_index = expected_index.difference(observed_index)
    duplicate_count = int(df.duplicated(subset=["open_time"]).sum())
    continuity = float(1.0 - len(missing_index) / max(len(expected_index), 1))

    report = {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "expected_rows": int(len(expected_index)),
        "observed_rows": int(len(observed_index)),
        "missing_rows": int(len(missing_index)),
        "duplicate_rows": duplicate_count,
        "continuity_ratio": continuity,
        "missing_ratio": 1.0 - continuity,
        "missing_samples": [str(x) for x in missing_index[:20]],
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    return report


def save_quality_report(report: dict, path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Saved data quality report to %s", path)


def build_continuous_minute_frame(df: pd.DataFrame, start_utc: str, end_utc: str) -> pd.DataFrame:
    data = df.copy().sort_values("open_time")
    data["open_time"] = pd.to_datetime(data["open_time"], utc=True)
    data = data.set_index("open_time")

    full_index = pd.date_range(start=start_utc, end=end_utc, freq="1min", tz="UTC")
    aligned = data.reindex(full_index)
    aligned.index.name = "open_time"

    aligned["is_missing_original"] = aligned["close"].isna().astype(int)

    aligned["close"] = aligned["close"].ffill()
    aligned["open"] = aligned["open"].fillna(aligned["close"])
    aligned["high"] = aligned["high"].fillna(aligned["close"])
    aligned["low"] = aligned["low"].fillna(aligned["close"])

    for col in ["volume", "quote_asset_volume", "trades", "taker_base_vol", "taker_quote_vol"]:
        aligned[col] = aligned[col].fillna(0.0)

    aligned["close_time"] = aligned.index + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)
    aligned = aligned.reset_index()

    if aligned["close"].isna().any():
        raise RuntimeError("Close contains NaN after filling; likely missing first candle from source.")

    numeric_cols = [c for c in _NUMERIC_COLS if c in aligned.columns]
    aligned[numeric_cols] = aligned[numeric_cols].astype(np.float64)
    return aligned
