from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

OHLCV_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "trades",
    "taker_base_vol",
    "taker_quote_vol",
    "close_time",
]


@dataclass(frozen=True)
class SplitWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


@dataclass(frozen=True)
class FitArtifacts:
    model: Any
    feature_columns: list[str]


@dataclass(frozen=True)
class NextHourPrediction:
    asof_time_utc: datetime
    target_time_utc: datetime
    predicted_close: float


@dataclass(frozen=True)
class Paths:
    raw_path: Path
    feature_path: Path
    quality_report_path: Path
    model_path: Path
    backtest_metrics_path: Path
    backtest_detail_path: Path
    holdout_metrics_path: Path
    holdout_detail_path: Path
    feature_importance_path: Path
    next_prediction_path: Path
    actual_vs_pred_fig_path: Path
    residual_fig_path: Path
