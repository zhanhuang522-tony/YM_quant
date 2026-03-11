from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def generate_rolling_windows(
    timestamps: pd.Series,
    train_hours: int,
    valid_hours: int,
    step_hours: int,
) -> list[TimeWindow]:
    ts = pd.to_datetime(timestamps, utc=True).sort_values().reset_index(drop=True)
    if ts.empty:
        return []

    train_delta = pd.Timedelta(hours=train_hours)
    valid_delta = pd.Timedelta(hours=valid_hours)
    step_delta = pd.Timedelta(hours=step_hours)

    min_ts = ts.iloc[0]
    max_ts = ts.iloc[-1]
    windows: list[TimeWindow] = []
    train_start = min_ts

    while True:
        train_end = train_start + train_delta - pd.Timedelta(hours=1)
        valid_start = train_end + pd.Timedelta(hours=1)
        valid_end = valid_start + valid_delta - pd.Timedelta(hours=1)
        if valid_end > max_ts:
            break
        windows.append(
            TimeWindow(
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )
        train_start = train_start + step_delta

    return windows


def apply_window(df: pd.DataFrame, window: TimeWindow) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data["open_time"] = pd.to_datetime(data["open_time"], utc=True)

    train_mask = (data["open_time"] >= window.train_start) & (data["open_time"] <= window.train_end)
    valid_mask = (data["open_time"] >= window.valid_start) & (data["open_time"] <= window.valid_end)
    return data.loc[train_mask].copy(), data.loc[valid_mask].copy()
