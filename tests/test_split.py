from __future__ import annotations

import pandas as pd

from btc_forecast.split import generate_rolling_windows


def test_generate_windows_hourly() -> None:
    ts = pd.date_range("2026-01-01", periods=24 * 30, freq="1h", tz="UTC")
    windows = generate_rolling_windows(ts.to_series(index=range(len(ts))), train_hours=24 * 7, valid_hours=24 * 3, step_hours=24 * 3)
    assert len(windows) > 0
    assert windows[0].train_end < windows[0].valid_start
