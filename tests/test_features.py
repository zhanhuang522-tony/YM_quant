from __future__ import annotations

import numpy as np
import pandas as pd

from btc_forecast.features import TARGET_COL, build_feature_dataset


def _make_hourly_df(n: int = 300) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = 40000 + np.linspace(0, 1000, n)
    return pd.DataFrame(
        {
            "open_time": ts,
            "open": close - 5,
            "high": close + 8,
            "low": close - 9,
            "close": close,
            "volume": np.linspace(100, 300, n),
            "quote_asset_volume": np.linspace(10000, 30000, n),
            "trades": np.linspace(1000, 3000, n),
        }
    )


def test_target_is_next_hour_log_return() -> None:
    raw = _make_hourly_df(300)
    out = build_feature_dataset(raw)
    row = out.iloc[0]
    idx = raw.index[raw["open_time"] == row["open_time"]][0]
    expected = float(np.log(raw.loc[idx + 1, "close"] / raw.loc[idx, "close"]))
    assert abs(row[TARGET_COL] - expected) < 1e-12
