from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from btc_forecast.config import load_config
from btc_forecast.pipeline import (
    build_datasets,
    evaluate_holdout,
    predict_next_hour,
    run_backtest,
    run_fama_macbeth,
    train_best_model,
)


def _make_panel(start: str, periods: int, symbols: list[str]) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    rows = []
    for s_idx, symbol in enumerate(symbols):
        base = 100 + s_idx * 10
        close = base + np.sin(np.arange(periods) / 10.0) * 2 + np.linspace(0, 3, periods)
        volume = np.abs(np.sin(np.arange(periods) / 8.0)) * 100 + 50 + s_idx
        for i, t in enumerate(ts):
            rows.append(
                {
                    "open_time": t,
                    "open": close[i] - 0.3,
                    "high": close[i] + 0.5,
                    "low": close[i] - 0.6,
                    "close": close[i],
                    "volume": volume[i],
                    "quote_asset_volume": volume[i] * close[i],
                    "trades": 100 + i % 20,
                    "symbol": symbol,
                }
            )
    return pd.DataFrame(rows)


def test_pipeline_smoke(tmp_path: Path) -> None:
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
    panel = _make_panel("2026-01-01", periods=24 * 50, symbols=symbols)

    cfg_text = """
data:
  timeframe: "1h"
  start_utc: "2026-01-01T00:00:00Z"
  end_utc: "2026-02-19T23:00:00Z"
  target_symbol: "BTCUSDT"
  symbols: ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
  raw_dir: data/raw/hourly
  target_raw_path: data/raw/hourly/target_1h.csv
  panel_raw_path: data/raw/hourly/panel_1h.csv
features:
  target_col: target_log_ret_1h
  feature_path: data/processed/btc_1h_features.parquet
  panel_feature_path: data/processed/panel_1h_features.parquet
cv:
  train_hours: 24
  valid_hours: 24
  step_hours: 24
  holdout_hours: 24
model:
  enabled: ["ols", "ridge"]
  random_seed: 42
  ridge_alpha: 1.0
  lasso_alpha: 0.001
  prediction_blend_weight: 0.2
  lightgbm_params:
    n_estimators: 10
fama_macbeth:
  factor_columns: [log_ret_1h, log_ret_3h, vol_24h, rsi_14]
  min_cross_section: 5
  quantiles: 3
outputs:
  backtest_metrics_path: outputs/metrics/backtest_metrics.csv
  backtest_detail_path: outputs/metrics/backtest_predictions.csv
  model_ranking_path: outputs/metrics/model_ranking.csv
  holdout_metrics_path: outputs/metrics/holdout_metrics.csv
  next_prediction_path: outputs/predictions/next_hour_prediction.json
  model_artifact_path: outputs/models/best_model.pkl
  feature_importance_path: outputs/metrics/feature_importance.csv
  fama_beta_ts_path: outputs/fama/beta_timeseries.csv
  fama_summary_path: outputs/fama/fama_macbeth_summary.csv
  fama_group_backtest_path: outputs/fama/group_backtest.csv
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(cfg_path)

    (tmp_path / "data/raw/hourly").mkdir(parents=True, exist_ok=True)
    panel.to_csv(tmp_path / cfg.data.panel_raw_path, index=False)
    panel[panel["symbol"] == cfg.data.target_symbol].to_csv(tmp_path / cfg.data.target_raw_path, index=False)

    build_datasets(cfg, tmp_path)
    m, p, r = run_backtest(cfg, tmp_path)
    assert not m.empty
    assert not p.empty
    assert not r.empty

    train_best_model(cfg, tmp_path)
    h = evaluate_holdout(cfg, tmp_path)
    assert not h.empty

    beta, summary, group = run_fama_macbeth(cfg, tmp_path)
    assert not beta.empty
    assert not summary.empty
    assert not group.empty

    pred = predict_next_hour(cfg, tmp_path)
    assert "predicted_log_return_1h" in pred
