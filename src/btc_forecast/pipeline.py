from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .binance_client import BinanceClient
from .config import AppConfig, resolve_paths
from .evaluate import compute_forecast_metrics, compute_naive_metrics
from .features import FEATURE_COLUMNS, TARGET_COL, build_feature_dataset, build_panel_feature_dataset
from .model import TrainedModel, feature_importance, fit_model, load_model, predict, save_model
from .split import apply_window, generate_rolling_windows

logger = logging.getLogger(__name__)


def _ensure_parent(paths: list[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def _symbol_filename(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def fetch_hourly_data(cfg: AppConfig, root: Path, base_url: str = "https://api.binance.com") -> pd.DataFrame:
    paths = resolve_paths(cfg, root)
    _ensure_parent([paths.target_raw_path, paths.panel_raw_path])
    paths.raw_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(cfg.data.start_utc, tz="UTC").to_pydatetime()
    end = pd.Timestamp(cfg.data.end_utc, tz="UTC").to_pydatetime()
    symbols = list(dict.fromkeys([cfg.data.target_symbol] + cfg.data.symbols))

    client = BinanceClient(base_url=base_url)
    panel_list: list[pd.DataFrame] = []

    for symbol in symbols:
        raw = client.fetch_ohlcv(symbol=symbol, interval=cfg.data.timeframe, start_utc=start, end_utc=end)
        raw["symbol"] = symbol
        raw.to_csv(paths.raw_dir / f"{_symbol_filename(symbol)}_{cfg.data.timeframe}.csv", index=False)
        panel_list.append(raw)

    panel = pd.concat(panel_list, ignore_index=True).sort_values(["open_time", "symbol"]).reset_index(drop=True)
    panel.to_csv(paths.panel_raw_path, index=False)

    target = panel[panel["symbol"] == cfg.data.target_symbol].copy()
    if target.empty:
        raise RuntimeError(f"Target symbol not found in panel: {cfg.data.target_symbol}")
    target.to_csv(paths.target_raw_path, index=False)

    logger.info("Fetched panel rows=%s symbols=%s", len(panel), len(symbols))
    return panel


def build_datasets(cfg: AppConfig, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = resolve_paths(cfg, root)
    _ensure_parent([paths.feature_path, paths.panel_feature_path])

    target_raw = pd.read_csv(paths.target_raw_path)
    panel_raw = pd.read_csv(paths.panel_raw_path)

    target_feat = build_feature_dataset(target_raw)
    panel_feat = build_panel_feature_dataset(panel_raw)

    target_feat.to_parquet(paths.feature_path, index=False)
    panel_feat.to_parquet(paths.panel_feature_path, index=False)

    logger.info("Built target features rows=%s and panel features rows=%s", len(target_feat), len(panel_feat))
    return target_feat, panel_feat


def _load_target_features(cfg: AppConfig, root: Path) -> pd.DataFrame:
    paths = resolve_paths(cfg, root)
    return pd.read_parquet(paths.feature_path).sort_values("open_time").reset_index(drop=True)


def _blend(y_pred: np.ndarray, weight: float) -> np.ndarray:
    return weight * y_pred


def run_backtest(cfg: AppConfig, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = resolve_paths(cfg, root)
    _ensure_parent([paths.backtest_metrics_path, paths.backtest_detail_path, paths.model_ranking_path])

    df = _load_target_features(cfg, root)
    windows = generate_rolling_windows(
        timestamps=df["open_time"],
        train_hours=cfg.cv.train_hours,
        valid_hours=cfg.cv.valid_hours,
        step_hours=cfg.cv.step_hours,
    )
    if not windows:
        raise RuntimeError("No CV windows generated")

    metric_records: list[dict] = []
    pred_records: list[dict] = []

    for model_name in cfg.model.enabled:
        for i, window in enumerate(windows, start=1):
            train_df, valid_df = apply_window(df, window)
            if train_df.empty or valid_df.empty:
                continue

            trained = fit_model(
                model_name=model_name,
                train_df=train_df,
                feature_cols=FEATURE_COLUMNS,
                target_col=TARGET_COL,
                seed=cfg.model.random_seed,
                ridge_alpha=cfg.model.ridge_alpha,
                lasso_alpha=cfg.model.lasso_alpha,
                lgb_params=cfg.model.lightgbm_params,
            )
            y_true_train = train_df[TARGET_COL].to_numpy()
            y_pred_train = _blend(predict(trained, train_df), cfg.model.prediction_blend_weight)
            y_true_valid = valid_df[TARGET_COL].to_numpy()
            y_pred_valid = _blend(predict(trained, valid_df), cfg.model.prediction_blend_weight)

            train_metrics = compute_forecast_metrics(y_true_train, y_pred_train)
            valid_metrics = compute_forecast_metrics(y_true_valid, y_pred_valid)
            valid_naive = compute_naive_metrics(y_true_valid)

            metric_records.append(
                {
                    "model": model_name,
                    "window_id": i,
                    "train_start": str(window.train_start),
                    "train_end": str(window.train_end),
                    "valid_start": str(window.valid_start),
                    "valid_end": str(window.valid_end),
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"valid_{k}": v for k, v in valid_metrics.items()},
                    **{f"valid_naive_{k}": v for k, v in valid_naive.items()},
                    "valid_ic_gain_vs_naive": valid_metrics["ic"] - valid_naive["ic"],
                    "valid_sharpe_gain_vs_naive": valid_metrics["sharpe"] - valid_naive["sharpe"],
                    "valid_mse_gain_vs_naive": valid_naive["mse"] - valid_metrics["mse"],
                    "valid_beats_naive_mse": bool(valid_metrics["mse"] < valid_naive["mse"]),
                }
            )

            for ts, y_t, y_p, close in zip(valid_df["open_time"], y_true_valid, y_pred_valid, valid_df["close"], strict=True):
                pred_records.append(
                    {
                        "model": model_name,
                        "window_id": i,
                        "open_time": ts,
                        "y_true": float(y_t),
                        "y_pred": float(y_p),
                        "close": float(close),
                        "predicted_close_t1": float(close * np.exp(y_p)),
                        "actual_close_t1": float(close * np.exp(y_t)),
                    }
                )

    metric_df = pd.DataFrame(metric_records)
    pred_df = pd.DataFrame(pred_records)

    ranking = (
        metric_df.groupby("model", as_index=False)
        .agg(
            cv_valid_ic_mean=("valid_ic", "mean"),
            cv_valid_sharpe_mean=("valid_sharpe", "mean"),
            cv_valid_mse_mean=("valid_mse", "mean"),
            cv_valid_beats_naive_ratio=("valid_beats_naive_mse", "mean"),
        )
        .sort_values(["cv_valid_ic_mean", "cv_valid_sharpe_mean", "cv_valid_mse_mean"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    metric_df.to_csv(paths.backtest_metrics_path, index=False)
    pred_df.to_csv(paths.backtest_detail_path, index=False)
    ranking.to_csv(paths.model_ranking_path, index=False)

    logger.info("Backtest done models=%s windows=%s", len(cfg.model.enabled), len(windows))
    return metric_df, pred_df, ranking


def _best_model_name(cfg: AppConfig, root: Path) -> str:
    paths = resolve_paths(cfg, root)
    ranking = pd.read_csv(paths.model_ranking_path)
    if ranking.empty:
        raise RuntimeError("Model ranking empty")
    return str(ranking.iloc[0]["model"])


def train_best_model(cfg: AppConfig, root: Path) -> TrainedModel:
    paths = resolve_paths(cfg, root)
    _ensure_parent([paths.model_artifact_path, paths.feature_importance_path])

    df = _load_target_features(cfg, root)
    holdout_rows = min(cfg.cv.holdout_hours, len(df) - 1)
    train_df = df.iloc[:-holdout_rows].copy()
    if train_df.empty:
        raise RuntimeError("No training rows")

    model_name = _best_model_name(cfg, root)
    trained = fit_model(
        model_name=model_name,
        train_df=train_df,
        feature_cols=FEATURE_COLUMNS,
        target_col=TARGET_COL,
        seed=cfg.model.random_seed,
        ridge_alpha=cfg.model.ridge_alpha,
        lasso_alpha=cfg.model.lasso_alpha,
        lgb_params=cfg.model.lightgbm_params,
    )
    save_model(trained, paths.model_artifact_path)
    feature_importance(trained).to_csv(paths.feature_importance_path, index=False)

    logger.info("Trained best model=%s", model_name)
    return trained


def evaluate_holdout(cfg: AppConfig, root: Path) -> pd.DataFrame:
    paths = resolve_paths(cfg, root)
    _ensure_parent([paths.holdout_metrics_path])

    df = _load_target_features(cfg, root)
    model = load_model(paths.model_artifact_path)

    holdout_rows = min(cfg.cv.holdout_hours, len(df) - 1)
    holdout_df = df.iloc[-holdout_rows:].copy()
    y_true = holdout_df[TARGET_COL].to_numpy()
    y_pred = _blend(predict(model, holdout_df), cfg.model.prediction_blend_weight)

    model_metrics = compute_forecast_metrics(y_true, y_pred)
    naive_metrics = compute_naive_metrics(y_true)
    out = pd.DataFrame(
        [
            {"model": model.name, **model_metrics},
            {"model": "naive_zero_return", **naive_metrics},
        ]
    )
    out["mse_gain_vs_naive"] = np.where(out["model"] == model.name, naive_metrics["mse"] - model_metrics["mse"], 0.0)
    out["ic_gain_vs_naive"] = np.where(out["model"] == model.name, model_metrics["ic"] - naive_metrics["ic"], 0.0)
    out["sharpe_gain_vs_naive"] = np.where(
        out["model"] == model.name, model_metrics["sharpe"] - naive_metrics["sharpe"], 0.0
    )
    out.to_csv(paths.holdout_metrics_path, index=False)

    logger.info("Holdout done model=%s", model.name)
    return out


def predict_next_hour(cfg: AppConfig, root: Path) -> dict:
    paths = resolve_paths(cfg, root)
    _ensure_parent([paths.next_prediction_path])

    df = _load_target_features(cfg, root)
    model = load_model(paths.model_artifact_path)
    last_row = df.iloc[-1]

    pred_ret = float(_blend(predict(model, pd.DataFrame([last_row])), cfg.model.prediction_blend_weight)[0])
    asof = pd.to_datetime(last_row["open_time"], utc=True)
    close_now = float(last_row["close"])

    result = {
        "symbol": cfg.data.target_symbol,
        "asof_time_utc": asof.isoformat(),
        "target_time_utc": (asof + pd.Timedelta(hours=1)).isoformat(),
        "predicted_log_return_1h": pred_ret,
        "predicted_close": float(close_now * np.exp(pred_ret)),
    }

    with paths.next_prediction_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def run_fama_macbeth(cfg: AppConfig, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = resolve_paths(cfg, root)
    _ensure_parent([paths.fama_beta_ts_path, paths.fama_summary_path, paths.fama_group_backtest_path])

    panel = pd.read_parquet(paths.panel_feature_path).copy()
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    factors = [c for c in cfg.fama_macbeth.factor_columns if c in panel.columns]
    if not factors:
        raise RuntimeError("No valid factor columns for Fama-MacBeth")

    beta_rows: list[dict] = []
    group_rows: list[dict] = []

    for ts, g in panel.groupby("open_time", sort=True):
        g = g.dropna(subset=factors + [cfg.features.target_col]).copy()
        if len(g) < max(cfg.fama_macbeth.min_cross_section, len(factors) + 2):
            continue

        X = g[factors].to_numpy(float)
        y = g[cfg.features.target_col].to_numpy(float)
        X = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        row = {"open_time": ts, "beta_intercept": float(beta[0])}
        for i, fac in enumerate(factors, start=1):
            row[f"beta_{fac}"] = float(beta[i])
        beta_rows.append(row)

        score = X @ beta
        if np.std(score) < 1e-12:
            continue
        ic = float(np.corrcoef(score, y)[0, 1]) if np.std(y) > 1e-12 else 0.0

        try:
            q = pd.qcut(score, q=cfg.fama_macbeth.quantiles, labels=False, duplicates="drop")
            q = np.asarray(q)
            top = y[q == np.nanmax(q)]
            bottom = y[q == np.nanmin(q)]
            ls_ret = float(np.mean(top) - np.mean(bottom)) if len(top) and len(bottom) else 0.0
        except ValueError:
            ls_ret = 0.0

        group_rows.append({"open_time": ts, "ic": ic, "long_short_ret": ls_ret})

    beta_ts = pd.DataFrame(beta_rows)
    if beta_ts.empty:
        raise RuntimeError("No valid cross-sections for Fama-MacBeth")

    summary_rows: list[dict] = []
    for col in [c for c in beta_ts.columns if c.startswith("beta_")]:
        s = beta_ts[col].dropna().to_numpy(float)
        if len(s) < 2:
            continue
        mean_beta = float(np.mean(s))
        se = float(np.std(s, ddof=1) / np.sqrt(len(s)))
        t_value = mean_beta / se if se > 1e-12 else 0.0
        summary_rows.append({"factor": col.replace("beta_", ""), "mean_beta": mean_beta, "t_value": t_value, "n": len(s)})

    group_df = pd.DataFrame(group_rows)
    if not group_df.empty:
        ls = group_df["long_short_ret"].to_numpy(float)
        sharpe = float(np.mean(ls) / np.std(ls, ddof=1) * np.sqrt(24 * 365)) if len(ls) > 1 and np.std(ls, ddof=1) > 1e-12 else 0.0
        group_df.attrs["long_short_sharpe"] = sharpe

    summary_df = pd.DataFrame(summary_rows).sort_values("t_value", ascending=False).reset_index(drop=True)

    beta_ts.to_csv(paths.fama_beta_ts_path, index=False)
    summary_df.to_csv(paths.fama_summary_path, index=False)
    group_df.to_csv(paths.fama_group_backtest_path, index=False)

    logger.info("Fama-MacBeth done sections=%s", len(beta_ts))
    return beta_ts, summary_df, group_df


def run_all(cfg: AppConfig, root: Path, base_url: str = "https://api.binance.com") -> dict:
    paths = resolve_paths(cfg, root)
    if not paths.target_raw_path.exists() or not paths.panel_raw_path.exists():
        fetch_hourly_data(cfg, root, base_url=base_url)
    if not paths.feature_path.exists() or not paths.panel_feature_path.exists():
        build_datasets(cfg, root)

    run_backtest(cfg, root)
    train_best_model(cfg, root)
    evaluate_holdout(cfg, root)
    run_fama_macbeth(cfg, root)
    return predict_next_hour(cfg, root)
