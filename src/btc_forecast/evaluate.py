from __future__ import annotations

import numpy as np


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _sharpe(ret: np.ndarray, annualization: float = 24 * 365) -> float:
    if len(ret) < 2:
        return 0.0
    std = float(np.std(ret, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(ret) / std * np.sqrt(annualization))


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    mse = float(np.mean(err**2))
    ic = _safe_corr(y_pred, y_true)
    direction_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    strategy_ret = np.sign(y_pred) * y_true
    sharpe = _sharpe(strategy_ret)

    return {
        "mse": mse,
        "ic": ic,
        "direction_acc": direction_acc,
        "sharpe": sharpe,
    }


def compute_naive_metrics(y_true: np.ndarray) -> dict[str, float]:
    return compute_forecast_metrics(y_true=y_true, y_pred=np.zeros_like(y_true))
