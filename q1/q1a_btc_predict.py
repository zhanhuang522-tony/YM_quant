"""
题目 1A：BTC/USDT 1h 收益率预测

- 数据：Binance 现货 BTC/USDT 1h OHLCV，近 365 天
- 特征：动量、波动率、量价、技术指标（共 ~11 个）
- 模型：Ridge + LightGBM，TimeSeriesSplit(5折) 滚动验证
- 评估：IC、ICIR、方向准确率、多空 Sharpe
- 输出：每模型 CV 均值/std 指标表，holdout 评估，下一小时预测值
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: lightgbm not installed, skipping LGB model")

# ─────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
RAW_CSV = DATA_DIR / "btc_1h_raw.csv"
# 优先使用已有缓存
_EXISTING_BTC_CSV = DATA_DIR / "raw" / "hourly" / "BTCUSDT_1h.csv"

# ─────────────────────────────────────────────
# 数据拉取
# ─────────────────────────────────────────────

def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> list:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval,
              "startTime": start_ms, "endTime": end_ms, "limit": limit}
    for attempt in range(5):
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            wait = min(2 ** attempt, 16)
            print(f"  fetch retry {attempt+1}/5: {exc}, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("Failed to fetch klines after 5 retries")


def load_btc_data() -> pd.DataFrame:
    # 优先使用已有缓存（src 管道落盘的文件）
    for csv_path in [_EXISTING_BTC_CSV, RAW_CSV]:
        if csv_path.exists():
            print(f"Loading cached data from {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["open_time"])
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.sort_values("open_time").reset_index(drop=True)

    print("Fetching BTC/USDT 1h OHLCV from Binance (past 365 days)...")
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=365)
    step_ms = 3_600_000  # 1h in ms

    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    cursor = start_ms
    rows = []

    while cursor <= end_ms:
        batch = _fetch_klines("BTCUSDT", "1h", cursor, end_ms)
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + step_ms
        time.sleep(0.05)
        if len(batch) < 1000:
            break

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_base_vol", "taker_quote_vol", "ignore"]
    df = pd.DataFrame(rows, columns=cols).drop(columns=["ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("open_time").reset_index(drop=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"Saved to {RAW_CSV} ({len(df)} rows)")
    return df

# ─────────────────────────────────────────────
# 特征工程
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "log_ret_1h", "log_ret_3h", "log_ret_6h", "log_ret_12h", "log_ret_24h",
    "vol_24h", "rv_24h",
    "volume_chg_1h", "volume_over_ma_24h",
    "rsi_14", "macd_hist", "bb_width_20",
]
TARGET_COL = "target"


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("open_time").reset_index(drop=True)
    close = data["close"]

    # 动量
    data["log_ret_1h"] = np.log(close / close.shift(1))
    data["log_ret_3h"] = np.log(close / close.shift(3))
    data["log_ret_6h"] = np.log(close / close.shift(6))
    data["log_ret_12h"] = np.log(close / close.shift(12))
    data["log_ret_24h"] = np.log(close / close.shift(24))

    # 波动率
    data["vol_24h"] = data["log_ret_1h"].rolling(24, min_periods=24).std()
    data["rv_24h"] = np.sqrt((data["log_ret_1h"] ** 2).rolling(24, min_periods=24).sum())

    # 量价
    data["volume_chg_1h"] = data["volume"].pct_change(1)
    vol_ma24 = data["volume"].rolling(24, min_periods=24).mean()
    data["volume_over_ma_24h"] = data["volume"] / vol_ma24 - 1.0

    # RSI
    data["rsi_14"] = _rsi(close, 14)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    data["macd_hist"] = macd - macd_sig

    # Bollinger Band Width
    ma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    data["bb_width_20"] = (4 * std20) / ma20.replace(0.0, np.nan)

    # 目标变量
    data[TARGET_COL] = np.log(close.shift(-1) / close)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return data

# ─────────────────────────────────────────────
# 评估指标
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) < 2:
        return {"ic": 0.0, "dir_acc": 0.0, "sharpe": 0.0}
    # IC
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        ic = 0.0
    else:
        ic = float(np.corrcoef(y_pred, y_true)[0, 1])
    # Direction accuracy
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    # Long-short Sharpe
    strat_ret = np.sign(y_pred) * y_true
    std_ret = np.std(strat_ret, ddof=1)
    sharpe = float(np.mean(strat_ret) / std_ret * np.sqrt(24 * 365)) if std_ret > 1e-12 else 0.0
    return {"ic": ic, "dir_acc": dir_acc, "sharpe": sharpe}

# ─────────────────────────────────────────────
# 交叉验证
# ─────────────────────────────────────────────

def run_cv(X: np.ndarray, y: np.ndarray, model_name: str, n_splits: int = 5) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_val_s)
        elif model_name == "LightGBM":
            model = lgb.LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, n_jobs=-1, random_state=42, verbose=-1,
            )
            model.fit(X_tr_s, y_tr,
                      eval_set=[(X_val_s, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(period=-1)])
            y_pred = model.predict(X_val_s)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        m = compute_metrics(y_val, y_pred)
        fold_metrics.append(m)
        print(f"  Fold {fold_idx+1}: IC={m['ic']:.4f}  DirAcc={m['dir_acc']:.4f}  Sharpe={m['sharpe']:.3f}")

    # Aggregate
    keys = fold_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [fm[k] for fm in fold_metrics]
        agg[f"{k}_mean"] = np.mean(vals)
        agg[f"{k}_std"] = np.std(vals)

    # ICIR
    ic_vals = [fm["ic"] for fm in fold_metrics]
    agg["icir"] = np.mean(ic_vals) / (np.std(ic_vals) + 1e-12)
    return agg


def train_final_model(X_tr: np.ndarray, y_tr: np.ndarray,
                      X_ho: np.ndarray, y_ho: np.ndarray,
                      X_next: np.ndarray, model_name: str) -> tuple[dict, float]:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_ho_s = scaler.transform(X_ho)
    X_next_s = scaler.transform(X_next)

    if model_name == "Ridge":
        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, y_tr)
    elif model_name == "LightGBM":
        model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, n_jobs=-1, random_state=42, verbose=-1,
        )
        model.fit(X_tr_s, y_tr, callbacks=[lgb.log_evaluation(period=-1)])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    y_pred_ho = model.predict(X_ho_s)
    metrics = compute_metrics(y_ho, y_pred_ho)
    next_pred = float(model.predict(X_next_s)[0])
    return metrics, next_pred

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("题目 1A: BTC/USDT 1h 收益率预测")
    print("=" * 60)

    OUT_DIR = ROOT / "outputs" / "q1a"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    raw = load_btc_data()
    data = build_features(raw)
    print(f"\n特征数据集: {len(data)} 行, {len(FEATURE_COLS)} 个特征")

    X = data[FEATURE_COLS].values
    y = data[TARGET_COL].values

    # 2. 划分 holdout（最后 15%）
    n_ho = int(len(data) * 0.15)
    X_cv, y_cv = X[:-n_ho], y[:-n_ho]
    X_ho, y_ho = X[-n_ho:], y[-n_ho:]
    X_next = X[[-1]]

    # 3. CV 评估
    models = ["Ridge"]
    if HAS_LGB:
        models.append("LightGBM")

    # 删除旧的 run_cv / train_final_model 调用路径，统一在 main 中完成
    cv_results = {}
    fold_records = []
    for mname in models:
        print(f"\n--- {mname} 5折时序 CV ---")
        tscv = TimeSeriesSplit(n_splits=5)
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
            X_tr, X_val = X_cv[train_idx], X_cv[val_idx]
            y_tr, y_val = y_cv[train_idx], y_cv[val_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            if mname == "Ridge":
                model = Ridge(alpha=1.0)
                model.fit(X_tr_s, y_tr)
                y_pred = model.predict(X_val_s)
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, n_jobs=-1, random_state=42, verbose=-1,
                )
                model.fit(X_tr_s, y_tr,
                          eval_set=[(X_val_s, y_val)],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(period=-1)])
                y_pred = model.predict(X_val_s)
            m = compute_metrics(y_val, y_pred)
            fold_metrics.append(m)
            print(f"  Fold {fold_idx+1}: IC={m['ic']:.4f}  DirAcc={m['dir_acc']:.4f}  Sharpe={m['sharpe']:.3f}")
            fold_records.append({"model": mname, "fold": fold_idx + 1, **m})

        keys = fold_metrics[0].keys()
        agg = {}
        for k in keys:
            vals = [fm[k] for fm in fold_metrics]
            agg[f"{k}_mean"] = np.mean(vals)
            agg[f"{k}_std"] = np.std(vals)
        ic_vals = [fm["ic"] for fm in fold_metrics]
        agg["icir"] = np.mean(ic_vals) / (np.std(ic_vals) + 1e-12)
        cv_results[mname] = agg

    # 4. 打印并保存 CV 汇总
    print("\n" + "=" * 60)
    print("CV 汇总（5折均值 ± std）")
    print("=" * 60)
    header = f"{'模型':<12} {'IC_mean':>10} {'IC_std':>9} {'ICIR':>8} {'DirAcc':>9} {'Sharpe':>9}"
    print(header)
    print("-" * 60)
    cv_rows = []
    for mname, agg in cv_results.items():
        print(f"{mname:<12} {agg['ic_mean']:>10.4f} {agg['ic_std']:>9.4f} "
              f"{agg['icir']:>8.3f} {agg['dir_acc_mean']:>9.4f} {agg['sharpe_mean']:>9.3f}")
        cv_rows.append({"model": mname, **agg})

    pd.DataFrame(cv_rows).to_csv(OUT_DIR / "cv_summary.csv", index=False)
    pd.DataFrame(fold_records).to_csv(OUT_DIR / "cv_folds.csv", index=False)

    # 5. Holdout 评估 + 下一小时预测
    print("\n" + "=" * 60)
    print(f"Holdout 评估（最后 {n_ho} 根 K 线 ≈ 15%）")
    print("=" * 60)
    holdout_rows = []
    pred_rows = []
    last_time = data["open_time"].iloc[-1]
    last_close = float(data["close"].iloc[-1])
    next_time = last_time + pd.Timedelta(hours=1)

    for mname in models:
        m_ho, next_pred_logret = train_final_model(X_cv, y_cv, X_ho, y_ho, X_next, mname)
        # 对数收益率转换为价格：P_{t+1} = P_t * exp(log_ret)
        next_price = last_close * np.exp(next_pred_logret)
        price_chg = next_price - last_close
        price_chg_pct = (next_price / last_close - 1) * 100

        print(f"\n[{mname}]  Holdout IC={m_ho['ic']:.4f}  "
              f"DirAcc={m_ho['dir_acc']:.4f}  Sharpe={m_ho['sharpe']:.3f}")
        print(f"  当前价格 ({last_time}):    ${last_close:>12,.2f}")
        print(f"  预测价格 ({next_time}): ${next_price:>12,.2f}  "
              f"({price_chg:+.2f} / {price_chg_pct:+.4f}%)")
        print(f"  方向: {'看多' if next_pred_logret > 0 else '看空'}")

        holdout_rows.append({"model": mname, **m_ho})
        pred_rows.append({
            "model": mname,
            "as_of_time": str(last_time),
            "current_price_usd": round(last_close, 2),
            "pred_time": str(next_time),
            "pred_price_usd": round(next_price, 2),
            "pred_log_ret": next_pred_logret,
            "pred_chg_pct": round(price_chg_pct, 6),
            "direction": "long" if next_pred_logret > 0 else "short",
        })

    pd.DataFrame(holdout_rows).to_csv(OUT_DIR / "holdout_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "next_hour_prediction.csv", index=False)

    print(f"\n结果已保存到 {OUT_DIR}/")
    print(f"  cv_summary.csv           — CV 各模型汇总")
    print(f"  cv_folds.csv             — 每折详细指标")
    print(f"  holdout_metrics.csv      — Holdout 评估")
    print(f"  next_hour_prediction.csv — 下一小时价格预测")


if __name__ == "__main__":
    main()
