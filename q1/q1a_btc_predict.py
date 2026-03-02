"""
题目 1A：BTC/USDT 1h 收益率预测

- 数据：Binance 现货 BTC/USDT 1h OHLCV，近 365 天
         + Binance 永续合约资金费率（每 8h 结算）
         + Binance 持仓量历史（1h，最近约 28 天）
- 特征：动量、波动率、量价、技术指标、资金费率、持仓量变化（共 15 个）
- 模型：Ridge + LightGBM，TimeSeriesSplit(5折) 滚动验证
- 评估：IC、ICIR、方向准确率、多空 Sharpe
- 输出：每模型 CV 均值/std 指标表，holdout 评估，下一小时预测值
"""

from __future__ import annotations

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
_EXISTING_BTC_CSV = DATA_DIR / "raw" / "hourly" / "BTCUSDT_1h.csv"
FR_CSV  = DATA_DIR / "btc_funding_rate.csv"
OI_CSV  = DATA_DIR / "btc_open_interest.csv"

# ─────────────────────────────────────────────
# 数据拉取 — K 线
# ─────────────────────────────────────────────

def _get(url: str, params: dict) -> list | dict:
    for attempt in range(5):
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            wait = min(2 ** attempt, 16)
            print(f"  retry {attempt+1}/5: {exc}, wait {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed: {url}")


def load_btc_data() -> pd.DataFrame:
    for csv_path in [_EXISTING_BTC_CSV, RAW_CSV]:
        if csv_path.exists():
            print(f"Loading cached BTC data from {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["open_time"])
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.sort_values("open_time").reset_index(drop=True)

    print("Fetching BTC/USDT 1h OHLCV from Binance (past 365 days)...")
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=365)
    cursor = int(start_utc.timestamp() * 1000)
    end_ms  = int(end_utc.timestamp() * 1000)
    step_ms = 3_600_000
    rows = []
    while cursor <= end_ms:
        batch = _get("https://api.binance.com/api/v3/klines",
                     {"symbol": "BTCUSDT", "interval": "1h",
                      "startTime": cursor, "endTime": end_ms, "limit": 1000})
        if not isinstance(batch, list) or not batch:
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
# 数据拉取 — 资金费率
# ─────────────────────────────────────────────

def load_funding_rate() -> pd.DataFrame:
    """资金费率：永续合约每 8 小时结算一次，全年约 1095 条。"""
    if FR_CSV.exists():
        print(f"Loading cached funding rate from {FR_CSV}")
        df = pd.read_csv(FR_CSV)
        df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
        return df

    print("Fetching BTC funding rate from Binance Futures...")
    end_utc   = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=365)
    cursor    = int(start_utc.timestamp() * 1000)
    end_ms    = int(end_utc.timestamp() * 1000)
    rows = []
    while cursor <= end_ms:
        batch = _get("https://fapi.binance.com/fapi/v1/fundingRate",
                     {"symbol": "BTCUSDT", "startTime": cursor, "limit": 1000})
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        time.sleep(0.05)
        if len(batch) < 1000:
            break

    df = pd.DataFrame(rows)
    df["time"]         = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["time", "funding_rate"]].sort_values("time").reset_index(drop=True)
    df.to_csv(FR_CSV, index=False)
    print(f"Saved {len(df)} rows → {df.time.min()} to {df.time.max()}")
    return df

# ─────────────────────────────────────────────
# 数据拉取 — 持仓量（Open Interest）
# ─────────────────────────────────────────────

def load_open_interest() -> pd.DataFrame:
    """
    Binance /futures/data/openInterestHist 只保留最近约 28 天数据，
    不支持 startTime 参数，只能用 endTime 向前翻页。
    """
    if OI_CSV.exists():
        print(f"Loading cached open interest from {OI_CSV}")
        df = pd.read_csv(OI_CSV)
        df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
        return df

    print("Fetching BTC open interest from Binance Futures...")
    rows = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while True:
        batch = _get("https://fapi.binance.com/futures/data/openInterestHist",
                     {"symbol": "BTCUSDT", "period": "1h",
                      "limit": 500, "endTime": end_ms})
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        oldest_ts = int(batch[0]["timestamp"])
        end_ms_next = oldest_ts - 1
        # Check if older data exists
        probe = _get("https://fapi.binance.com/futures/data/openInterestHist",
                     {"symbol": "BTCUSDT", "period": "1h",
                      "limit": 1, "endTime": end_ms_next})
        if not isinstance(probe, list) or not probe:
            break
        end_ms = end_ms_next
        time.sleep(0.05)

    if not rows:
        print("  No OI data available")
        return pd.DataFrame(columns=["time", "open_interest"])

    df = pd.DataFrame(rows)
    df["time"]          = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    df = (df[["time", "open_interest"]]
          .sort_values("time")
          .drop_duplicates("time")
          .reset_index(drop=True))
    df.to_csv(OI_CSV, index=False)
    print(f"Saved {len(df)} rows → {df.time.min()} to {df.time.max()}")
    return df

# ─────────────────────────────────────────────
# 特征工程
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # 动量（5 个）
    "log_ret_1h", "log_ret_3h", "log_ret_6h", "log_ret_12h", "log_ret_24h",
    # 波动率（2 个）
    "vol_24h", "rv_24h",
    # 量价（2 个）
    "volume_chg_1h", "volume_over_ma_24h",
    # 技术指标（3 个）
    "rsi_14", "macd_hist", "bb_width_20",
    # 资金费率（2 个，全年覆盖）
    "funding_rate", "funding_rate_ma3",
    # 持仓量变化（1 个，近 28 天有值，其余填 0）
    "oi_chg_1h",
]
TARGET_COL = "target"


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def build_features(df_btc: pd.DataFrame,
                   df_fr: pd.DataFrame,
                   df_oi: pd.DataFrame) -> pd.DataFrame:
    data  = df_btc.copy().sort_values("open_time").reset_index(drop=True)
    close = data["close"]

    # ── 价格动量 ──────────────────────────────
    data["log_ret_1h"]  = np.log(close / close.shift(1))
    data["log_ret_3h"]  = np.log(close / close.shift(3))
    data["log_ret_6h"]  = np.log(close / close.shift(6))
    data["log_ret_12h"] = np.log(close / close.shift(12))
    data["log_ret_24h"] = np.log(close / close.shift(24))

    # ── 波动率 ────────────────────────────────
    data["vol_24h"] = data["log_ret_1h"].rolling(24, min_periods=24).std()
    data["rv_24h"]  = np.sqrt((data["log_ret_1h"] ** 2).rolling(24, min_periods=24).sum())

    # ── 量价 ──────────────────────────────────
    data["volume_chg_1h"]    = data["volume"].pct_change(1)
    vol_ma24                  = data["volume"].rolling(24, min_periods=24).mean()
    data["volume_over_ma_24h"] = data["volume"] / vol_ma24 - 1.0

    # ── 技术指标 ──────────────────────────────
    data["rsi_14"] = _rsi(close, 14)
    ema12          = close.ewm(span=12, adjust=False).mean()
    ema26          = close.ewm(span=26, adjust=False).mean()
    macd           = ema12 - ema26
    data["macd_hist"]   = macd - macd.ewm(span=9, adjust=False).mean()
    ma20                = close.rolling(20, min_periods=20).mean()
    std20               = close.rolling(20, min_periods=20).std()
    data["bb_width_20"] = (4 * std20) / ma20.replace(0.0, np.nan)

    # ── 资金费率 ──────────────────────────────
    # 每 8h 结算一次，forward-fill 到每小时（占用的 8h 周期内费率不变）
    if not df_fr.empty:
        fr = df_fr.set_index("time")["funding_rate"].sort_index()
        # 把 fr 重建为小时频率索引，然后前向填充
        hourly_idx = data["open_time"]
        fr_hourly  = fr.reindex(hourly_idx).ffill()
        data["funding_rate"] = fr_hourly.values
        # funding_rate_ma3：最近 3 次结算费率的均值（即过去约 24h 内结算的均值）
        fr_shifted = fr.reindex(hourly_idx, method="ffill")
        # 计算在时序上最近 3 个结算点的滚动均值（用 ffill 后的值做 ewm 近似）
        data["funding_rate_ma3"] = (
            pd.Series(data["funding_rate"].values, index=hourly_idx)
            .ewm(span=3, adjust=False).mean().values
        )
    else:
        data["funding_rate"]    = 0.0
        data["funding_rate_ma3"] = 0.0

    # ── 持仓量变化 ────────────────────────────
    # OI 只有最近约 28 天；无数据时填 0（中性假设，不引入偏差）
    if not df_oi.empty:
        oi = df_oi.set_index("time")["open_interest"].sort_index()
        oi_hourly = oi.reindex(data["open_time"])      # 仅对齐，不 ffill
        oi_hourly_ffill = oi.reindex(data["open_time"], method="ffill")
        # 1h % 变化（对数变化，与价格收益率一致）
        oi_log_chg = np.log(oi_hourly / oi_hourly_ffill.shift(1))
        data["oi_chg_1h"] = oi_log_chg.values
    else:
        data["oi_chg_1h"] = 0.0

    # OI 缺失期（约 92% 的行）填 0，保留全量训练数据
    data["oi_chg_1h"] = data["oi_chg_1h"].fillna(0.0)

    # ── 目标变量 ──────────────────────────────
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
    ic = (float(np.corrcoef(y_pred, y_true)[0, 1])
          if np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12 else 0.0)
    dir_acc  = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    strat    = np.sign(y_pred) * y_true
    std_ret  = np.std(strat, ddof=1)
    sharpe   = float(np.mean(strat) / std_ret * np.sqrt(24 * 365)) if std_ret > 1e-12 else 0.0
    return {"ic": ic, "dir_acc": dir_acc, "sharpe": sharpe}

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
    raw   = load_btc_data()
    df_fr = load_funding_rate()
    df_oi = load_open_interest()

    print(f"\nBTC K 线: {len(raw)} 行  |  "
          f"资金费率: {len(df_fr)} 条  |  "
          f"持仓量: {len(df_oi)} 条（{df_oi.time.min().date() if not df_oi.empty else 'N/A'} 起）")

    data = build_features(raw, df_fr, df_oi)
    print(f"\n特征数据集: {len(data)} 行, {len(FEATURE_COLS)} 个特征")
    print(f"特征列: {FEATURE_COLS}")

    # 2. 计算每个特征与目标的 IC（帮助理解各特征贡献）
    print("\n各特征与 target 的相关系数（IC）：")
    for f in FEATURE_COLS:
        ic_f = float(np.corrcoef(data[f], data[TARGET_COL])[0, 1])
        print(f"  {f:<24}: {ic_f:+.4f}")

    X = data[FEATURE_COLS].values
    y = data[TARGET_COL].values

    # 3. 划分 holdout（最后 15%）
    n_ho = int(len(data) * 0.15)
    X_cv, y_cv = X[:-n_ho], y[:-n_ho]
    X_ho, y_ho = X[-n_ho:], y[-n_ho:]
    X_next = X[[-1]]

    # 4. CV 评估
    models = ["Ridge"]
    if HAS_LGB:
        models.append("LightGBM")

    cv_results  = {}
    fold_records = []
    for mname in models:
        print(f"\n--- {mname} 5折时序 CV ---")
        tscv         = TimeSeriesSplit(n_splits=5)
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
            X_tr, X_val = X_cv[train_idx], X_cv[val_idx]
            y_tr, y_val = y_cv[train_idx], y_cv[val_idx]

            scaler   = StandardScaler()
            X_tr_s   = scaler.fit_transform(X_tr)
            X_val_s  = scaler.transform(X_val)

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
            print(f"  Fold {fold_idx+1}: IC={m['ic']:.4f}  "
                  f"DirAcc={m['dir_acc']:.4f}  Sharpe={m['sharpe']:.3f}")
            fold_records.append({"model": mname, "fold": fold_idx + 1, **m})

        keys = fold_metrics[0].keys()
        agg  = {}
        for k in keys:
            vals         = [fm[k] for fm in fold_metrics]
            agg[f"{k}_mean"] = np.mean(vals)
            agg[f"{k}_std"]  = np.std(vals)
        ic_vals   = [fm["ic"] for fm in fold_metrics]
        agg["icir"] = np.mean(ic_vals) / (np.std(ic_vals) + 1e-12)
        cv_results[mname] = agg

    # 5. 打印 CV 汇总
    print("\n" + "=" * 60)
    print("CV 汇总（5折均值 ± std）")
    print("=" * 60)
    print(f"{'模型':<12} {'IC_mean':>10} {'IC_std':>9} {'ICIR':>8} "
          f"{'DirAcc':>9} {'Sharpe':>9}")
    print("-" * 60)
    cv_rows = []
    for mname, agg in cv_results.items():
        print(f"{mname:<12} {agg['ic_mean']:>10.4f} {agg['ic_std']:>9.4f} "
              f"{agg['icir']:>8.3f} {agg['dir_acc_mean']:>9.4f} "
              f"{agg['sharpe_mean']:>9.3f}")
        cv_rows.append({"model": mname, **agg})

    pd.DataFrame(cv_rows).to_csv(OUT_DIR / "cv_summary.csv", index=False)
    pd.DataFrame(fold_records).to_csv(OUT_DIR / "cv_folds.csv", index=False)

    # 6. Holdout 评估 + 下一小时预测
    print("\n" + "=" * 60)
    print(f"Holdout 评估（最后 {n_ho} 根 K 线 ≈ 15%）")
    print("=" * 60)
    holdout_rows = []
    pred_rows    = []
    last_time    = data["open_time"].iloc[-1]
    last_close   = float(data["close"].iloc[-1])
    next_time    = last_time + pd.Timedelta(hours=1)

    for mname in models:
        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_cv)
        X_ho_s  = scaler.transform(X_ho)
        X_nxt_s = scaler.transform(X_next)

        if mname == "Ridge":
            model = Ridge(alpha=1.0)
            model.fit(X_tr_s, y_cv)
        else:
            model = lgb.LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, n_jobs=-1, random_state=42, verbose=-1,
            )
            model.fit(X_tr_s, y_cv, callbacks=[lgb.log_evaluation(period=-1)])

        y_pred_ho      = model.predict(X_ho_s)
        m_ho           = compute_metrics(y_ho, y_pred_ho)
        next_pred_logret = float(model.predict(X_nxt_s)[0])
        next_price     = last_close * np.exp(next_pred_logret)
        price_chg_pct  = (next_price / last_close - 1) * 100

        print(f"\n[{mname}]  Holdout IC={m_ho['ic']:.4f}  "
              f"DirAcc={m_ho['dir_acc']:.4f}  Sharpe={m_ho['sharpe']:.3f}")
        print(f"  当前价格 ({last_time}):    ${last_close:>12,.2f}")
        print(f"  预测价格 ({next_time}): ${next_price:>12,.2f}  "
              f"({next_price - last_close:+.2f} / {price_chg_pct:+.4f}%)")
        print(f"  方向: {'看多' if next_pred_logret > 0 else '看空'}")

        holdout_rows.append({"model": mname, **m_ho})
        pred_rows.append({
            "model":           mname,
            "as_of_time":      str(last_time),
            "current_price_usd": round(last_close, 2),
            "pred_time":       str(next_time),
            "pred_price_usd":  round(next_price, 2),
            "pred_log_ret":    next_pred_logret,
            "pred_chg_pct":    round(price_chg_pct, 6),
            "direction":       "long" if next_pred_logret > 0 else "short",
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
