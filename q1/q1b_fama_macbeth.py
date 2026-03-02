"""
题目 1B：Fama-MacBeth 截面回归检验

- 数据：Binance 永续合约 ~20 个主流币种，1h OHLCV，近 365 天
- 因子：与 1A 相同的特征（动量/波动率/量价/技术指标）
- 流程：
    1. 每小时 t 做横截面 OLS: r_{i,t+1} = alpha_t + sum(beta_k_t * f_k_{i,t})
    2. 对每个因子的 beta_t 时序：beta_mean, t_stat = beta_mean / (std / sqrt(T))
    3. 按因子值 5 分位分组，做多空组合，计算 Sharpe
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
PANEL_CSV = DATA_DIR / "futures_panel_1h.csv"
_EXISTING_PANEL_CSV = DATA_DIR / "raw" / "hourly" / "panel_1h.csv"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT", "LINKUSDT",
    "UNIUSDT", "ATOMUSDT", "XRPUSDT", "LTCUSDT", "ETCUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
]

FACTOR_COLS = [
    "log_ret_1h", "log_ret_3h", "log_ret_6h", "log_ret_12h", "log_ret_24h",
    "vol_24h", "rv_24h",
    "volume_chg_1h", "volume_over_ma_24h",
    "rsi_14", "macd_hist", "bb_width_20",
]

# ─────────────────────────────────────────────
# 数据拉取
# ─────────────────────────────────────────────

def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    url = "https://fapi.binance.com/fapi/v1/klines"  # 合约 API
    params = {"symbol": symbol, "interval": interval,
              "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    for attempt in range(5):
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            wait = min(2 ** attempt, 16)
            print(f"  retry {attempt+1}/5 {symbol}: {exc}, wait {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed for {symbol}")


def fetch_symbol(symbol: str) -> pd.DataFrame:
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=365)
    step_ms = 3_600_000
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    cursor = start_ms
    rows = []
    while cursor <= end_ms:
        batch = _fetch_klines(symbol, "1h", cursor, end_ms)
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + step_ms
        time.sleep(0.05)
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "tb_vol", "tq_vol", "ignore"]
    df = pd.DataFrame(rows, columns=cols).drop(columns=["ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = symbol
    return df.sort_values("open_time").reset_index(drop=True)


def load_panel_data() -> pd.DataFrame:
    if PANEL_CSV.exists():
        print(f"Loading cached panel from {PANEL_CSV}")
        df = pd.read_csv(PANEL_CSV, parse_dates=["open_time"])
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        return df

    print(f"Fetching {len(SYMBOLS)} symbols from Binance Futures...")
    dfs = []
    for i, sym in enumerate(SYMBOLS):
        print(f"  [{i+1}/{len(SYMBOLS)}] {sym}")
        df = fetch_symbol(sym)
        if not df.empty:
            dfs.append(df)
    panel = pd.concat(dfs, ignore_index=True)
    panel.to_csv(PANEL_CSV, index=False)
    print(f"Saved panel to {PANEL_CSV} ({len(panel)} rows)")
    return panel

# ─────────────────────────────────────────────
# 特征工程
# ─────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def build_features_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("open_time").reset_index(drop=True)
    close = data["close"]

    data["log_ret_1h"] = np.log(close / close.shift(1))
    data["log_ret_3h"] = np.log(close / close.shift(3))
    data["log_ret_6h"] = np.log(close / close.shift(6))
    data["log_ret_12h"] = np.log(close / close.shift(12))
    data["log_ret_24h"] = np.log(close / close.shift(24))

    data["vol_24h"] = data["log_ret_1h"].rolling(24, min_periods=24).std()
    data["rv_24h"] = np.sqrt((data["log_ret_1h"] ** 2).rolling(24, min_periods=24).sum())

    data["volume_chg_1h"] = data["volume"].pct_change(1)
    vol_ma24 = data["volume"].rolling(24, min_periods=24).mean()
    data["volume_over_ma_24h"] = data["volume"] / vol_ma24 - 1.0

    data["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    data["macd_hist"] = macd - macd_sig

    ma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    data["bb_width_20"] = (4 * std20) / ma20.replace(0.0, np.nan)

    # 下一期收益率（回归因变量）
    data["fwd_ret"] = np.log(close.shift(-1) / close)
    return data


def build_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        feat = build_features_for_symbol(g)
        out.append(feat)
    df = pd.concat(out, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["open_time", "symbol"]).reset_index(drop=True)

# ─────────────────────────────────────────────
# Fama-MacBeth
# ─────────────────────────────────────────────

def fama_macbeth(panel_feat: pd.DataFrame, factor_cols: list[str],
                 min_obs: int = 10) -> pd.DataFrame:
    """
    每小时做横截面 OLS，收集各因子的 beta_t；
    最后对 beta_t 做 t 检验（Newey-West 方式为简化版标准 t 检验）。
    """
    # 只保留有 fwd_ret 的行
    needed = factor_cols + ["fwd_ret", "open_time", "symbol"]
    df = panel_feat.dropna(subset=needed).copy()

    times = sorted(df["open_time"].unique())
    beta_series = {f: [] for f in factor_cols}

    for t in times:
        sub = df[df["open_time"] == t]
        if len(sub) < min_obs:
            continue
        y = sub["fwd_ret"].values
        X = sub[factor_cols].values
        # OLS with intercept
        X_aug = np.column_stack([np.ones(len(X)), X])
        try:
            coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        except Exception:
            continue
        for j, f in enumerate(factor_cols):
            beta_series[f].append(coef[j + 1])  # skip intercept

    results = []
    for f in factor_cols:
        betas = np.array(beta_series[f])
        T = len(betas)
        if T < 2:
            results.append({"factor": f, "beta_mean": np.nan, "t_stat": np.nan, "T": T, "sig": ""})
            continue
        beta_mean = np.mean(betas)
        t_stat = beta_mean / (np.std(betas, ddof=1) / np.sqrt(T))
        sig = "**" if abs(t_stat) > 2.576 else ("*" if abs(t_stat) > 1.96 else "")
        results.append({"factor": f, "beta_mean": beta_mean, "t_stat": t_stat, "T": T, "sig": sig})

    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# 分组回测
# ─────────────────────────────────────────────

def quintile_ls_sharpe(panel_feat: pd.DataFrame, factor: str,
                        n_groups: int = 5) -> float:
    """
    每小时按因子值分 n_groups 组，多 Q5 空 Q1，计算年化多空 Sharpe。
    """
    needed = [factor, "fwd_ret", "open_time"]
    df = panel_feat.dropna(subset=needed).copy()

    ls_rets = []
    for t, sub in df.groupby("open_time"):
        if len(sub) < n_groups * 2:
            continue
        sub = sub.copy()
        sub["quintile"] = pd.qcut(sub[factor], n_groups, labels=False, duplicates="drop")
        q_ret = sub.groupby("quintile")["fwd_ret"].mean()
        max_q = q_ret.index.max()
        min_q = q_ret.index.min()
        if max_q == min_q:
            continue
        ls_rets.append(q_ret[max_q] - q_ret[min_q])

    if len(ls_rets) < 2:
        return np.nan
    r = np.array(ls_rets)
    std = np.std(r, ddof=1)
    if std < 1e-12:
        return np.nan
    return float(np.mean(r) / std * np.sqrt(24 * 365))

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("题目 1B：Fama-MacBeth 截面回归检验")
    print("=" * 60)

    panel_raw = load_panel_data()
    print(f"\n面板数据: {panel_raw['symbol'].nunique()} 个币种, "
          f"{len(panel_raw)} 行")

    print("\n计算因子特征...")
    panel_feat = build_panel_features(panel_raw)
    print(f"特征面板: {len(panel_feat)} 行")

    # Fama-MacBeth
    print("\n运行 Fama-MacBeth 横截面回归...")
    fm_result = fama_macbeth(panel_feat, FACTOR_COLS)

    print("\n" + "=" * 60)
    print("Fama-MacBeth 结果（|t|>1.96 *, |t|>2.576 **）")
    print("=" * 60)
    print(f"{'Factor':<22} {'Beta_mean':>12} {'t_stat':>10} {'T':>6} {'Sig':>5}")
    print("-" * 60)
    for _, row in fm_result.iterrows():
        print(f"{row['factor']:<22} {row['beta_mean']:>12.6f} "
              f"{row['t_stat']:>10.3f} {int(row['T']):>6} {row['sig']:>5}")

    # 分组 Sharpe
    print("\n" + "=" * 60)
    print("5 分位分组多空 Sharpe（Q5 long - Q1 short）")
    print("=" * 60)
    print(f"{'Factor':<22} {'L-S Sharpe':>12}")
    print("-" * 40)
    for f in FACTOR_COLS:
        sharpe = quintile_ls_sharpe(panel_feat, f)
        print(f"{f:<22} {sharpe:>12.3f}")


if __name__ == "__main__":
    main()
