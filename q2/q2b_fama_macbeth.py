"""
题目 2B：对 TC / PWMA / CFO 因子做 Fama-MacBeth 检验

- 数据：Binance 永续合约多币种 1h OHLCV（自动拉取或使用本地缓存）
- 对每个币种计算 TC_n12、PWMA_10、CFO_14
- Fama-MacBeth 横截面回归：每小时 t 做 OLS，收集 beta_t 时序，
  最终输出 beta 均值、t 统计量及分组多空 Sharpe
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 复用 q2a_factors 中的因子函数
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2a_factors import tc_factor, pwma_factor, cfo_factor

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PANEL_CSV = DATA_DIR / "futures_panel_1h.csv"
_EXISTING_PANEL_CSV = DATA_DIR / "raw" / "hourly" / "panel_1h.csv"

# 合约币种列表
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "XRPUSDT", "LTCUSDT", "NEARUSDT", "APTUSDT",
]

FACTOR_COLS = ["TC_n12", "PWMA_10", "CFO_14"]

# ─────────────────────────────────────────────
# 数据拉取（仅在无缓存时执行）
# ─────────────────────────────────────────────

def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "1h",
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
    raise RuntimeError(f"Failed to fetch {symbol}")


def fetch_symbol(symbol: str) -> pd.DataFrame:
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=365)
    step_ms = 3_600_000
    cursor = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    rows = []
    while cursor <= end_ms:
        batch = _fetch_klines(symbol, cursor, end_ms)
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
    # 优先使用已有缓存
    for csv_path in [_EXISTING_PANEL_CSV, PANEL_CSV]:
        if csv_path.exists():
            print(f"Loading cached panel from {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["open_time"])
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
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
# 因子计算
# ─────────────────────────────────────────────

def build_factors_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("open_time").reset_index(drop=True)
    close = data["close"].astype(float)

    data["TC_n12"] = tc_factor(close, n=12).values

    # PWMA 转换为价格偏离度（%），使不同量级币种可以做截面比较
    # 正值 = 价格高于 PWMA（上方动量），负值 = 价格低于 PWMA（均值回归信号）
    pwma_raw = pwma_factor(close, n=10).values
    data["PWMA_10"] = (close.values - pwma_raw) / np.where(pwma_raw > 0, pwma_raw, np.nan) * 100

    data["CFO_14"] = cfo_factor(close, n=14).values

    # 下一期对数收益率（因变量）
    data["fwd_ret"] = np.log(close.shift(-1) / close)

    return data


def build_panel_factors(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        feat = build_factors_for_symbol(g)
        out.append(feat)
    df = pd.concat(out, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["open_time", "symbol"]).reset_index(drop=True)

# ─────────────────────────────────────────────
# Fama-MacBeth（与 Q1B 相同框架）
# ─────────────────────────────────────────────

def fama_macbeth(panel_feat: pd.DataFrame, factor_cols: list[str],
                 min_obs: int = 8) -> pd.DataFrame:
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
        X_aug = np.column_stack([np.ones(len(X)), X])
        try:
            coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        except Exception:
            continue
        for j, f in enumerate(factor_cols):
            beta_series[f].append(coef[j + 1])

    results = []
    for f in factor_cols:
        betas = np.array(beta_series[f])
        T = len(betas)
        if T < 2:
            results.append({"factor": f, "beta_mean": np.nan,
                            "t_stat": np.nan, "T": T, "sig": ""})
            continue
        beta_mean = np.mean(betas)
        t_stat = beta_mean / (np.std(betas, ddof=1) / np.sqrt(T))
        sig = "**" if abs(t_stat) > 2.576 else ("*" if abs(t_stat) > 1.96 else "")
        results.append({"factor": f, "beta_mean": beta_mean,
                        "t_stat": t_stat, "T": T, "sig": sig})

    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# 分组 Sharpe
# ─────────────────────────────────────────────

def quintile_ls_sharpe(panel_feat: pd.DataFrame, factor: str,
                        n_groups: int = 3) -> float:
    """3 分组（三分位）以适配 8 个币种的截面。多 T3 空 T1。"""
    needed = [factor, "fwd_ret", "open_time"]
    df = panel_feat.dropna(subset=needed).copy()

    ls_rets = []
    for t, sub in df.groupby("open_time"):
        if len(sub) < n_groups + 1:   # 至少 n_groups+1 个有效观测
            continue
        sub = sub.copy()
        sub["grp"] = pd.qcut(sub[factor], n_groups, labels=False,
                              duplicates="drop")
        q_ret = sub.groupby("grp")["fwd_ret"].mean()
        max_q, min_q = q_ret.index.max(), q_ret.index.min()
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
    print("题目 2B：TC / PWMA / CFO 因子 Fama-MacBeth 检验")
    print("=" * 60)

    OUT_DIR = ROOT / "outputs" / "q2b"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel_raw = load_panel_data()
    print(f"\n面板数据: {panel_raw['symbol'].nunique()} 个币种, {len(panel_raw)} 行")
    print(f"币种列表: {sorted(panel_raw['symbol'].unique())}")

    print("\n计算 TC_n12 / PWMA_10 / CFO_14 因子...")
    panel_feat = build_panel_factors(panel_raw)
    print(f"因子面板: {len(panel_feat)} 行")

    for f in FACTOR_COLS:
        valid = panel_feat[f].notna().sum()
        print(f"  {f}: {valid} 有效行")

    # Fama-MacBeth
    print("\n运行 Fama-MacBeth 横截面回归...")
    fm_result = fama_macbeth(panel_feat, FACTOR_COLS)

    print("\n" + "=" * 60)
    print("Fama-MacBeth 结果（|t|>1.96 *, |t|>2.576 **）")
    print("=" * 60)
    print(f"{'Factor':<12} {'Beta_mean':>14} {'t_stat':>10} {'T':>6} {'Sig':>5}")
    print("-" * 50)
    for _, row in fm_result.iterrows():
        print(f"{row['factor']:<12} {row['beta_mean']:>14.6f} "
              f"{row['t_stat']:>10.3f} {int(row['T']):>6} {row['sig']:>5}")

    # 分组 Sharpe
    print("\n" + "=" * 60)
    print("5 分位分组多空 Sharpe（Q5 long - Q1 short）")
    print("=" * 60)
    print(f"{'Factor':<12} {'L-S Sharpe':>12}")
    print("-" * 26)
    sharpe_rows = []
    for f in FACTOR_COLS:
        sharpe = quintile_ls_sharpe(panel_feat, f)
        print(f"{f:<12} {sharpe:>12.3f}")
        sharpe_rows.append({"factor": f, "ls_sharpe": sharpe})

    # 保存结果
    fm_result.to_csv(OUT_DIR / "fama_macbeth_summary.csv", index=False)
    pd.DataFrame(sharpe_rows).to_csv(OUT_DIR / "quintile_sharpe.csv", index=False)
    print(f"\n结果已保存到 {OUT_DIR}/")
    print(f"  fama_macbeth_summary.csv — FM 检验汇总")
    print(f"  quintile_sharpe.csv      — 分组多空 Sharpe")


if __name__ == "__main__":
    main()
