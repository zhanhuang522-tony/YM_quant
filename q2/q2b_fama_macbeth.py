"""
题目 2B：对 TC / PWMA / CFO 因子做 Fama-MacBeth 检验

复用 Q1B 的期货面板数据（data/futures_panel_1h.csv），
对每个币种计算 TC_n12、PWMA_10、CFO_14，
然后用与 Q1B 完全相同的 Fama-MacBeth 框架运行。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 复用 q2a_factors 中的因子函数
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2a_factors import tc_factor, pwma_factor, cfo_factor

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PANEL_CSV = DATA_DIR / "futures_panel_1h.csv"

FACTOR_COLS = ["TC_n12", "PWMA_10", "CFO_14"]

# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_panel_data() -> pd.DataFrame:
    if not PANEL_CSV.exists():
        raise FileNotFoundError(
            f"{PANEL_CSV} not found.\n"
            "Please run  python q1/q1b_fama_macbeth.py  first to download and cache the panel data."
        )
    print(f"Loading cached panel from {PANEL_CSV}")
    df = pd.read_csv(PANEL_CSV, parse_dates=["open_time"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df

# ─────────────────────────────────────────────
# 因子计算
# ─────────────────────────────────────────────

def build_factors_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("open_time").reset_index(drop=True)
    close = data["close"].astype(float)

    data["TC_n12"] = tc_factor(close, n=12).values
    data["PWMA_10"] = pwma_factor(close, n=10).values
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
                        n_groups: int = 5) -> float:
    needed = [factor, "fwd_ret", "open_time"]
    df = panel_feat.dropna(subset=needed).copy()

    ls_rets = []
    for t, sub in df.groupby("open_time"):
        if len(sub) < n_groups * 2:
            continue
        sub = sub.copy()
        sub["quintile"] = pd.qcut(sub[factor], n_groups, labels=False,
                                   duplicates="drop")
        q_ret = sub.groupby("quintile")["fwd_ret"].mean()
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

    panel_raw = load_panel_data()
    print(f"\n面板数据: {panel_raw['symbol'].nunique()} 个币种, {len(panel_raw)} 行")

    print("\n计算 TC_n12 / PWMA_10 / CFO_14 因子...")
    panel_feat = build_panel_factors(panel_raw)
    print(f"因子面板: {len(panel_feat)} 行")

    # 各因子有效行数
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
    for f in FACTOR_COLS:
        sharpe = quintile_ls_sharpe(panel_feat, f)
        print(f"{f:<12} {sharpe:>12.3f}")


if __name__ == "__main__":
    main()
