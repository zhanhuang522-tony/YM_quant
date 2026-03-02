"""
题目 2A：TC / PWMA / CFO 因子实现

TC  (Time-Correlation R²)：价格与时间线性拟合的决定系数
PWMA (Pascal-Weighted Moving Average)：帕斯卡三角权重移动平均
CFO  (Chande Forecast Oscillator)：价格与线性回归预测值之差的百分比

实现均采用向量化滚动运算（O(N) 或 O(N log N)），不使用逐行 loop。
"""

from __future__ import annotations

import warnings
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# TC 因子
# ─────────────────────────────────────────────

def tc_factor(close: pd.Series, n: int) -> pd.Series:
    """
    TC_n: 在窗口 [T-12n, T-n]（窗口长 W = 11n+1）内，对价格序列与时间做 OLS，
    返回 R²（决定系数）。

    向量化实现：
        x = [0, 1, ..., W-1]（固定时间轴，与位置等价）
        Sx  = W*(W-1)/2
        Sx2 = W*(W-1)*(2W-1)/6
        Sy  = rolling_sum(close, W)
        Sy2 = rolling_sum(close^2, W)
        Sxy = rolling_sum(j*close) - start_idx * Sy   （j = np.arange(N)）
              其中 start_idx_t = t - W + 1 => rolling_sum(j*close) - (j - W + 1)*Sy

        r² = [W*Sxy - Sx*Sy]² / [(W*Sx2 - Sx²)(W*Sy2 - Sy²)]

    至少需要 max(4*n, W) 个有效点才返回值，否则 NaN。
    """
    W = 11 * n + 1
    min_periods = max(4 * n, W)

    close_vals = close.values.astype(float)
    N = len(close_vals)

    # 常数（x 的统计量）
    Sx = W * (W - 1) / 2.0
    Sx2 = W * (W - 1) * (2 * W - 1) / 6.0

    # rolling sums on close
    close_s = pd.Series(close_vals)
    Sy = close_s.rolling(W, min_periods=W).sum().values
    Sy2 = (close_s ** 2).rolling(W, min_periods=W).sum().values

    # Sxy trick: sum_{k=0}^{W-1} k * close[t-W+1+k]
    # = sum_{j=t-W+1}^{t} (j - (t-W+1)) * close[j]
    # = rolling_sum(j*close, W) - (t-W+1)*rolling_sum(close, W)
    j = np.arange(N, dtype=float)
    jclose = pd.Series(j * close_vals)
    rolling_jclose = jclose.rolling(W, min_periods=W).sum().values
    # start_idx at position t: t - W + 1
    start_idx = j - W + 1
    Sxy = rolling_jclose - start_idx * Sy

    numerator = W * Sxy - Sx * Sy
    denom_x = W * Sx2 - Sx ** 2  # scalar > 0
    denom_y = W * Sy2 - Sy ** 2  # array

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(denom_x * denom_y)
        r = np.where(denom > 1e-12, numerator / denom, np.nan)
        r2 = r ** 2

    # Apply min_periods: first min_periods-1 positions are NaN
    r2[:min_periods - 1] = np.nan
    # Also NaN where Sy is NaN (rolling didn't have enough data)
    r2[np.isnan(Sy)] = np.nan

    return pd.Series(r2, index=close.index, name=f"TC_n{n}")


# ─────────────────────────────────────────────
# PWMA 因子
# ─────────────────────────────────────────────

def pwma_factor(close: pd.Series, n: int) -> pd.Series:
    """
    PWMA_n：使用第 n-1 行帕斯卡三角系数作为权重的加权移动平均。
    权重 w_k = C(n-1, k)，k=0..n-1，最新价格对应最大 k（最右端权重最大）。

    PWMA_n[t] = sum_{k=0}^{n-1} C(n-1,k) * close[t-n+1+k] / 2^(n-1)
              = sum_{k=0}^{n-1} C(n-1, n-1-k) * close[t-k] / 2^(n-1)

    用 fftconvolve 实现 O(N log N) 卷积，前 n-1 个值设为 NaN。
    """
    # weights[k] = C(n-1, n-1-k)，k=0 对应最新价格
    weights = np.array([comb(n - 1, n - 1 - k) for k in range(n)], dtype=float)
    weights_norm = weights / weights.sum()  # sum = 2^(n-1) 归一化

    close_vals = close.values.astype(float)
    # fftconvolve: mode='full' 会产生 N+n-1 个点，取前 N 个
    conv = fftconvolve(close_vals, weights_norm, mode="full")[:len(close_vals)]

    # 前 n-1 个窗口不完整，设为 NaN
    conv[:n - 1] = np.nan

    return pd.Series(conv, index=close.index, name=f"PWMA_{n}")


# ─────────────────────────────────────────────
# CFO 因子
# ─────────────────────────────────────────────

def cfo_factor(close: pd.Series, n: int) -> pd.Series:
    """
    CFO_n（Chande Forecast Oscillator）：
      1. 在 [t-n+1, t] 窗口内对价格做 OLS（x = 0..n-1）
      2. 预测值 forecast = slope * (n-1) + intercept
      3. CFO = (close[t] - forecast) * 100 / close[t]

    向量化 rolling OLS（与 TC 同样的 rolling sum 技巧，O(N)）：
        x = [0, 1, ..., n-1]  => Sx = n*(n-1)/2，Sx2 = n*(n-1)*(2n-1)/6
        Sy  = rolling_sum(close, n)
        Sxy = rolling_jclose - start_idx * Sy
        slope = (n*Sxy - Sx*Sy) / (n*Sx2 - Sx²)
        intercept = (Sy - slope*Sx) / n
        forecast = slope*(n-1) + intercept
    """
    close_vals = close.values.astype(float)
    N = len(close_vals)

    Sx = n * (n - 1) / 2.0
    Sx2 = n * (n - 1) * (2 * n - 1) / 6.0
    denom_x = n * Sx2 - Sx ** 2  # scalar

    close_s = pd.Series(close_vals)
    Sy = close_s.rolling(n, min_periods=n).sum().values

    j = np.arange(N, dtype=float)
    jclose = pd.Series(j * close_vals)
    rolling_jclose = jclose.rolling(n, min_periods=n).sum().values
    start_idx = j - n + 1
    Sxy = rolling_jclose - start_idx * Sy

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(abs(denom_x) > 1e-12,
                         (n * Sxy - Sx * Sy) / denom_x, np.nan)
        intercept = (Sy - slope * Sx) / n
        forecast = slope * (n - 1) + intercept
        cfo = np.where(
            np.abs(close_vals) > 1e-12,
            (close_vals - forecast) * 100.0 / close_vals,
            np.nan,
        )

    # 前 n-1 个值不完整
    cfo[:n - 1] = np.nan
    cfo[np.isnan(Sy)] = np.nan

    return pd.Series(cfo, index=close.index, name=f"CFO_{n}")


# ─────────────────────────────────────────────
# 演示 & 验证
# ─────────────────────────────────────────────

def main():
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "data"

    # 优先使用已有缓存（现货或合约均可）
    for candidate in [
        DATA_DIR / "raw" / "hourly" / "BTCUSDT_1h.csv",
        DATA_DIR / "btc_1h_raw.csv",
    ]:
        if candidate.exists():
            print(f"Using cached BTC data from {candidate}")
            df = pd.read_csv(candidate, parse_dates=["open_time"])
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            close = df.set_index("open_time")["close"].dropna()
            break
    else:
        print("No cached data found, generating synthetic price series (N=500)")
        np.random.seed(42)
        N = 500
        close = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(N) * 0.002)),
            name="close",
        )

    print(f"\n价格序列长度: {len(close)}")
    print("=" * 60)

    # ── TC 因子 ──────────────────────────────
    print("\n[TC 因子] 时间-价格线性相关 R²")
    for n in [6, 12, 24]:
        tc = tc_factor(close, n)
        valid = tc.dropna()
        print(f"  TC_n{n:>2}: 窗口={11*n+1:>4}  有效={len(valid):>5}  "
              f"首个NaN位置={tc.first_valid_index()}  "
              f"均值={valid.mean():.4f}  max={valid.max():.4f}")
    # 边界验证：窗口=67时，前66行应为NaN，第67行（index 66）为首个有效值
    tc6 = tc_factor(close, 6)
    W6 = 11 * 6 + 1
    print(f"\n  边界验证 TC_n6（窗口={W6}）:")
    print(f"    前 {W6-1} 行全为 NaN: {tc6.iloc[:W6-1].isna().all()}")
    print(f"    第 {W6} 行有值:        {not pd.isna(tc6.iloc[W6-1])}")

    # ── PWMA 因子 ────────────────────────────
    print("\n[PWMA 因子] 帕斯卡加权移动平均")
    for n in [5, 10, 20]:
        pwma = pwma_factor(close, n)
        valid = pwma.dropna()
        print(f"  PWMA_{n:>2}: 期数={n:>4}  有效={len(valid):>5}  "
              f"前 {n-1} 位 NaN={pwma.iloc[:n-1].isna().all()}  "
              f"第 {n} 位有值={not pd.isna(pwma.iloc[n-1])}")
    print(f"\n  PWMA_10 前 15 行:")
    pwma10 = pwma_factor(close, 10)
    print(pwma10.head(15).to_string())

    # ── CFO 因子 ─────────────────────────────
    print("\n[CFO 因子] Chande 预测振荡器")
    for n in [7, 14, 28]:
        cfo = cfo_factor(close, n)
        valid = cfo.dropna()
        print(f"  CFO_{n:>2}: 期数={n:>4}  有效={len(valid):>5}  "
              f"均值={valid.mean():.4f}  std={valid.std():.4f}")
    print(f"\n  CFO_14 前 20 行:")
    cfo14 = cfo_factor(close, 14)
    print(cfo14.head(20).to_string())

    # ── 综合输出（三因子都有值的最早行开始）────
    print("\n" + "=" * 60)
    print("三因子合并（首个共同有值行起，展示 10 行）")
    print("=" * 60)
    result = pd.DataFrame({
        "close":   close.values,
        "TC_n12":  tc_factor(close, 12).values,
        "PWMA_10": pwma_factor(close, 10).values,
        "CFO_14":  cfo_factor(close, 14).values,
    }, index=close.index)
    print(result.dropna().head(10).to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
