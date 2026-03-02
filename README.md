# BTC 1H Forecast + Fama-MacBeth

按“1A + 1B”高分流程实现：
- 1A: 比特币 1 小时收益率预测（多模型、时间序列CV、IC/MSE/方向准确率/Sharpe）
- 1B: 多币种 Fama-MacBeth 横截面检验（beta 均值与 t 值、分组多空回测）

## 1. 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e '.[dev]'
```

## 2. 一键运行

```bash
./scripts/run_all.sh configs/default.yaml
# 或
btc-predict run-all --config configs/default.yaml
```

## 2A. 只跑第一题（1A）

```bash
./scripts/run_1a.sh configs/default.yaml
```

## 3. 分步命令

```bash
btc-predict fetch --config configs/default.yaml
btc-predict build-dataset --config configs/default.yaml
btc-predict backtest --config configs/default.yaml
btc-predict train --config configs/default.yaml
btc-predict evaluate-holdout --config configs/default.yaml
btc-predict fama-macbeth --config configs/default.yaml
btc-predict predict-next-hour --config configs/default.yaml
```

## 4. 方法说明

### 4.1 数据
- 数据源: Binance 现货 `1h` Kline
- 时间范围: 默认过去 1 年
- 标的: 目标 `BTCUSDT`，同时拉多币种用于 Fama-MacBeth
- 清洗: UTC 对齐、按时间排序、数值类型统一

### 4.2 特征工程（1A）
- 动量: `log_ret_1h/3h/6h/12h/24h`
- 波动率: `vol_24h`, `rv_24h`
- 量价: `volume_chg_1h`, `volume_over_ma_24h`
- 技术指标: `RSI`, `MACD`, `Bollinger width`
- 预测目标: `target_log_ret_1h = log(P[t+1]/P[t])`

### 4.3 模型与验证（1A）
- 模型: `OLS`, `Ridge`, `Lasso`, `LightGBM`
- 验证: rolling window 时间序列交叉验证（非随机切分）
- 指标: `IC`, `MSE`, `Direction Accuracy`, `Sharpe`
- 额外稳健化: `prediction_blend_weight` 对预测收益率做收缩，降低过拟合振幅

### 4.4 Fama-MacBeth（1B）
- 每小时横截面回归: `r_{i,t+1} = beta_t * f_{i,t} + eps`
- 对 `beta_t` 做时间均值并计算 t-value
- 同时输出分组多空回测（Top-Bottom）

## 5. 关键输出

- 目标原始数据: `data/raw/hourly/target_1h.csv`
- 多币种原始数据: `data/raw/hourly/panel_1h.csv`
- 目标特征: `data/processed/btc_1h_features.parquet`
- 面板特征: `data/processed/panel_1h_features.parquet`
- 回测明细: `outputs/metrics/backtest_predictions.csv`
- 回测指标: `outputs/metrics/backtest_metrics.csv`
- 模型排名: `outputs/metrics/model_ranking.csv`
- Holdout 指标: `outputs/metrics/holdout_metrics.csv`
- 最终模型: `outputs/models/best_model.pkl`
- 特征重要性: `outputs/metrics/feature_importance.csv`
- Fama beta 时序: `outputs/fama/beta_timeseries.csv`
- Fama 汇总: `outputs/fama/fama_macbeth_summary.csv`
- 分组回测: `outputs/fama/group_backtest.csv`
- 下一小时预测: `outputs/predictions/next_hour_prediction.json`

## 6. 备注
- 若网络受限，可先用已有 `target_1h.csv / panel_1h.csv` 直接从 `build-dataset` 开始。
- 若 `lightgbm` 不可用，会自动回退到 `GradientBoostingRegressor`。
