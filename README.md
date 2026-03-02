# YM_quant

记录笔试项目的仓库，涵盖各类量化相关的笔试题目与解题思路。

---

## 题目

### 1. 自行收集制作过去 1 年的比特币量价数据集，并根据该数据集预测未来 1 小时的币价（具体方法不限，要求逻辑严谨、步骤完备，需要给出完整代码）。


### 2.

**A.** 根据要求复现因子，要求尽量降低运算时间（需要给出完整因子计算代码）。
- TC：T-12n 小时到 T-n 小时（即不包括最近 n 小时）内每小时价格对日期序列的回归 R 平方（需要至少 4n 个可用数据点）。
- PWMA：https://blog.xcaldata.com/pascals-weighted-moving-average-pwma-a-powerfulindicator/
- CFO：https://library.tradingtechnologies.com/trade/chrt-ti-chande-forecast-oscillator.html

**B.** 自行收集币安交易所历史合约币种数据，对 A 中的因子进行 Fama-MacBeth 检验（需要给出完整代码）

### 3. 设计一个多因子中性策略，方法不限。要求：
- 写出策略构思全流程，包括但不限于:
  - 策略核心投资逻辑；
  - 因子来源及预处理逻辑（无需写出具体因子）；
  - 因子组合方式（线性/非线性、具体模型选取及其原因）；
  - 回测方式与绩效评价维度；
  - 风控措施；
  - 实盘执行和交易注意事项；
- 无需给出具体代码。

---

## 项目结构

```
q1/
└── q1a_btc_predict.py      # 题目1：BTC 1h 价格预测（Ridge + LightGBM）
q2/
├── q2a_factors.py          # 题目2A：TC / PWMA / CFO 因子实现
└── q2b_fama_macbeth.py     # 题目2B：对2A因子做 Fama-MacBeth 检验
q3/
└── strategy_design.md      # 题目3：多因子中性策略设计
```

## 运行方式

```bash
# 安装依赖
pip install pandas numpy scikit-learn lightgbm scipy httpx

# 题目 1（自动拉取数据并预测下一小时价格）
python q1/q1a_btc_predict.py

# 题目 2A（TC / PWMA / CFO 因子计算与验证）
python q2/q2a_factors.py

# 题目 2B（Fama-MacBeth，自动拉取合约面板数据）
python q2/q2b_fama_macbeth.py
```

## 输出

| 脚本 | 输出目录 | 主要文件 |
|------|---------|---------|
| q1a_btc_predict.py | `outputs/q1a/` | cv_summary.csv、holdout_metrics.csv、next_hour_prediction.csv |
| q2b_fama_macbeth.py | `outputs/q2b/` | fama_macbeth_summary.csv、quintile_sharpe.csv |
