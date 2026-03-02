# YM_quant

记录笔试项目的仓库，涵盖各类量化相关的笔试题目与解题思路。

# 1. 自行收集制作过去 1 年的比特币量价数据集，并根据该数据集预测未来 1 小时的币价（具体方法不限，要求逻辑严谨、步骤完备，需要给出完整代码）。


# 2. 
## A. 根据要求复现因子，要求尽量降低运算时间（需要给出完整因子计算代码）。
- TC：T-12n 小时到 T-n 小时（即不包括最近 n 小时）内每小时价格对日期序列的回归 R 平方（需要至少
4n 个可用数据点）。
- PWMA：https://blog.xcaldata.com/pascals-weighted-moving-average-pwma-a-powerfulindicator/
- CFO：https://library.tradingtechnologies.com/trade/chrt-ti-chande-forecast-oscillator.html#:~:text=The%20Chande%20Forecast%20Oscillator%20plots,zero%20if%20it%20is%20below

## B. 自行收集币安交易所历史合约币种数据，对 A 中的因子进行 Fama-MacBeth 检验（需要给出完整代码）

# 3. 设计一个多因子中性策略，方法不限。要求：
- 写出策略构思全流程，包括但不限于:
- 策略核心投资逻辑；
 - 因子来源及预处理逻辑（无需写出具体因子）；
 - 因子组合方式（线性/非线性、具体模型选取及其原因）；
- 回测方式与绩效评价维度；
- 风控措施；
- 实盘执行和交易注意事项；
(无需给出具体代码).
