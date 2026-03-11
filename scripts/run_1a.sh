#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"

btc-predict fetch --config "$CONFIG_PATH"
btc-predict build-dataset --config "$CONFIG_PATH"
btc-predict backtest --config "$CONFIG_PATH"
btc-predict train --config "$CONFIG_PATH"
btc-predict evaluate-holdout --config "$CONFIG_PATH"
btc-predict predict-next-hour --config "$CONFIG_PATH"
