from __future__ import annotations

import logging
import os
from pathlib import Path

import typer

from .config import load_config
from .pipeline import (
    build_datasets,
    evaluate_holdout,
    fetch_hourly_data,
    predict_next_hour,
    run_all,
    run_backtest,
    run_fama_macbeth,
    train_best_model,
)

app = typer.Typer(help="BTC 1h prediction + Fama-MacBeth pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _root() -> Path:
    return Path.cwd()


def _cfg(path: str):
    return load_config(path)


@app.command("fetch")
def cmd_fetch(config: str = typer.Option("configs/default.yaml", help="Path to yaml config")) -> None:
    cfg = _cfg(config)
    base_url = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
    fetch_hourly_data(cfg, _root(), base_url=base_url)


@app.command("build-dataset")
def cmd_build_dataset(config: str = typer.Option("configs/default.yaml", help="Path to yaml config")) -> None:
    cfg = _cfg(config)
    build_datasets(cfg, _root())


@app.command("backtest")
def cmd_backtest(config: str = typer.Option("configs/default.yaml", help="Path to yaml config")) -> None:
    cfg = _cfg(config)
    run_backtest(cfg, _root())


@app.command("train")
def cmd_train(config: str = typer.Option("configs/default.yaml", help="Path to yaml config")) -> None:
    cfg = _cfg(config)
    train_best_model(cfg, _root())


@app.command("evaluate-holdout")
def cmd_evaluate_holdout(config: str = typer.Option("configs/default.yaml", help="Path to yaml config")) -> None:
    cfg = _cfg(config)
    evaluate_holdout(cfg, _root())


@app.command("fama-macbeth")
def cmd_fama_macbeth(config: str = typer.Option("configs/default.yaml", help="Path to yaml config")) -> None:
    cfg = _cfg(config)
    run_fama_macbeth(cfg, _root())


@app.command("predict-next-hour")
def cmd_predict_next_hour(config: str = typer.Option("configs/default.yaml", help="Path to yaml config")) -> None:
    cfg = _cfg(config)
    result = predict_next_hour(cfg, _root())
    logger.info("Prediction: %s", result)


@app.command("run-all")
def cmd_run_all(config: str = typer.Option("configs/default.yaml", help="Path to yaml config")) -> None:
    cfg = _cfg(config)
    base_url = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
    result = run_all(cfg, _root(), base_url=base_url)
    logger.info("Run all done. Prediction: %s", result)


if __name__ == "__main__":
    app()
