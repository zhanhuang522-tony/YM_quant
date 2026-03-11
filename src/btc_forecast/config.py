from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    timeframe: str = "1h"
    start_utc: str
    end_utc: str
    target_symbol: str = "BTCUSDT"
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
    raw_dir: str = "data/raw/hourly"
    target_raw_path: str = "data/raw/hourly/target_1h.csv"
    panel_raw_path: str = "data/raw/hourly/panel_1h.csv"


class FeatureConfig(BaseModel):
    target_col: str = "target_log_ret_1h"
    feature_path: str = "data/processed/btc_1h_features.parquet"
    panel_feature_path: str = "data/processed/panel_1h_features.parquet"


class CVConfig(BaseModel):
    train_hours: int = Field(default=24 * 90, ge=24)
    valid_hours: int = Field(default=24 * 7, ge=24)
    step_hours: int = Field(default=24 * 7, ge=24)
    holdout_hours: int = Field(default=24 * 30, ge=24)


class ModelConfig(BaseModel):
    enabled: list[str] = Field(default_factory=lambda: ["ols", "ridge", "lasso", "lightgbm"])
    random_seed: int = 42
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.0005
    prediction_blend_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    lightgbm_params: dict[str, float | int | str] = Field(
        default_factory=lambda: {
            "objective": "huber",
            "metric": "l1",
            "verbosity": -1,
            "num_leaves": 31,
            "learning_rate": 0.03,
            "n_estimators": 500,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "subsample_freq": 5,
            "min_child_samples": 40,
        }
    )


class FamaMacBethConfig(BaseModel):
    factor_columns: list[str] = Field(
        default_factory=lambda: [
            "log_ret_1h",
            "log_ret_3h",
            "log_ret_6h",
            "vol_24h",
            "rv_24h",
            "volume_over_ma_24h",
            "rsi_14",
            "macd_hist",
            "bb_width_20",
        ]
    )
    min_cross_section: int = Field(default=10, ge=5)
    quantiles: int = Field(default=5, ge=3, le=10)


class OutputConfig(BaseModel):
    backtest_metrics_path: str = "outputs/metrics/backtest_metrics.csv"
    backtest_detail_path: str = "outputs/metrics/backtest_predictions.csv"
    model_ranking_path: str = "outputs/metrics/model_ranking.csv"
    holdout_metrics_path: str = "outputs/metrics/holdout_metrics.csv"
    next_prediction_path: str = "outputs/predictions/next_hour_prediction.json"
    model_artifact_path: str = "outputs/models/best_model.pkl"
    feature_importance_path: str = "outputs/metrics/feature_importance.csv"
    fama_beta_ts_path: str = "outputs/fama/beta_timeseries.csv"
    fama_summary_path: str = "outputs/fama/fama_macbeth_summary.csv"
    fama_group_backtest_path: str = "outputs/fama/group_backtest.csv"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    features: FeatureConfig
    cv: CVConfig
    model: ModelConfig
    fama_macbeth: FamaMacBethConfig
    outputs: OutputConfig


class ResolvedPaths(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_dir: Path
    target_raw_path: Path
    panel_raw_path: Path
    feature_path: Path
    panel_feature_path: Path
    backtest_metrics_path: Path
    backtest_detail_path: Path
    model_ranking_path: Path
    holdout_metrics_path: Path
    next_prediction_path: Path
    model_artifact_path: Path
    feature_importance_path: Path
    fama_beta_ts_path: Path
    fama_summary_path: Path
    fama_group_backtest_path: Path


    @property
    def model_parent_paths(self) -> list[Path]:
        return [
            self.target_raw_path,
            self.panel_raw_path,
            self.feature_path,
            self.panel_feature_path,
            self.backtest_metrics_path,
            self.backtest_detail_path,
            self.model_ranking_path,
            self.holdout_metrics_path,
            self.next_prediction_path,
            self.model_artifact_path,
            self.feature_importance_path,
            self.fama_beta_ts_path,
            self.fama_summary_path,
            self.fama_group_backtest_path,
        ]


def resolve_paths(cfg: AppConfig, root: Path) -> ResolvedPaths:
    return ResolvedPaths(
        raw_dir=root / cfg.data.raw_dir,
        target_raw_path=root / cfg.data.target_raw_path,
        panel_raw_path=root / cfg.data.panel_raw_path,
        feature_path=root / cfg.features.feature_path,
        panel_feature_path=root / cfg.features.panel_feature_path,
        backtest_metrics_path=root / cfg.outputs.backtest_metrics_path,
        backtest_detail_path=root / cfg.outputs.backtest_detail_path,
        model_ranking_path=root / cfg.outputs.model_ranking_path,
        holdout_metrics_path=root / cfg.outputs.holdout_metrics_path,
        next_prediction_path=root / cfg.outputs.next_prediction_path,
        model_artifact_path=root / cfg.outputs.model_artifact_path,
        feature_importance_path=root / cfg.outputs.feature_importance_path,
        fama_beta_ts_path=root / cfg.outputs.fama_beta_ts_path,
        fama_summary_path=root / cfg.outputs.fama_summary_path,
        fama_group_backtest_path=root / cfg.outputs.fama_group_backtest_path,
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
