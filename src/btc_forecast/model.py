from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    HAS_LGBM = True
except Exception:  # noqa: BLE001
    lgb = None
    HAS_LGBM = False


@dataclass
class TrainedModel:
    name: str
    model: Any
    feature_columns: list[str]


def build_model(name: str, *, seed: int, ridge_alpha: float, lasso_alpha: float, lgb_params: dict[str, Any]):
    n = name.lower()
    if n == "ols":
        return LinearRegression()
    if n == "ridge":
        return Ridge(alpha=ridge_alpha, random_state=seed)
    if n == "lasso":
        return Lasso(alpha=lasso_alpha, random_state=seed, max_iter=5000)
    if n == "lightgbm":
        if HAS_LGBM:
            params = dict(lgb_params)
            params.setdefault("random_state", seed)
            return lgb.LGBMRegressor(**params)
        logger.warning("lightgbm unavailable, fallback to GradientBoostingRegressor")
        return GradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Unsupported model name: {name}")


def fit_model(
    model_name: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    seed: int,
    ridge_alpha: float,
    lasso_alpha: float,
    lgb_params: dict[str, Any],
) -> TrainedModel:
    model = build_model(
        model_name,
        seed=seed,
        ridge_alpha=ridge_alpha,
        lasso_alpha=lasso_alpha,
        lgb_params=lgb_params,
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    return TrainedModel(name=model_name, model=model, feature_columns=feature_cols)


def predict(model: TrainedModel, feature_df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.model.predict(feature_df[model.feature_columns]), dtype=float)


def feature_importance(model: TrainedModel) -> pd.DataFrame:
    m = model.model
    if HAS_LGBM and hasattr(m, "booster_"):
        scores = m.booster_.feature_importance(importance_type="gain")
    elif hasattr(m, "coef_"):
        scores = np.abs(np.asarray(m.coef_, dtype=float))
    elif hasattr(m, "feature_importances_"):
        scores = np.asarray(m.feature_importances_, dtype=float)
    else:
        scores = np.zeros(len(model.feature_columns))
    out = pd.DataFrame({"feature": model.feature_columns, "importance": scores})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model(model: TrainedModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    meta = {"model_name": model.name, "feature_columns": model.feature_columns}
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_model(path: Path) -> TrainedModel:
    with path.open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, TrainedModel):
        raise TypeError("Invalid model artifact")
    return model
