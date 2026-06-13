"""
models/yield_regressor.py
--------------------------
Module B: Crop yield regression using XGBoost.

The model predicts expected crop yield (Tonnes/Ha) given current soil
nutrient levels and climate conditions. Trained on FAOSTAT India data
joined with the soil/weather feature distributions from the crop dataset.

Mathematical Basis — Gradient Boosted Trees (Friedman, 2001)
-------------------------------------------------------------
XGBoost builds an additive ensemble: F_M(x) = sum_{m=1}^M eta * h_m(x)

At each round m, the objective is:
  L^(m) = sum_i l(y_i, F_{m-1}(x_i) + f_m(x_i)) + Omega(f_m)

Using second-order Taylor expansion around F_{m-1}(x_i):
  L^(m) ≈ sum_i [g_i * f_m(x_i) + (1/2) * h_i * f_m(x_i)^2] + Omega(f_m)

where:
  g_i = ∂l(y_i, F_{m-1}(x_i)) / ∂F_{m-1}(x_i)    [first-order gradient]
  h_i = ∂²l(y_i, F_{m-1}(x_i)) / ∂F_{m-1}(x_i)²  [Hessian / curvature]

For MSE loss l = (1/2)(y − ŷ)²:
  g_i = ŷ_i − y_i   [residual]
  h_i = 1

Regularization term:
  Omega(f) = γT + (1/2)λ||w||²
  T = number of leaves, w = leaf weights, λ = L2 penalty

Optimal leaf weight for leaf j (closed-form solution):
  w_j* = −(Σ_{i∈j} g_i) / (Σ_{i∈j} h_i + λ)

Best split score for a candidate split:
  Gain = (1/2) * [(Σg_L)²/(Σh_L+λ) + (Σg_R)²/(Σh_R+λ) − (Σg)²/(Σh+λ)] − γ

The Hessian h_i enables adaptive per-sample step sizes, making XGBoost
significantly more efficient than standard gradient boosting for
non-MSE losses (Huber, Tweedie, etc.).

Yield Modeling Note
-------------------
Because we do not have paired (soil-features, yield) observations in a
single dataset, we train on FAOSTAT yearly national averages augmented
with soil/weather percentile features. The regressor learns the
mapping: soil_profile × climate_conditions -> yield_t_ha.
In production, this would be calibrated on field-level sensor data.
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import math

logger = logging.getLogger(__name__)


class YieldRegressor:

    def __init__(self, cfg: dict):
        self.cfg   = cfg
        self.model = xgb.XGBRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],      # eta: shrinkage per step
            max_depth=cfg["max_depth"],
            subsample=cfg["subsample"],              # row subsampling
            colsample_bytree=cfg["colsample_bytree"],# column subsampling
            reg_lambda=cfg["reg_lambda"],            # L2 on leaf weights
            reg_alpha=cfg["reg_alpha"],              # L1 on leaf weights
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=cfg["early_stopping_rounds"],
            random_state=42,
            verbosity=0,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train with early stopping on the validation set.
        Early stopping halts when RMSE on val set stops improving for
        early_stopping_rounds consecutive rounds — prevents overfitting
        without needing to tune n_estimators manually.
        """
        logger.info("Training XGBoost Regressor: max %d rounds, lr=%.3f",
                    self.cfg["n_estimators"], self.cfg["learning_rate"])
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best = self.model.best_iteration
        logger.info("XGBoost best iteration: %d (early stopping active)", best)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Regression metrics:
          RMSE = sqrt(mean((y_true − y_pred)²))
               penalizes large errors more than MAE
          MAE  = mean(|y_true − y_pred|)
               robust to outliers, interpretable in original units (T/Ha)
          R²   = 1 − SS_res/SS_tot
               fraction of variance explained by the model
        """
        preds = self.model.predict(X_test)
        rmse  = math.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        logger.info("[Module B] RMSE: %.4f T/Ha | MAE: %.4f T/Ha | R²: %.4f",
                    rmse, mae, r2)
        return {
            "rmse": rmse,
            "mae":  mae,
            "r2":   r2,
            "feature_importance": self.model.feature_importances_,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("YieldRegressor saved -> %s", path)

    @classmethod
    def load(cls, path: str, cfg: dict) -> "YieldRegressor":
        obj = cls(cfg)
        obj.model = joblib.load(path)
        logger.info("YieldRegressor loaded <- %s", path)
        return obj
