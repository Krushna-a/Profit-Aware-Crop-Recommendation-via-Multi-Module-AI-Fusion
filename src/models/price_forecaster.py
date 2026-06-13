"""
models/price_forecaster.py
---------------------------
Module C: Commodity price forecasting using a sequential MLP.

Forecasts the monthly market price (USD/MT) of wheat, rice, and cotton
for the upcoming harvest window using World Bank Pink Sheet data.

Architecture
------------
The model treats price forecasting as a supervised regression problem
via sliding window transformation:
  Features: [p_{t-24}, p_{t-23}, ..., p_{t-1}]  (24-month lookback)
  Target:   p_t                                   (next month price)

An MLPRegressor with ReLU activations and Adam optimizer is trained
per crop on chronologically split data.

Production LSTM Reference (why LSTM is the right choice at scale)
------------------------------------------------------------------
In production this module should use an LSTM network. The math is
documented here for interview completeness.

LSTM cell equations (timestep t):
  f_t = σ(W_f·[h_{t−1}, x_t] + b_f)    [forget gate: what to erase]
  i_t = σ(W_i·[h_{t−1}, x_t] + b_i)    [input gate: what to write]
  g_t = tanh(W_g·[h_{t−1}, x_t] + b_g) [candidate values]
  o_t = σ(W_o·[h_{t−1}, x_t] + b_o)    [output gate: what to expose]
  c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t      [cell state: long-term memory]
  h_t = o_t ⊙ tanh(c_t)                 [hidden state: short-term memory]

Vanishing Gradient Problem in Vanilla RNNs:
  h_t = tanh(W·h_{t−1} + U·x_t)
  ∂L/∂h_1 = ∏_{t=2}^T (W^T · diag(tanh'(h_t)))
  When the spectral radius ρ(W) < 1, this product shrinks exponentially.
  Gradient from time T cannot reach time 1 -> network cannot learn
  long-range dependencies.

LSTM Fix — Constant Error Carousel (CEC):
  ∂c_t/∂c_{t−1} = f_t  (forget gate)
  f_t is a learned gate; the network drives it near 1.0 when long
  memory is needed. The cell state gradient flows back additively,
  not multiplicatively — no exponential decay.
  This enables learning dependencies spanning hundreds of timesteps.

Adam Optimizer (used for LSTM training):
  First  moment:  m_t = β₁·m_{t−1} + (1−β₁)·∇L
  Second moment:  v_t = β₂·v_{t−1} + (1−β₂)·(∇L)²
  Bias correction: m̂_t = m_t/(1−β₁^t),  v̂_t = v_t/(1−β₂^t)
  Update: θ <- θ − α · m̂_t / (√v̂_t + ε)
  Per-parameter adaptive rates are critical for noisy time-series gradients.

Data Leakage Reminder
---------------------
NEVER use a random train/test split for time-series models.
Random split: model trains on p_900 while predicting p_300 -> look-ahead bias.
MAPE appears ~1%, but the model is completely useless for real forecasting.

This class enforces chronological ordering by accepting only pre-split
data from PriceSeriesPreprocessor which applies a strict sequential cut.
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import math

logger = logging.getLogger(__name__)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error:
      MAPE = (1/n) Σ |y_true − y_pred| / |y_true| × 100

    Expressed as a percentage. Well-suited for price series where
    absolute magnitude varies over time (avoids scale bias vs RMSE).
    Note: undefined when y_true = 0; guarded with a small epsilon.
    """
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])
                         / np.abs(y_true[mask])) * 100)


class PriceForecaster:

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.models: Dict[str, MLPRegressor] = {}

    def _build_model(self) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=tuple(self.cfg["hidden_layers"]),
            activation=self.cfg["activation"],
            solver="adam",
            learning_rate_init=self.cfg["learning_rate"],
            max_iter=self.cfg["max_iter"],
            early_stopping=self.cfg["early_stopping"],
            validation_fraction=0.10,
            n_iter_no_change=self.cfg["n_iter_no_change"],
            random_state=42,
        )

    def train(self, split_data: Dict[str, Dict]) -> None:
        """
        Trains one MLP per crop on chronologically split sliding-window data.

        Parameters
        ----------
        split_data : output of PriceSeriesPreprocessor.prepare()
                     {crop -> {X_train, y_train, X_val, y_val, ...}}
        """
        for crop, d in split_data.items():
            # Concatenate train + val for final model training
            # Val was used to monitor early stopping during development
            X_tr = np.vstack([d["X_train"], d["X_val"]])
            y_tr = np.concatenate([d["y_train"], d["y_val"]])

            model = self._build_model()
            model.fit(X_tr, y_tr)
            self.models[crop] = model
            logger.info("[Module C] Forecaster trained for %s | "
                        "train_samples=%d | iters=%d",
                        crop, len(X_tr), model.n_iter_)

    def evaluate(self, split_data: Dict[str, Dict],
                 preprocessor) -> Dict[str, Dict]:
        """
        Evaluates on the chronological test set (last 10% of months).

        Metrics
        -------
        MAPE: interpretable in % terms, good for commodity prices.
        RMSE: in original USD/MT units after inverse transform.
        Direction accuracy: % of months where we correctly predicted
          price direction (up/down) — often more useful for trading decisions.
        """
        results = {}
        for crop, d in split_data.items():
            X_te = d["X_test"]
            y_te = d["y_test"]
            if len(X_te) == 0:
                logger.warning("No test samples for %s", crop)
                continue

            preds_scaled = self.models[crop].predict(X_te)
            y_true_usd   = preprocessor.inverse_transform(crop, y_te)
            y_pred_usd   = preprocessor.inverse_transform(crop, preds_scaled)

            mape = _mape(y_true_usd, y_pred_usd)
            rmse = math.sqrt(mean_squared_error(y_true_usd, y_pred_usd))

            # Direction accuracy: sign of month-over-month change
            if len(y_true_usd) > 1:
                true_dir = np.sign(np.diff(y_true_usd))
                pred_dir = np.sign(np.diff(y_pred_usd))
                dir_acc  = float(np.mean(true_dir == pred_dir))
            else:
                dir_acc = float("nan")

            logger.info("[Module C] %s — MAPE: %.2f%% | RMSE: $%.2f/MT | "
                        "Direction Acc: %.2f%%",
                        crop, mape, rmse, dir_acc * 100)
            results[crop] = {
                "mape":          mape,
                "rmse_usd":      rmse,
                "direction_acc": dir_acc,
                "y_true":        y_true_usd,
                "y_pred":        y_pred_usd,
            }
        return results

    def forecast_horizon(self, crop: str, preprocessor,
                         price_df, horizon: int = 3) -> np.ndarray:
        """
        Iterative multi-step forecast for `horizon` months ahead.
        Each predicted value is appended to the window before the next step.

        This is the autoregressive / recursive forecasting strategy:
          ŷ_{T+1} = f([p_{T-23}, ..., p_T])
          ŷ_{T+2} = f([p_{T-22}, ..., p_T, ŷ_{T+1}])
          ...
        Error accumulates over the horizon; acceptable for 3-month windows.

        Returns
        -------
        np.ndarray of shape (horizon,) — forecast prices in USD/MT
        """
        window = preprocessor.get_last_window(crop, price_df).flatten()
        forecasts_scaled = []
        for _ in range(horizon):
            pred = self.models[crop].predict(window.reshape(1, -1))[0]
            forecasts_scaled.append(pred)
            window = np.append(window[1:], pred)

        return preprocessor.inverse_transform(
            crop, np.array(forecasts_scaled)
        )

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        for crop, model in self.models.items():
            path = Path(directory) / f"price_forecaster_{crop}.joblib"
            joblib.dump(model, path)
        logger.info("PriceForecaster models saved -> %s", directory)

    @classmethod
    def load(cls, directory: str, cfg: dict) -> "PriceForecaster":
        obj = cls(cfg)
        for path in Path(directory).glob("price_forecaster_*.joblib"):
            crop = path.stem.replace("price_forecaster_", "")
            obj.models[crop] = joblib.load(path)
            logger.info("PriceForecaster [%s] loaded <- %s", crop, path)
        return obj
