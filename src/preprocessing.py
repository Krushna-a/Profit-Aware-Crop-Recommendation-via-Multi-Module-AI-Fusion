"""
preprocessing.py
----------------
All feature engineering, scaling, encoding, and data splitting logic.

Design principles
-----------------
- Scalers and encoders are fit ONLY on training data, then applied to
  validation/test sets. Fitting on held-out data is a form of data leakage
  that inflates metrics and is a common mistake in production ML systems.

- For the time-series price module, we use strict chronological splitting.
  A random split on time-series data is catastrophic: the model can see
  future prices while predicting past ones, producing artificially low MAPE
  (~1–2%) that has zero real-world value. See _chronological_split() below.

- All transformations are encapsulated in a Preprocessor class so they
  can be serialized (joblib) alongside the models for serving.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# Features used by Module A (classifier) and Module B (yield regressor)
SOIL_WEATHER_FEATURES = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]


class CropDataPreprocessor:
    """
    Handles preprocessing for the crop classification and yield regression modules.

    Transformations applied
    -----------------------
    1. Column normalization (lowercase, strip whitespace)
    2. Duplicate and null row removal
    3. Restrict dataset to the three target crops for profit modeling
       (wheat, rice, cotton) — other crops remain for classifier training
    4. StandardScaler on continuous features
       z = (x − μ) / σ  → zero mean, unit variance
       This is critical for gradient-based models and ensures no feature
       dominates due to scale differences.
    5. LabelEncoder on crop name → integer class index
    """

    def __init__(self, target_crops: list):
        self.target_crops = [c.lower() for c in target_crops]
        self.scaler  = StandardScaler()
        self.encoder = LabelEncoder()
        self._fitted = False

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        df["label"] = df["label"].str.strip().str.lower()
        before = len(df)
        df = df.drop_duplicates().dropna()
        logger.info("Cleaned crop dataset: %d → %d rows (removed %d)",
                    before, len(df), before - len(df))
        return df

    def split(self, df: pd.DataFrame, test_size: float,
              seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stratified split: preserves class distribution in both train and test.

        Stratification is important here because some crops have fewer than
        100 samples in the 2200-row dataset — a random split could leave
        entire classes absent from training.
        """
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        idx_train, idx_test = next(sss.split(df, df["label"]))
        train = df.iloc[idx_train].reset_index(drop=True)
        test  = df.iloc[idx_test].reset_index(drop=True)
        logger.info("Stratified split: train=%d, test=%d (test_size=%.0f%%)",
                    len(train), len(test), test_size * 100)
        return train, test

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler + encoder on training data, then transform.
        Call transform() for val/test sets — never fit again.
        """
        X = train_df[SOIL_WEATHER_FEATURES].values.astype(float)
        y = train_df["label"].values
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.encoder.fit_transform(y)
        self._fitted = True
        logger.info("Scaler fit: mean=%s (sample), classes=%s",
                    np.round(self.scaler.mean_[:3], 2), list(self.encoder.classes_))
        return X_scaled, y_encoded

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Apply already-fitted scaler and encoder. Raises if not fitted."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        X = df[SOIL_WEATHER_FEATURES].values.astype(float)
        y = df["label"].values
        return self.scaler.transform(X), self.encoder.transform(y)

    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        return self.encoder.inverse_transform(y_encoded)

    def encode_single(self, row: pd.DataFrame) -> np.ndarray:
        """Scale a single observation for inference."""
        X = row[SOIL_WEATHER_FEATURES].values.astype(float)
        return self.scaler.transform(X)


class PriceSeriesPreprocessor:
    """
    Prepares World Bank monthly price series for time-series forecasting.

    CHRONOLOGICAL SPLIT — WHY THIS IS NON-NEGOTIABLE
    --------------------------------------------------
    Commodity prices are a non-stationary time series with autocorrelation:
      Corr(p_t, p_{t-k}) > 0 for lags k = 1, 2, ..., 12+

    If you randomly shuffle rows and split 80/20:
      → The training set contains, e.g., prices from Jan 2020 and Jan 2022.
      → The test set contains prices from Jun 2021.
      → The model has SEEN future and surrounding prices during training.
      → MAPE will be ~1–2% (near-perfect) but the model is completely
         useless for actual forward-looking price prediction.
      → This mistake has been published in dozens of papers and is a
         standard trap that an ML interviewer will probe for.

    Correct approach:
      train: t = 0 ... 0.80*T        (earliest 80% of months)
      val:   t = 0.80*T ... 0.90*T   (next 10%)
      test:  t = 0.90*T ... T        (most recent 10%)

    Respects causality. Metrics reflect true out-of-sample performance.

    TimeSeriesSplit (sklearn) implements this for cross-validation with
    expanding windows: fold k trains on [0..k], tests on [k+1..k+step].
    Always past → future. Never future → past.
    """

    def __init__(self, seq_len: int = 24, year_start: int = 2000):
        self.seq_len    = seq_len
        self.year_start = year_start
        self.scalers: Dict[str, StandardScaler] = {}

    def prepare(self, price_df: pd.DataFrame,
                train_ratio: float, val_ratio: float
                ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        For each crop price series:
          1. Filter to post-year_start, forward-fill sparse months
          2. Normalize with StandardScaler (fit on train only)
          3. Build sliding-window sequences:
               X[i] = prices[i : i+seq_len]   (lookback window)
               y[i] = prices[i + seq_len]     (next month target)
          4. Split chronologically into train / val / test

        Returns
        -------
        dict[crop_name → {X_train, y_train, X_val, y_val, X_test, y_test,
                           raw_series, scaler}]
        """
        results = {}
        price_df = price_df[price_df.index.year >= self.year_start].copy()
        # Forward-fill up to 3 months (holiday/reporting gaps in commodities data)
        price_df = price_df.ffill(limit=3).dropna()

        for col in price_df.columns:
            crop = col.replace("price_", "")
            series = price_df[col].values.astype(float)

            # Fit StandardScaler on training portion only
            n_seq   = len(series) - self.seq_len
            n_train = int(n_seq * train_ratio)
            n_val   = int(n_seq * val_ratio)

            train_raw = series[: n_train + self.seq_len]
            scaler    = StandardScaler()
            scaler.fit(train_raw.reshape(-1, 1))
            self.scalers[crop] = scaler

            scaled = scaler.transform(series.reshape(-1, 1)).flatten()

            # Build supervised sequences
            X, y = self._make_sequences(scaled)
            X_tr, y_tr = X[:n_train], y[:n_train]
            X_va, y_va = X[n_train: n_train + n_val], y[n_train: n_train + n_val]
            X_te, y_te = X[n_train + n_val:], y[n_train + n_val:]

            logger.info("Price series [%s]: total=%d months, "
                        "train_seqs=%d, val_seqs=%d, test_seqs=%d",
                        crop, len(series), len(X_tr), len(X_va), len(X_te))

            results[crop] = {
                "X_train": X_tr, "y_train": y_tr,
                "X_val":   X_va, "y_val":   y_va,
                "X_test":  X_te, "y_test":  y_te,
                "raw_series": series,
                "price_index": price_df.index,
                "scaler": scaler,
            }

        return results

    def _make_sequences(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding window transformation:
          Input window:  X[i] = series[i : i+seq_len]
          Prediction target: y[i] = series[i+seq_len]

        This converts the 1D time series into a supervised learning problem
        where each sample's features are the previous seq_len observations.
        """
        X, y = [], []
        for i in range(len(series) - self.seq_len):
            X.append(series[i: i + self.seq_len])
            y.append(series[i + self.seq_len])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def inverse_transform(self, crop: str, values: np.ndarray) -> np.ndarray:
        """Undo StandardScaler normalization to get USD/MT prices."""
        return self.scalers[crop].inverse_transform(
            values.reshape(-1, 1)
        ).flatten()

    def get_last_window(self, crop: str,
                        price_df: pd.DataFrame) -> np.ndarray:
        """
        Returns the last seq_len observations of the price series,
        scaled and ready for inference. Used to forecast the next period.
        """
        series = price_df[f"price_{crop}"].values.astype(float)
        scaled = self.scalers[crop].transform(series.reshape(-1, 1)).flatten()
        return scaled[-self.seq_len:].reshape(1, -1)
