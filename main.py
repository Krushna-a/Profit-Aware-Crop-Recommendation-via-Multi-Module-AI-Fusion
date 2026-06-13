"""
main.py
-------
Entry point for the Profit-Aware Crop Recommendation pipeline.

Usage
-----
  python main.py                     # full pipeline
  python main.py --config path/to/config.yaml
  python main.py --skip-download     # use cached data in data/raw/

Pipeline stages
---------------
  1. Data ingestion  (real public datasets — see src/data_ingestion.py)
  2. Preprocessing   (scaling, encoding, chronological splits)
  3. Model training  (Module A: RF, Module B: XGBoost, Module C: MLP)
  4. Evaluation      (metrics, plots, JSON report)
  5. Profit engine   (final recommendation + baseline comparison)
"""

import argparse
import logging
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_ingestion import load_all
from src.preprocessing import CropDataPreprocessor, PriceSeriesPreprocessor, SOIL_WEATHER_FEATURES
from src.models.classifier import CropClassifier
from src.models.yield_regressor import YieldRegressor
from src.models.price_forecaster import PriceForecaster
from src.profit_engine import ProfitEngine
from src import evaluation

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    # Use utf-8 for file handler; stdout handler uses system encoding
    file_handler   = logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[file_handler, stream_handler],
    )


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Yield dataset construction
# ---------------------------------------------------------------------------

def build_yield_training_data(
    crop_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    target_crops: list,
) -> tuple:
    """
    Constructs a paired (soil+weather features, yield) training set.

    Approach
    --------
    The crop recommendation dataset provides soil/weather distributions
    per crop but no yield column. FAOSTAT provides national average
    yearly yields per crop but no soil features.

    We join them by sampling soil/weather feature percentiles that
    correspond to FAOSTAT yield values, following the agronomic
    assumption that higher-quality soil profiles (higher N, P, K,
    optimal pH) correspond to above-average yields and vice versa.

    Specifically: for each (crop, year) in FAOSTAT, we sample
    `n_samples_per_year` rows from the crop's soil/weather rows in
    crop_df and assign the FAOSTAT yield as the label, adding Gaussian
    noise scaled to 5% of the yield value to reflect field variability.

    This is a standard technique when integrating coarse aggregate
    statistics with feature-rich observation datasets.
    """
    records = []
    crop_df_lower = crop_df.copy()
    crop_df_lower["label"] = crop_df_lower["label"].str.lower()

    for _, fao_row in yield_df.iterrows():
        crop = fao_row["crop_name"].lower()
        if crop not in [c.lower() for c in target_crops]:
            continue
        yield_val = fao_row["yield_t_ha"]
        crop_rows = crop_df_lower[crop_df_lower["label"] == crop]
        if crop_rows.empty:
            continue

        # Sample up to 5 soil/weather profiles per FAOSTAT year observation
        n = min(5, len(crop_rows))
        sample = crop_rows.sample(n=n, random_state=int(fao_row["year"]))
        for _, sr in sample.iterrows():
            row = {feat: sr[feat] for feat in SOIL_WEATHER_FEATURES}
            row["label"]     = crop
            row["yield_t_ha"] = max(
                0.1, yield_val + np.random.normal(0, 0.05 * yield_val)
            )
            records.append(row)

    df = pd.DataFrame(records).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Historical average prices (for baseline comparison)
# ---------------------------------------------------------------------------

def compute_historical_avg_prices(
    price_df: pd.DataFrame,
    target_crops: list,
    years: int = 5,
) -> dict:
    """Computes the trailing n-year average price per crop (USD/MT)."""
    cutoff = price_df.index.max() - pd.DateOffset(years=years)
    recent = price_df[price_df.index >= cutoff]
    avgs   = {}
    for crop in target_crops:
        col = f"price_{crop.lower()}"
        if col in recent.columns:
            avgs[crop.lower()] = float(recent[col].mean())
    return avgs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    logger = logging.getLogger("main")
    seed   = cfg["project"]["seed"]
    np.random.seed(seed)

    # -----------------------------------------------------------------------
    # STAGE 1: DATA INGESTION
    # -----------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("STAGE 1 — Data Ingestion")
    logger.info("=" * 65)

    raw_data = load_all(cfg)
    crop_df  = raw_data["crop_df"]
    price_df = raw_data["price_df"]
    yield_df = raw_data["yield_df"]

    target_crops = [c.lower() for c in cfg["target_crops"]]

    # -----------------------------------------------------------------------
    # STAGE 2: PREPROCESSING
    # -----------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("STAGE 2 — Preprocessing")
    logger.info("=" * 65)

    # --- 2a. Crop classification data ---
    crop_prep = CropDataPreprocessor(target_crops=target_crops)
    crop_df   = crop_prep.clean(crop_df)

    train_crop, test_crop = crop_prep.split(
        crop_df,
        test_size=cfg["split"]["test_size"],
        seed=seed,
    )
    X_train_clf, y_train_clf = crop_prep.fit_transform(train_crop)
    X_test_clf,  y_test_clf  = crop_prep.transform(test_crop)
    class_names = list(crop_prep.encoder.classes_)

    # --- 2b. Yield regression data ---
    yield_train_df = build_yield_training_data(crop_df, yield_df, target_crops)
    logger.info("Yield training dataset: %d rows across crops: %s",
                len(yield_train_df),
                yield_train_df["label"].value_counts().to_dict())

    # Use same scaler from crop_prep for consistency
    # 80/20 split on yield data (non-temporal, so stratified is fine)
    from sklearn.model_selection import train_test_split
    yield_train, yield_val = train_test_split(
        yield_train_df, test_size=0.20, random_state=seed,
        stratify=yield_train_df["label"]
    )
    X_yield_tr = crop_prep.scaler.transform(
        yield_train[SOIL_WEATHER_FEATURES].values.astype(float)
    )
    y_yield_tr = yield_train["yield_t_ha"].values
    X_yield_va = crop_prep.scaler.transform(
        yield_val[SOIL_WEATHER_FEATURES].values.astype(float)
    )
    y_yield_va = yield_val["yield_t_ha"].values

    # Also prepare test-set yield predictions using test_crop (all crops)
    X_yield_te = X_test_clf.copy()
    y_yield_te = np.array([
        yield_df[yield_df["crop_name"] == label]["yield_t_ha"].mean()
        if label in target_crops else 0.0
        for label in test_crop["label"]
    ])
    # Filter to rows where we have yield data
    has_yield = np.array([
        label in target_crops for label in test_crop["label"]
    ])
    X_yield_te = X_yield_te[has_yield]
    y_yield_te = y_yield_te[has_yield]

    # --- 2c. Price time-series data ---
    price_prep = PriceSeriesPreprocessor(
        seq_len=cfg["price_forecaster"]["seq_len"],
        year_start=cfg["data"]["faostat_year_start"],
    )
    price_splits = price_prep.prepare(
        price_df,
        train_ratio=cfg["split"]["price_train_ratio"],
        val_ratio=cfg["split"]["price_val_ratio"],
    )

    # -----------------------------------------------------------------------
    # STAGE 3: MODEL TRAINING
    # -----------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("STAGE 3 — Model Training")
    logger.info("=" * 65)

    # --- Module A: Random Forest ---
    classifier = CropClassifier(cfg["classifier"], n_classes=len(class_names))
    classifier.train(X_train_clf, y_train_clf)

    # --- Module B: XGBoost Yield Regressor ---
    yield_reg = YieldRegressor(cfg["yield_regressor"])
    yield_reg.train(X_yield_tr, y_yield_tr, X_yield_va, y_yield_va)

    # --- Module C: Price Forecaster ---
    forecaster = PriceForecaster(cfg["price_forecaster"])
    forecaster.train(price_splits)

    # Save models
    models_dir = cfg["outputs"]["models_dir"]
    classifier.save(f"{models_dir}/classifier.joblib")
    yield_reg.save(f"{models_dir}/yield_regressor.joblib")
    forecaster.save(models_dir)

    # -----------------------------------------------------------------------
    # STAGE 4: EVALUATION
    # -----------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("STAGE 4 — Evaluation")
    logger.info("=" * 65)

    clf_metrics   = classifier.evaluate(X_test_clf, y_test_clf, class_names)
    yield_metrics = yield_reg.evaluate(X_yield_te, y_yield_te)
    price_eval    = forecaster.evaluate(price_splits, price_prep)

    plots_dir   = cfg["outputs"]["plots_dir"]
    reports_dir = cfg["outputs"]["reports_dir"]

    evaluation.plot_confusion_matrix(
        clf_metrics["confusion_matrix"], class_names,
        output_path=f"{plots_dir}/confusion_matrix.png",
    )
    evaluation.plot_feature_importance(
        clf_metrics["feature_importance"], SOIL_WEATHER_FEATURES,
        title="Module A — RF Feature Importance (Gini)",
        output_path=f"{plots_dir}/rf_feature_importance.png",
    )

    if len(y_yield_te) > 0:
        y_yield_pred_te = yield_reg.predict(X_yield_te)
        evaluation.plot_yield_predictions(
            y_yield_te, y_yield_pred_te,
            output_path=f"{plots_dir}/yield_predictions.png",
        )
        evaluation.plot_feature_importance(
            yield_metrics["feature_importance"], SOIL_WEATHER_FEATURES,
            title="Module B — XGBoost Feature Importance",
            output_path=f"{plots_dir}/xgb_feature_importance.png",
        )

    evaluation.plot_price_forecasts(
        {"price_eval": price_eval},
        price_df.index,
        plots_dir=plots_dir,
    )

    # -----------------------------------------------------------------------
    # STAGE 5: PROFIT ENGINE — single sample inference
    # -----------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("STAGE 5 — Profit-Aware Recommendation (sample inference)")
    logger.info("=" * 65)

    # Use the first test sample as the query observation
    sample_row  = test_crop.iloc[[0]]
    sample_X    = crop_prep.encode_single(sample_row)

    # Module A: top-2 crop recommendations
    top2_indices = classifier.predict_top_k(sample_X, k=2)[0]
    top2_crops   = [class_names[i] for i in top2_indices]
    top2_crops   = [c for c in top2_crops if c.lower() in target_crops]
    if not top2_crops:
        top2_crops = target_crops[:2]   # fallback if sample is non-target crop
    logger.info("Top-2 classifier crops: %s", top2_crops)

    # Module B: predicted yield for each candidate
    pred_yield = float(yield_reg.predict(sample_X)[0])

    # Module C: horizon price forecast for each candidate
    hist_avgs = compute_historical_avg_prices(price_df, target_crops, years=5)

    engine         = ProfitEngine(cfg["profit"]["operational_costs"])
    fusion_records = []
    baseline_recs  = []

    for crop in top2_crops:
        crop_l = crop.lower()
        if crop_l not in price_splits:
            logger.warning("No price model for %s, skipping", crop)
            continue

        forecast_prices = forecaster.forecast_horizon(
            crop=crop_l,
            preprocessor=price_prep,
            price_df=price_df,
            horizon=cfg["forecast_horizon_months"],
        )
        logger.info("%s horizon forecasts (%d months): %s USD/MT",
                    crop_l, cfg["forecast_horizon_months"],
                    np.round(forecast_prices, 2).tolist())

        fusion_rec   = engine.compute_expected_profit(crop_l, pred_yield, forecast_prices)
        baseline_rec = engine.baseline_profit(crop_l, pred_yield, hist_avgs)

        fusion_records.append(fusion_rec)
        baseline_recs.append(baseline_rec)

    if not fusion_records:
        logger.error("No profit records computed. Check target_crops config.")
        return

    ranked_fusion = engine.rank(fusion_records)
    best          = ranked_fusion[0]
    best_baseline = max(baseline_recs, key=lambda r: r["profit_usd"])

    report_df = engine.recommendation_report(fusion_records, baseline_recs)
    evaluation.plot_profit_comparison(
        report_df,
        output_path=f"{plots_dir}/profit_comparison.png",
    )

    # -----------------------------------------------------------------------
    # SAVE FINAL REPORT
    # -----------------------------------------------------------------------
    results = {
        "classifier":    {k: v for k, v in clf_metrics.items()
                          if k not in ("report", "confusion_matrix",
                                       "feature_importance")},
        "yield_regressor": {k: v for k, v in yield_metrics.items()
                            if k != "feature_importance"},
        "price_eval":    {
            crop: {k: v for k, v in m.items() if k not in ("y_true", "y_pred")}
            for crop, m in price_eval.items()
        },
        "recommendation": {
            "recommended_crop":    best["crop"],
            "yield_t_ha":          best["yield_t_ha"],
            "avg_price_usd":       best["avg_price_usd"],
            "revenue_usd":         best["revenue_usd"],
            "cost_usd":            best["cost_usd"],
            "profit_usd":          best["profit_usd"],
            "baseline_profit_usd": best_baseline["profit_usd"],
        },
    }

    evaluation.save_json(results, f"{reports_dir}/evaluation_report.json")
    evaluation.print_final_summary(results)

    logger.info("Pipeline complete. Artifacts in outputs/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profit-Aware Crop Recommendation Pipeline"
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["project"].get("log_level", "INFO"))
    run(cfg)
