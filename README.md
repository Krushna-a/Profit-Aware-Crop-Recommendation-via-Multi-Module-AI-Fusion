# Profit-Aware Crop Recommendation via Multi-Module AI Fusion

**Amazon ML Summer School Portfolio Project**

A production-grade machine learning pipeline that recommends the most financially optimal crop to grow, combining three ML modules trained on real public datasets: soil classification, yield regression, and market price forecasting.

---

## Problem Statement

Existing crop recommendation systems tell farmers *what to grow* based on soil and climate conditions — but ignore *what it will sell for*. A soil-optimal crop recommended at harvest time during a price trough destroys profit.

This project fuses three ML modules to answer: **what crop maximizes expected profit given today's soil conditions and the forecasted market price at harvest?**

---

## Data Sources

| Dataset | Source | Records | Use |
|---|---|---|---|
| Crop Recommendation | [Kaggle / ICAR](https://www.kaggle.com/datasets/varshitanalluri/crop-recommendation-dataset) | 2,200 rows, 22 crops | Classifier training |
| Commodity Prices | [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) | Monthly, 1960–present | Price forecasting |
| Crop Yields | [FAOSTAT QCL Domain](https://www.fao.org/faostat) | India, 2000–2023 | Yield regression |

All datasets are publicly available and downloaded automatically by `src/data_ingestion.py`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT: Soil + Climate                       │
│         N, P, K, pH, Temperature, Humidity, Rainfall          │
└───────────────────┬──────────────────────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │  Module A           │  Random Forest Classifier
          │  Crop Classifier    │  → Top-K crop candidates
          └─────────┬──────────┘  (bagging, m=√p features)
                    │
          ┌─────────▼──────────┐
          │  Module B           │  XGBoost Regressor
          │  Yield Predictor    │  → Predicted Yield (T/Ha)
          └─────────┬──────────┘  (gradient boosting, L2 reg)
                    │
          ┌─────────▼──────────┐
          │  Module C           │  MLP over sliding windows
          │  Price Forecaster   │  → Forecast Price (USD/MT)
          └─────────┬──────────┘  (chronological split enforced)
                    │
          ┌─────────▼──────────┐
          │  Profit Engine      │  Profit = Yield × Price − Cost
          │  Decision Layer     │  → Final Recommendation
          └────────────────────┘
```

---

## Mathematical Foundations

### Module A — Random Forest (Bagging)

Given B trees trained on bootstrap samples with m = √p random features per split:

```
Var(f_bag) = ρ·σ² + (1−ρ)/B · σ²
```

Feature randomization drives pairwise tree correlation ρ → 0. As B → ∞, variance → 0 while bias is unchanged. This is why RF outperforms single decision trees.

### Module B — XGBoost (Gradient Boosting)

At round m, the objective is minimized via second-order Taylor expansion:

```
L(m) ≈ Σᵢ [gᵢ·f_m(xᵢ) + ½·hᵢ·f_m(xᵢ)²] + Ω(f_m)
```

Optimal leaf weight: `w*_j = −(Σgᵢ) / (Σhᵢ + λ)` where λ is L2 regularization. The Hessian hᵢ enables adaptive per-sample step sizes.

### Module C — LSTM Reference (Production)

The LSTM Constant Error Carousel solves the vanishing gradient problem in vanilla RNNs:

```
∂c_t/∂c_{t-1} = f_t  (forget gate, learnable, stays near 1.0)
```

Gradient flows back additively through cell state — not multiplicatively — preventing exponential decay over long sequences.

**Data leakage note:** Random train/test splits on time-series data allow the model to see future prices during training (look-ahead bias), producing MAPE ≈ 1–2% with zero real predictive power. This project enforces strict chronological splits.

### Profit Decision Function

```
Expected_Profit(c) = Predicted_Yield(c) × Avg_Forecast_Price(c) − Operational_Cost(c)
```

---

## Project Structure

```
├── config/
│   └── config.yaml          # All hyperparameters and data source URLs
├── data/
│   └── raw/                 # Auto-downloaded, gitignored
├── src/
│   ├── data_ingestion.py    # Downloads Kaggle/World Bank/FAOSTAT data
│   ├── preprocessing.py     # Scaling, encoding, chronological splits
│   └── models/
│       ├── classifier.py    # Module A: Random Forest
│       ├── yield_regressor.py  # Module B: XGBoost
│       └── price_forecaster.py # Module C: MLP / LSTM reference
├── src/profit_engine.py     # Stage 4: profit computation + ranking
├── src/evaluation.py        # Metrics, plots, JSON reports
├── main.py                  # Pipeline entry point
├── requirements.txt
└── outputs/                 # Models, reports, plots (auto-created)
```

---

## Setup & Run

```bash
# 1. Create environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# Optional: use a different config
python main.py --config config/config.yaml
```

The pipeline downloads all data automatically on first run and caches it in `data/raw/`. Subsequent runs use the cache.

---

## Outputs

After running, the following artifacts are generated:

```
outputs/
├── models/
│   ├── classifier.joblib
│   ├── yield_regressor.joblib
│   └── price_forecaster_wheat.joblib  (per crop)
├── reports/
│   └── evaluation_report.json
└── plots/
    ├── confusion_matrix.png
    ├── rf_feature_importance.png
    ├── xgb_feature_importance.png
    ├── yield_predictions.png
    ├── price_forecast_wheat.png
    ├── price_forecast_rice.png
    ├── price_forecast_cotton.png
    └── profit_comparison.png
```

---

## Evaluation Metrics

| Module | Metric | Rationale |
|---|---|---|
| A — Classifier | Accuracy, Weighted F1 | F1 handles class imbalance across 22 crops |
| A — Classifier | OOB Score | Free bootstrap cross-validation estimate |
| B — Regressor | RMSE, MAE, R² | RMSE penalizes large yield errors; MAE is interpretable in T/Ha |
| C — Forecaster | MAPE, Direction Accuracy | MAPE is scale-invariant; direction acc. is decision-relevant |
| Stage 4 | Profit delta vs baseline | Financial value-add over naive soil-only recommendation |

---

## Key Design Decisions

**Why three modules instead of one end-to-end model?**
Interpretability. Each module has a clear input/output contract that a farmer or agronomist can validate independently. End-to-end models obscure where errors originate.

**Why XGBoost for yield instead of a neural network?**
Tabular data with ~300 training samples (FAOSTAT years × crops) is exactly the regime where gradient boosted trees outperform neural networks. XGBoost's L2-regularized leaf weights prevent overfitting on small datasets.

**Why chronological split for prices?**
Commodity prices are autocorrelated. Random shuffling creates look-ahead bias — the model memorizes future prices rather than learning temporal patterns. See `src/preprocessing.py` for a detailed explanation.

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8).
- Friedman, J. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*.
- FAO (2024). FAOSTAT Crops and Livestock Products Database. fao.org/faostat
- World Bank (2024). Commodity Markets — Pink Sheet. worldbank.org
