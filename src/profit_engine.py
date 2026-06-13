"""
profit_engine.py
----------------
Stage 4: Profit-Aware Recommendation Engine.

Takes outputs from all three ML modules and produces the final
financially-optimized crop recommendation.

Decision Function
-----------------
For each candidate crop c in the top-k classifier output:

  Expected_Revenue(c) = Predicted_Yield(c) × Avg_Forecast_Price(c)
  Expected_Profit(c)  = Expected_Revenue(c) − Operational_Cost(c)

where:
  Predicted_Yield(c)     ~ XGBoost regressor (Module B)  [Tonnes/Ha]
  Avg_Forecast_Price(c)  ~ mean of horizon-step forecasts (Module C) [USD/MT]
  Operational_Cost(c)    = config-specified fixed cost     [USD/Ha]

Final recommendation: argmax_c Expected_Profit(c)

Risk-Adjusted Extension (documented for completeness):
------------------------------------------------------
In production with uncertainty quantification:
  - Use XGBoost prediction intervals (quantile regression) for yield
  - Use LSTM dropout MC sampling for price distribution
  - Risk-adjusted score: E[Profit] − λ·σ[Profit]
    where λ is a farmer-specified risk aversion coefficient

Baseline Comparison
-------------------
Naive baseline: recommends the crop with the highest classifier
confidence, using static 5-year average market prices (no forecasting).
This simulates the common "soil-only" recommendation approach.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ProfitEngine:

    def __init__(self, operational_costs: Dict[str, float]):
        """
        Parameters
        ----------
        operational_costs : dict
            {crop_name: USD_per_hectare}  from config.yaml profit section.
        """
        self.costs = {k.lower(): v for k, v in operational_costs.items()}

    def compute_expected_profit(
        self,
        crop: str,
        predicted_yield: float,
        forecasted_prices: np.ndarray,
    ) -> Dict:
        """
        Computes expected profit for a single crop.

        Parameters
        ----------
        crop             : str      crop name (lowercase)
        predicted_yield  : float    XGBoost yield estimate [T/Ha]
        forecasted_prices: ndarray  Module C horizon forecasts [USD/MT]

        Returns
        -------
        dict with keys: crop, yield_t_ha, avg_price_usd, revenue, cost, profit
        """
        avg_price = float(np.mean(forecasted_prices))
        cost      = self.costs.get(crop, 0.0)
        revenue   = predicted_yield * avg_price
        profit    = revenue - cost

        return {
            "crop":          crop,
            "yield_t_ha":    round(predicted_yield, 3),
            "avg_price_usd": round(avg_price, 2),
            "revenue_usd":   round(revenue, 2),
            "cost_usd":      round(cost, 2),
            "profit_usd":    round(profit, 2),
        }

    def rank(self, profit_records: List[Dict]) -> List[Dict]:
        """Sort candidate crops by descending expected profit."""
        return sorted(profit_records, key=lambda r: r["profit_usd"], reverse=True)

    def baseline_profit(
        self,
        crop: str,
        predicted_yield: float,
        historical_avg_prices: Dict[str, float],
    ) -> Dict:
        """
        Naive baseline: uses static 5-year historical average price
        instead of a forward-looking ML forecast.

        Comparison against this baseline demonstrates the added value
        of incorporating market price forecasting into the recommendation.
        """
        avg_price = historical_avg_prices.get(crop, 0.0)
        cost      = self.costs.get(crop, 0.0)
        revenue   = predicted_yield * avg_price
        profit    = revenue - cost
        return {
            "crop":          crop,
            "yield_t_ha":    round(predicted_yield, 3),
            "avg_price_usd": round(avg_price, 2),
            "revenue_usd":   round(revenue, 2),
            "cost_usd":      round(cost, 2),
            "profit_usd":    round(profit, 2),
        }

    def recommendation_report(
        self,
        fusion_records: List[Dict],
        baseline_records: List[Dict],
    ) -> pd.DataFrame:
        """
        Builds a side-by-side comparison DataFrame for the final report.
        """
        fusion_df   = pd.DataFrame(fusion_records).add_suffix("_fusion")
        fusion_df   = fusion_df.rename(columns={"crop_fusion": "crop"})
        baseline_df = pd.DataFrame(baseline_records).add_suffix("_baseline")
        baseline_df = baseline_df.rename(columns={"crop_baseline": "crop"})

        merged = pd.merge(fusion_df, baseline_df, on="crop", how="outer")
        merged["profit_delta"] = (
            merged["profit_usd_fusion"] - merged["profit_usd_baseline"]
        )
        return merged.sort_values("profit_usd_fusion", ascending=False)
