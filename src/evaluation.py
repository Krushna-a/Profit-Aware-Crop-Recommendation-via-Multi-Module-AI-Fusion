"""
evaluation.py
-------------
Generates all evaluation artifacts: metric reports, plots, and a
JSON summary that can be version-controlled alongside model checkpoints.
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for servers/CI)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return obj
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_convert)
    logger.info("Report saved -> %s", path)


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Module A — Random Forest Confusion Matrix", fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix plot -> %s", output_path)


def plot_feature_importance(importances: np.ndarray, feature_names: list,
                             title: str, output_path: str) -> None:
    idx    = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(sorted_vals)), sorted_vals, color="steelblue")
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=30, ha="right")
    ax.set_ylabel("Importance Score")
    ax.set_title(title)
    # Annotate bars
    for bar, val in zip(bars, sorted_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Feature importance plot -> %s", output_path)


def plot_yield_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                            output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: predicted vs actual
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=10, color="steelblue")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Yield (T/Ha)")
    axes[0].set_ylabel("Predicted Yield (T/Ha)")
    axes[0].set_title("Module B — XGBoost: Predicted vs Actual Yield")
    axes[0].legend()

    # Residuals
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=40, color="coral", edgecolor="white")
    axes[1].axvline(0, color="black", lw=1.5, ls="--")
    axes[1].set_xlabel("Residual (T/Ha)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Yield prediction plot -> %s", output_path)


def plot_price_forecasts(results: Dict, price_index, plots_dir: str) -> None:
    """
    Plots actual vs predicted monthly prices for each crop on the test set.
    Also plots the forecast horizon (future months).
    """
    for crop, d in results["price_eval"].items():
        y_true = d["y_true"]
        y_pred = d["y_pred"]
        n_test = len(y_true)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(n_test), y_true, label="Actual",    color="steelblue", lw=1.5)
        ax.plot(range(n_test), y_pred, label="Predicted", color="orange",
                lw=1.5, ls="--")
        ax.set_xlabel("Month (test set, chronological)")
        ax.set_ylabel("Price (USD/MT)")
        ax.set_title(f"Module C — {crop.capitalize()} Price Forecast "
                     f"(MAPE: {d['mape']:.2f}%, Dir. Acc: {d['direction_acc']*100:.1f}%)")
        ax.legend()
        plt.tight_layout()
        path = str(Path(plots_dir) / f"price_forecast_{crop}.png")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Price forecast plot [%s] -> %s", crop, path)


def plot_profit_comparison(report_df: pd.DataFrame, output_path: str) -> None:
    """
    Bar chart comparing AI-fusion profit vs naive baseline per crop.
    """
    crops       = report_df["crop"].tolist()
    fusion_p    = report_df["profit_usd_fusion"].tolist()
    baseline_p  = report_df["profit_usd_baseline"].tolist()
    x           = np.arange(len(crops))
    width       = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - width/2, fusion_p,   width, label="AI Fusion",       color="steelblue")
    b2 = ax.bar(x + width/2, baseline_p, width, label="Naive Baseline",  color="coral")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 5,
                f"${h:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in crops])
    ax.set_ylabel("Expected Profit (USD/Ha)")
    ax.set_title("Stage 4 — AI Fusion vs Naive Baseline: Profit Comparison")
    ax.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Profit comparison plot -> %s", output_path)


def print_final_summary(results: dict) -> None:
    """Prints a structured evaluation summary to stdout / logger."""
    sep = "=" * 70

    print(f"\n{sep}")
    print("  PROFIT-AWARE CROP RECOMMENDATION — EVALUATION SUMMARY")
    print(sep)

    # Module A
    clf = results["classifier"]
    print(f"\n  MODULE A — Random Forest Classifier")
    print(f"  {'Accuracy':<28}: {clf['accuracy']:.4f}")
    print(f"  {'Weighted F1':<28}: {clf['f1_weighted']:.4f}")
    print(f"  {'OOB Score (free CV estimate)':<28}: {clf['oob_score']:.4f}")

    # Module B
    yr = results["yield_regressor"]
    print(f"\n  MODULE B — XGBoost Yield Regressor")
    print(f"  {'RMSE':<28}: {yr['rmse']:.4f} T/Ha")
    print(f"  {'MAE':<28}: {yr['mae']:.4f} T/Ha")
    print(f"  {'R²':<28}: {yr['r2']:.4f}")

    # Module C
    print(f"\n  MODULE C — Price Forecaster (chronological test set)")
    for crop, m in results["price_eval"].items():
        print(f"  {crop.capitalize():<10} MAPE: {m['mape']:6.2f}%  "
              f"RMSE: ${m['rmse_usd']:.2f}/MT  "
              f"Dir.Acc: {m['direction_acc']*100:.1f}%")

    # Stage 4
    rec = results["recommendation"]
    print(f"\n  STAGE 4 — Profit-Aware Recommendation")
    print(f"  {'Recommended Crop':<28}: {rec['recommended_crop'].upper()}")
    print(f"  {'Expected Yield':<28}: {rec['yield_t_ha']} T/Ha")
    print(f"  {'Forecast Price (avg)':<28}: ${rec['avg_price_usd']}/MT")
    print(f"  {'Expected Revenue':<28}: ${rec['revenue_usd']}/Ha")
    print(f"  {'Operational Cost':<28}: ${rec['cost_usd']}/Ha")
    print(f"  {'Expected Profit (AI Fusion)':<28}: ${rec['profit_usd']}/Ha")
    print(f"  {'Expected Profit (Baseline)':<28}: ${rec['baseline_profit_usd']}/Ha")
    delta = rec["profit_usd"] - rec["baseline_profit_usd"]
    print(f"  {'Profit Delta (Fusion − Base)':<28}: ${delta:+.2f}/Ha")
    print(f"\n  {sep}")
