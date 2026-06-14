"""
predict.py
Offline threat forecasting engine.

Uses all three trained models to forecast vulnerability risks for a future year.
No network access required after training.

Output
------
For each tracked CWE, combines:
  - Linear Regression predicted count (volume forecast)
  - Logistic Regression surge probability (emerging threat signal)
  - Random Forest risk tier (multi-dimensional risk assessment)

Then ranks CWEs into:
  - Top emerging threats (high surge probability)
  - Persistent high-volume threats (high predicted count, stable)
  - Declining threats (negative growth trend)
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

RISK_TIER_LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}

def _load(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python main.py build' first.")
    with open(path, "rb") as fh:
        return pickle.load(fh)

def _build_forecast_rows(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    For each CWE, use the most recent available year's data
    to build a feature vector for forecasting the *next* year
    """
    max_year = int(df["year"].max())
    rows = []
    for cwe in df["cwe"].unique():
        sub = df[df["cwe"] == cwe].sort_values("year")
        if sub.empty:
            continue
        last = sub.iloc[-1]
        if int(last["year"]) < max_year - 1:
            continue  # skip CWEs with stale data

        row = last[feature_cols].to_dict()
        # Advance the window: current t becomes lag1, lag1 becomes lag2, etc.
        row["count_lag3"] = row.get("count_lag2", row["count_t"])
        row["count_lag2"] = row.get("count_lag1", row["count_t"])
        row["count_lag1"] = row["count_t"]
        row["count_t"]    = row["count_t"]  # unchanged (best estimate for next year)
        row["year_norm"]  = row["year_norm"] + 1

        rows.append({"cwe": cwe, "year_data": int(last["year"]), **row})

    return pd.DataFrame(rows)

def predict_threats(forecast_year: Optional[int] = None) -> dict:
    """
    Forecast emerging and recurring threats for *forecast_year*.
    If not specified, uses max(year in training data) + 1.
    """
    lr      = _load("linear_regression")
    log_reg = _load("logistic_regression")
    rf      = _load("random_forest")
    feat_cols = _load("feature_cols")

    df = pd.read_csv(PROCESSED_DATA_DIR / "temporal_features.csv")
    if forecast_year is None:
        forecast_year = int(df["year"].max()) + 1

    forecast_df = _build_forecast_rows(df, feat_cols)
    if forecast_df.empty:
        logger.error("No CWEs available for forecasting. Check your data.")
        return {}

    X = forecast_df[feat_cols].fillna(0).values

    log_count_pred  = lr.predict(X)
    count_pred      = np.expm1(log_count_pred).clip(0)   # reverse log1p
    surge_prob      = log_reg.predict_proba(X)[:, 1]
    risk_tier_pred  = rf.predict(X).astype(int)
    risk_tier_prob  = rf.predict_proba(X)

    results = []
    for i, row in forecast_df.iterrows():
        idx = i - forecast_df.index[0]
        results.append({
            "cwe":             row["cwe"],
            "data_from_year":  int(row["year_data"]),
            "predicted_count": int(round(count_pred[idx])),
            "surge_probability": round(float(surge_prob[idx]), 3),
            "risk_tier":       RISK_TIER_LABEL.get(risk_tier_pred[idx], "UNKNOWN"),
            "risk_tier_probs": {
                RISK_TIER_LABEL[j]: round(float(p), 3)
                for j, p in enumerate(risk_tier_prob[idx])
            },
            "current_count":   int(row["count_t"]),
            "growth_1y":       round(float(row["growth_1y"]), 3),
            "avg_score":       round(float(row["avg_score"]), 2),
        })

    results_df = pd.DataFrame(results)

    # Rank categories
    emerging   = (
        results_df[results_df["surge_probability"] >= 0.5]
        .sort_values(["surge_probability", "predicted_count"], ascending=False)
        .head(10)
    )
    persistent = (
        results_df[results_df["surge_probability"] < 0.5]
        .sort_values("predicted_count", ascending=False)
        .head(10)
    )
    declining  = (
        results_df[results_df["growth_1y"] < -0.1]
        .sort_values("growth_1y")
        .head(5)
    )

    output = {
        "forecast_year":        forecast_year,
        "cwe_categories_scored": len(results),
        "top_emerging_threats": emerging.to_dict(orient="records"),
        "persistent_high_volume": persistent.to_dict(orient="records"),
        "declining_threats":    declining.to_dict(orient="records"),
        "summary": {
            "critical_risk_cwes":  int((results_df["risk_tier"] == "CRITICAL").sum()),
            "high_risk_cwes":      int((results_df["risk_tier"] == "HIGH").sum()),
            "surge_candidates":    int((results_df["surge_probability"] >= 0.5).sum()),
        },
    }

    return output
