"""
evaluate.py: Model Evaluation for Supply-Chain CVE Forecasting
Only two charts are generated, both directly relevant to the forecasting task:

eval_lr_forecast_accuracy.png
    For each supply-chain CWE in the held-out test years, plots the Linear
    Regression model's predicted CVE count against the actual count using
    integer year labels (e.g. 2023, 2024, 2025). Each CWE is a separate line
    so the reader can see whether the model tracks individual weakness trends
    correctly or shows systematic bias for a specific category.

eval_forecast_2026.png
    The flagship deliverable chart. Plots the full historical count (2015–2025,
    one point per integer year) for each supply-chain CWE, then extends each
    line one step into 2026 with the Linear Regression forecast point. The
    forecast points are shown as filled squares (■) with a dashed connector to
    distinguish prediction from observation. This chart answers the primary
    research question: "Based on past trends, which supply-chain weakness
    categories are predicted to grow or contract in 2026?"

All confusion matrices, ROC curves, feature importance plots, and SHAP
visualisations have been removed. They do not directly address supply-chain
trend forecasting and would distract from the core narrative.
Metrics saved: results/metrics/metrics.json
"""

import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, SUPPLY_CHAIN_CATEGORIES

logger = logging.getLogger(__name__)

CWE_COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#f7b6d2",
]

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":      130,
})

def _load(name):
    with open(MODELS_DIR / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)

def _save(name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved: %s", name)

def _integer_year_axis(ax, years):
    ys = sorted(int(y) for y in set(years))
    ax.set_xticks(ys)
    ax.set_xticklabels(ys, rotation=45, ha="right")
    ax.set_xlim(ys[0] - 0.5, ys[-1] + 0.5)

def _cwe_category(cwe):
    for cat, members in SUPPLY_CHAIN_CATEGORIES.items():
        if cwe in members:
            return cat
    return "Other"

def plot_lr_accuracy(lr, X_te, test_df, y_vol_test):
    """
    Actual vs predicted CVE count for supply-chain CWEs in test years.

    Each line represents one CWE. The x-axis shows integer years (e.g. 2023,
    2024, 2025). Solid lines = actual counts; dashed lines = model predictions.
    Close tracking between solid and dashed indicates good generalisation.
    Systematic gaps (model always over- or under-predicting) point to CWEs
    where the trend changed after the training cutoff.
    """
    y_pred_log = lr.predict(X_te)
    y_pred_cnt = np.expm1(y_pred_log).clip(0)
    y_true_cnt = np.expm1(y_vol_test)

    rmse = float(np.sqrt(mean_squared_error(y_true_cnt, y_pred_cnt)))
    mae  = float(mean_absolute_error(y_true_cnt, y_pred_cnt))
    r2   = float(r2_score(y_vol_test, y_pred_log))

    df = test_df.copy()
    df["year"]       = df["year"].astype(int)
    df["actual"]     = y_true_cnt
    df["predicted"]  = y_pred_cnt
    df["cwe"]        = df["cwe"].astype(str)

    cwes  = df.groupby("cwe")["actual"].mean().nlargest(10).index.tolist()
    years = sorted(df["year"].unique())

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, cwe in enumerate(cwes):
        sub = df[df["cwe"] == cwe].sort_values("year")
        col = CWE_COLORS[i % len(CWE_COLORS)]
        ax.plot(sub["year"], sub["actual"],   "o-", color=col, lw=2, ms=5,
                label=f"{cwe} actual")
        ax.plot(sub["year"], sub["predicted"],"s--",color=col, lw=1.5, ms=4,
                alpha=0.7, label=f"{cwe} predicted")

    ax.set_xlabel("Year")
    ax.set_ylabel("CVE count")
    ax.set_title(
        f"Linear Regression: Actual vs Predicted Supply-Chain CVE Counts (Test Period)\n"
        f"Solid = actual  |  Dashed = model forecast  |  RMSE={rmse:.1f} CVEs  R²(log)={r2:.3f}\n"
        "Close tracking validates that past trend features generalise to unseen years"
    )
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    _integer_year_axis(ax, years)
    plt.tight_layout()
    _save("eval_lr_forecast_accuracy.png")
    return {"rmse_count": round(rmse,2), "mae_count": round(mae,2), "r2_log": round(r2,4)}

def plot_forecast_2026(lr, feat_cols, full_df):
    """
    Historical trend (integer years 2015–2025) + 2026 forecast per supply-chain CWE.

    This is the primary output chart. It reads:
      ● Filled circles connected by solid lines = observed counts (2015 onward)
      ■ Filled squares connected by dashed lines = 2026 model prediction

    How to interpret
    ----------------
    A CWE whose historical line slopes upward and whose 2026 forecast square sits
    above the 2025 point is predicted to *grow* — it should receive elevated
    attention in security audits and SBOM reviews.

    A CWE whose 2026 square sits *below* the 2025 point is predicted to *contract*,
    suggesting current remediation efforts may be working.

    The model uses three years of lagged counts, growth rate, acceleration, and
    severity mix as inputs. It does NOT extrapolate blindly; a CWE that spiked
    in one year but showed deceleration in the following two will receive a
    conservative (lower) forecast.
    """
    from src.predict import _build_forecast_rows

    full_df = full_df.copy()
    full_df["year"] = full_df["year"].astype(int)

    forecast_df = _build_forecast_rows(full_df, feat_cols)
    if forecast_df.empty:
        logger.warning("No forecast rows available.")
        return {}

    X_fc       = forecast_df[feat_cols].fillna(0).values
    pred_log   = lr.predict(X_fc)
    pred_count = np.expm1(pred_log).clip(0)
    forecast_df = forecast_df.copy()
    forecast_df["predicted_2026"] = pred_count.astype(int)

    cwes  = forecast_df.nlargest(12, "predicted_2026")["cwe"].tolist()
    all_years = sorted(full_df["year"].unique())
    hist_years = [int(y) for y in all_years]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, cwe in enumerate(cwes):
        col = CWE_COLORS[i % len(CWE_COLORS)]
        hist = (full_df[full_df["cwe"] == cwe]
                .drop_duplicates("year")
                .sort_values("year")
                .set_index("year")["count_t"]
                .reindex(hist_years, fill_value=None))

        fc_row = forecast_df[forecast_df["cwe"] == cwe]
        fc_val = int(fc_row["predicted_2026"].iloc[0]) if not fc_row.empty else None
        last_hist_yr = int(full_df[full_df["cwe"] == cwe]["year"].max())
        last_hist_val = full_df[
            (full_df["cwe"] == cwe) & (full_df["year"] == last_hist_yr)
        ]["count_t"].values
        last_val = float(last_hist_val[0]) if len(last_hist_val) > 0 else None

        ax.plot(hist_years, hist.values, "o-", color=col, lw=2, ms=5, label=cwe)

        if fc_val is not None and last_val is not None:
            ax.plot([last_hist_yr, 2026], [last_val, fc_val],
                    "s--", color=col, lw=1.5, ms=7)

    forecast_year = 2026
    ax.axvline(x=forecast_year - 0.5, color="gray", linestyle=":", lw=1.5, alpha=0.6)
    ax.text(forecast_year - 0.45, ax.get_ylim()[1] * 0.97,
            "← observed   predicted →", fontsize=8, color="gray", va="top")

    all_plot_years = hist_years + [2026]
    _integer_year_axis(ax, all_plot_years)

    ax.set_xlabel("Year")
    ax.set_ylabel("CVE count")
    ax.set_title(
        "Supply-Chain CVE Forecast: Historical Trend (2015–2025) + 2026 Prediction\n"
        "● Solid circles = observed counts  |  ■ Dashed squares = Linear Regression forecast\n"
        "Upward-pointing squares identify CWEs predicted to grow; downward = predicted decline"
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save("eval_forecast_2026.png")

    return {cwe: int(forecast_df[forecast_df["cwe"]==cwe]["predicted_2026"].iloc[0])
            for cwe in cwes if not forecast_df[forecast_df["cwe"]==cwe].empty}

def evaluate() -> dict:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    lr        = _load("linear_regression")
    log_reg   = _load("logistic_regression")
    rf        = _load("random_forest")
    feat_cols = _load("feature_cols")
    td        = _load("test_data")

    X_te      = td["X_test"]
    y_vol     = td["y_vol_test"]
    y_emer    = td["y_emer_test"]
    y_risk    = td["y_risk_test"].astype(int)
    test_df   = td["test_df"].copy()
    test_df["year"] = test_df["year"].astype(int)

    results = {}

    # LR metrics + accuracy chart
    lr_metrics = plot_lr_accuracy(lr, X_te, test_df, y_vol)
    results["linear_regression"] = {
        **lr_metrics,
        "task": "Next-year supply-chain CVE count forecast per CWE",
    }
    logger.info("LinearRegression  RMSE=%.1f CVEs  R²(log)=%.4f",
                lr_metrics["rmse_count"], lr_metrics["r2_log"])

    # LogReg — metrics only, no chart
    from sklearn.metrics import roc_auc_score, classification_report
    if len(np.unique(y_emer)) > 1:
        prob_emer = log_reg.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_emer, prob_emer))
        rep = classification_report(y_emer, log_reg.predict(X_te),
                                    target_names=["Stable","Emerging"],
                                    output_dict=True)
        results["logistic_regression"] = {
            "auc_roc":     round(auc, 4),
            "f1_emerging": round(rep.get("Emerging",{}).get("f1-score",0), 4),
            "accuracy":    round(rep["accuracy"], 4),
            "task": "Supply-chain CWE surge detection",
        }
        logger.info("LogisticRegression  AUC=%.4f", auc)

    # RF, metrics only, no chart
    from sklearn.metrics import accuracy_score
    y_pred_risk = rf.predict(X_te).astype(int)
    acc = float(accuracy_score(y_risk, y_pred_risk))
    results["random_forest"] = {
        "accuracy": round(acc, 4),
        "task": "Risk tier classification (LOW/MEDIUM/HIGH/CRITICAL)",
    }
    logger.info("RandomForest  accuracy=%.4f", acc)

    # Load full dataset for 2026 forecast chart
    try:
        full_df = pd.read_csv("data/processed/temporal_features.csv")
        full_df["year"] = full_df["year"].astype(int)
        forecasts = plot_forecast_2026(lr, feat_cols, full_df)
        results["forecast_2026"] = forecasts
    except Exception as e:
        logger.warning("Could not generate 2026 forecast chart: %s", e)

    with open(METRICS_DIR / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Metrics saved.")
    return results
