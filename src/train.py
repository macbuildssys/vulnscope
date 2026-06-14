"""
train.py
Train three temporal forecasting models on the (year, CWE) panel dataset.

LinearRegression:   forecast next year's CVE volume per CWE category
LogisticRegression: detect CWEs about to surge (count_next > 1.5× current)
RandomForest:       classify overall risk tier for next year (4-class)

Temporal split — no leakage
-----------------------------
Train on older years, test on recent years. This mirrors production use
where the model is trained on historical data and applied to future periods.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "count_t", "count_lag1", "count_lag2", "count_lag3",
    "avg_score", "pct_critical", "pct_high", "pct_network", "pct_no_auth",
    "growth_1y", "growth_3y", "accel", "volatility", "momentum", "year_norm",
]
TEMPORAL_CUTOFF = 2023   # train on < 2023, test on >= 2023

def _save(obj, name: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / f"{name}.pkl", "wb") as fh:
        pickle.dump(obj, fh)
    logger.info("Saved %s", name)

def load_processed(path: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_csv(path or PROCESSED_DATA_DIR / "temporal_features.csv")

def train(df: Optional[pd.DataFrame] = None) -> dict:
    if df is None:
        df = load_processed()

    X = df[FEATURE_COLS].fillna(0).values
    y_vol  = df["log_count_next"].values          # LR target: log(count_next + 1)
    y_emer = df["is_emerging"].values             # LogReg target: surge binary
    y_risk = df["risk_tier"].values.astype(int)   # RF target: 0-3 risk class

    # Temporal split
    test_mask = (df["year"] >= TEMPORAL_CUTOFF).values
    n_test = test_mask.sum()
    if n_test < 5:
        logger.warning(
            "Only %d test rows from %d+. Falling back to 80/20 split.", n_test, TEMPORAL_CUTOFF
        )
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(X))
        tr_i, te_i = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    else:
        tr_i = np.where(~test_mask)[0]
        te_i = np.where(test_mask)[0]

    X_tr, X_te = X[tr_i], X[te_i]
    y_vol_tr,  y_vol_te  = y_vol[tr_i],  y_vol[te_i]
    y_emer_tr, y_emer_te = y_emer[tr_i], y_emer[te_i]
    y_risk_tr, y_risk_te = y_risk[tr_i], y_risk[te_i]

    logger.info(
        "Temporal split: %d train rows (before %d), %d test rows",
        len(tr_i), TEMPORAL_CUTOFF, len(te_i),
    )

    skf = StratifiedKFold(n_splits=min(5, max(2, len(np.unique(y_risk_tr)))),
                          shuffle=True, random_state=RANDOM_STATE)

    # 1. Linear Regression — volume forecasting
    logger.info("Training LinearRegression (CVE volume forecasting) ...")
    lr_pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    lr_pipe.fit(X_tr, y_vol_tr)
    lr_cv = cross_val_score(lr_pipe, X_tr, y_vol_tr, cv=5, scoring="r2")
    logger.info("  CV R2=%.4f±%.4f", lr_cv.mean(), lr_cv.std())

    # 2. Logistic Regression — surge / emerging detection
    logger.info("Training LogisticRegression (emerging threat detection) ...")
    log_pipe = Pipeline([
        ("sc", StandardScaler()),
        ("m", LogisticRegression(
            max_iter=2000, C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])
    n_classes_emer = len(np.unique(y_emer_tr))
    if n_classes_emer < 2:
        logger.warning(
            "Training data contains only one class for is_emerging. "
            "Logistic Regression will be trained but CV metrics are not available. "
            "This is expected with small or synthetic datasets where no CWE surges."
        )
        log_pipe.fit(X_tr, y_emer_tr)
        log_cv_f1 = np.array([float("nan")])
        log_cv_au = np.array([float("nan")])
    else:
        log_pipe.fit(X_tr, y_emer_tr)
        log_cv_f1 = cross_val_score(log_pipe, X_tr, y_emer_tr, cv=skf, scoring="f1")
        log_cv_au = cross_val_score(log_pipe, X_tr, y_emer_tr, cv=skf, scoring="roc_auc")
        logger.info("  CV F1=%.4f±%.4f  AUC=%.4f±%.4f",
                    log_cv_f1.mean(), log_cv_f1.std(), log_cv_au.mean(), log_cv_au.std())

    # 3. Random Forest — risk tier classification
    logger.info("Training RandomForest (risk tier classification) ...")
    rf_pipe = Pipeline([
        ("m", RandomForestClassifier(
            n_estimators=300, max_depth=10,
            min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])
    rf_pipe.fit(X_tr, y_risk_tr)
    rf_cv = cross_val_score(rf_pipe, X_tr, y_risk_tr, cv=skf, scoring="f1_weighted")
    logger.info("  CV F1_weighted=%.4f±%.4f", rf_cv.mean(), rf_cv.std())

    test_data = {
        "X_test": X_te,
        "y_vol_test":  y_vol_te,
        "y_emer_test": y_emer_te,
        "y_risk_test": y_risk_te,
        "test_df": df.iloc[te_i].reset_index(drop=True),
    }

    _save(lr_pipe,   "linear_regression")
    _save(log_pipe,  "logistic_regression")
    _save(rf_pipe,   "random_forest")
    _save(test_data, "test_data")
    _save(FEATURE_COLS, "feature_cols")

    cv_summary = {
        "linear_regression":   {"cv_r2_mean": float(lr_cv.mean()), "task": "CVE volume forecast (log count)"},
        "logistic_regression": {"cv_f1_mean": float(log_cv_f1.mean()), "cv_auc_mean": float(log_cv_au.mean()), "task": "emerging threat detection"},
        "random_forest":       {"cv_f1_weighted": float(rf_cv.mean()), "task": "risk tier classification (4-class)"},
    }
    return {
        "linear_regression": lr_pipe,
        "logistic_regression": log_pipe,
        "random_forest": rf_pipe,
        "test_data": test_data,
        "feature_cols": FEATURE_COLS,
        "cv_summary": cv_summary,
    }
