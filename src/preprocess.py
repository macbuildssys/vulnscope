"""
preprocess.py: Supply-Chain CVE Temporal Feature Engineering

Unit of analysis: one row per (year, CWE) pair, restricted to the supply-chain
CWE set defined in config.py.

Features built from three years of lagged history:
  count_t, count_lag1..3   — CVE counts in current and prior years
  growth_1y, growth_3y     — year-over-year and 3-year growth rates
  accel                    — change in growth rate (second derivative)
  volatility               — std-dev of count over 3 prior years
  momentum                 — count × growth_1y (size-adjusted trend signal)
  avg_score                — mean CVSS base score for this (year, CWE)
  pct_critical / pct_high  — fraction at each severity level
  pct_network, pct_no_auth — fraction that are network-exploitable / need no auth
  year_norm                — year − min_year (keeps the feature numeric but small)
  sc_category_*            — one-hot indicator for supply-chain category

Targets:
  log_count_next  — for Linear Regression (volume forecast)
  is_emerging     — for Logistic Regression (count_next > 1.5 × count_t)
  risk_tier       — for Random Forest (LOW=0 / MEDIUM=1 / HIGH=2 / CRITICAL=3)
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import (
    PROCESSED_DATA_DIR, RAW_DATA_DIR,
    SUPPLY_CHAIN_CWES, SUPPLY_CHAIN_CATEGORIES,
)

logger = logging.getLogger(__name__)

LAG_YEARS     = 3
MIN_CVE_COUNT = 5       # lower threshold so less-common SC CWEs are included
SURGE_FACTOR  = 1.5

def _cwe_category(cwe: str) -> str:
    for cat, members in SUPPLY_CHAIN_CATEGORIES.items():
        if cwe in members:
            return cat
    return "Other"

def _build_lag_features(agg: pd.DataFrame) -> pd.DataFrame:
    lookup = {(int(r.year), r.cwe): r for r in agg.itertuples()}
    min_year = int(agg["year"].min())
    rows = []

    for (year, cwe), curr in lookup.items():
        lags = [lookup.get((year - l, cwe)) for l in range(1, LAG_YEARS + 1)]
        if any(l is None for l in lags):
            continue
        nxt = lookup.get((year + 1, cwe))
        if nxt is None:
            continue

        c_t  = float(curr.count)
        c_l1, c_l2, c_l3 = (float(l.count) for l in lags)
        c_n  = float(nxt.count)

        growth_1y = (c_t - c_l1) / max(c_l1, 1)
        growth_3y = (c_t - c_l3) / max(c_l3, 1)
        accel     = growth_1y - ((c_l1 - c_l2) / max(c_l2, 1))
        vol_3y    = float(np.std([c_l3, c_l2, c_l1]))
        mom       = c_t * growth_1y

        row = {
            "year":         int(year),
            "cwe":          cwe,
            "sc_category":  _cwe_category(cwe),
            "count_t":      c_t,
            "count_lag1":   c_l1,
            "count_lag2":   c_l2,
            "count_lag3":   c_l3,
            "count_next":   c_n,
            "avg_score":    float(curr.avg_score),
            "pct_critical": float(curr.pct_critical),
            "pct_high":     float(curr.pct_high),
            "pct_network":  float(curr.pct_network),
            "pct_no_auth":  float(curr.pct_no_auth),
            "growth_1y":    growth_1y,
            "growth_3y":    growth_3y,
            "accel":        accel,
            "volatility":   vol_3y,
            "momentum":     mom,
            "year_norm":    year - min_year,
            "log_count_next": float(np.log1p(c_n)),
            "is_emerging":  int(c_n > SURGE_FACTOR * c_t),
        }
        # One-hot supply-chain category
        for cat in SUPPLY_CHAIN_CATEGORIES:
            row[f"cat_{cat.replace(' ', '_').replace('/', '_')}"] = int(_cwe_category(cwe) == cat)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "No rows after lag construction. "
            "Your database may not have enough consecutive years for the selected CWEs."
        )

    composite = df["count_next"] * df["avg_score"]
    df["risk_tier"] = pd.qcut(
        composite, q=4, labels=[0, 1, 2, 3], duplicates="drop"
    ).astype(int)

    logger.info(
        "Supply-chain panel: %d rows, %d CWEs, years %d–%d",
        len(df), df["cwe"].nunique(), df["year"].min(), df["year"].max(),
    )
    return df

def _aggregate(raw_df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate a flat CVE DataFrame into (year, cwe) statistics
    agg = (
        raw_df.groupby(["year", "cwe"])
        .agg(
            count       =("base_score", "count"),
            avg_score   =("base_score", "mean"),
            pct_critical=("severity",   lambda x: (x == "CRITICAL").mean()),
            pct_high    =("severity",   lambda x: (x == "HIGH").mean()),
            pct_network =("av",         lambda x: (x == "NETWORK").mean()),
            pct_no_auth =("pr",         lambda x: (x == "NONE").mean()),
        )
        .reset_index()
    )
    agg = agg[agg["count"] >= MIN_CVE_COUNT].reset_index(drop=True)
    agg["avg_score"]    = agg["avg_score"].round(3)
    agg["pct_critical"] = agg["pct_critical"].round(3)
    agg["pct_high"]     = agg["pct_high"].round(3)
    agg["pct_network"]  = agg["pct_network"].round(3)
    agg["pct_no_auth"]  = agg["pct_no_auth"].round(3)
    return agg

def preprocess_from_db(out_path: Optional[Path] = None) -> pd.DataFrame:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or PROCESSED_DATA_DIR / "temporal_features.csv"

    from src.db import get_connection, DB_PATH
    conn = get_connection()
    raw = pd.read_sql_query(
        f"""
        SELECT year, cwe, base_score,
               base_severity AS severity,
               attack_vector AS av,
               privileges_required AS pr
        FROM cves
        WHERE year IS NOT NULL
          AND cwe IN ({','.join('?' for _ in SUPPLY_CHAIN_CWES)})
          AND base_score IS NOT NULL
        """,
        conn,
        params=list(SUPPLY_CHAIN_CWES),
    )
    conn.close()

    if raw.empty:
        raise ValueError(
            "No supply-chain CVEs found in the database. "
            "Run 'python main.py fetch' first, or try 'python main.py demo'."
        )

    raw["year"] = raw["year"].astype(int)
    agg = _aggregate(raw)
    df  = _build_lag_features(agg)
    df.to_csv(out_path, index=False)
    logger.info("Saved: %s (%d rows × %d cols)", out_path, *df.shape)
    return df

def preprocess_from_json(raw_path: Optional[Path] = None,
                         out_path:  Optional[Path] = None) -> pd.DataFrame:
    """Demo mode: build from the raw JSON file."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = raw_path or RAW_DATA_DIR / "cves.json"
    out_path = out_path or PROCESSED_DATA_DIR / "temporal_features.csv"

    with open(raw_path) as fh:
        records = json.load(fh)

    rows = []
    for rec in records:
        cve = rec.get("cve", {})
        pub  = cve.get("published", "")
        year = int(pub[:4]) if pub and len(pub) >= 4 else None
        if year is None:
            continue

        cwe = "CWE-OTHER"
        for w in cve.get("weaknesses", []):
            for d in w.get("description", []):
                v = d.get("value", "")
                if v.upper().startswith("CWE-") and v.upper() != "CWE-NOINFO":
                    cwe = v.upper(); break

        if cwe not in SUPPLY_CHAIN_CWES:
            continue

        metrics = cve.get("metrics", {})
        entry   = None
        for key in ("cvssMetricV31", "cvssMetricV30"):
            if metrics.get(key):
                entry = metrics[key][0]; break
        if entry is None:
            continue

        data = entry.get("cvssData", {})
        bs   = data.get("baseScore")
        if bs is None:
            continue

        rows.append({
            "year":     int(year),
            "cwe":      cwe,
            "base_score": float(bs),
            "severity": data.get("baseSeverity", ""),
            "av":       data.get("attackVector", ""),
            "pr":       data.get("privilegesRequired", ""),
        })

    if not rows:
        raise ValueError("No supply-chain CVEs found in JSON demo data.")

    raw_df = pd.DataFrame(rows)
    agg    = _aggregate(raw_df)
    df     = _build_lag_features(agg)
    df.to_csv(out_path, index=False)
    logger.info("Saved (demo): %s (%d rows × %d cols)", out_path, *df.shape)
    return df

preprocess = preprocess_from_db
