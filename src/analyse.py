"""
analyse.py: Supply-Chain CVE Trend Analysis

Generates three focused charts (all with yearly integer x-axis from 2015 to
the most recent year in the database) plus a text report.

Charts produced
---------------
trend_sc_volume_by_year.png
    Total supply-chain CVE count per year. Reveals whether the aggregate attack
    surface is growing, plateauing, or contracting over time.

trend_sc_cwe_per_year.png
    Year-by-year count for each tracked supply-chain CWE. Shows which weakness
    categories are accelerating, which are stable, and which are declining —
    the core recurring-vulnerability signal.

trend_sc_severity_by_year.png
    Stacked bar of CRITICAL / HIGH / MEDIUM / LOW severity counts per year.
    Captures whether the supply-chain threat landscape is becoming more or less
    severe independent of volume changes.

trend_sc_category_share.png
    Stacked area chart of the five supply-chain categories (Memory Safety,
    Code Execution, Injection, Parser/Protocol Bugs, Access Control) per year.
    Reveals structural shifts in what *type* of supply-chain weakness dominates.

results/analysis_report.txt
    Plain-text summary of recurring CVEs, fastest-growing CWEs, declining CWEs,
    and recent severity / exploitability statistics.
"""

import logging
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR, RESULTS_DIR,
    SUPPLY_CHAIN_CWES, SUPPLY_CHAIN_CATEGORIES,
)

logger     = logging.getLogger(__name__)
REPORT_PATH = RESULTS_DIR / "analysis_report.txt"

# Palette: one colour per supply-chain category
CAT_COLORS = {
    "Memory Safety":          "#1f77b4",
    "Code Execution":         "#d62728",
    "Injection":              "#ff7f0e",
    "Parser / Protocol Bugs": "#2ca02c",
    "Access Control":         "#9467bd",
    "Other":                  "#7f7f7f",
}

CWE_COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#f7b6d2",
]

plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "figure.dpi":          130,
    "axes.titlesize":      13,
    "axes.labelsize":      11,
})

def _save(name: str, fig=None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    (fig or plt).savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved: %s", path)

def _integer_year_axis(ax, years):
    # Force the x-axis to show every integer year in *years*
    ys = sorted(int(y) for y in years)
    ax.set_xticks(ys)
    ax.set_xticklabels(ys, rotation=45, ha="right")
    ax.set_xlim(ys[0] - 0.5, ys[-1] + 0.5)

def _cwe_to_category(cwe: str) -> str:
    for cat, members in SUPPLY_CHAIN_CATEGORIES.items():
        if cwe in members:
            return cat
    return "Other"

def _query_db() -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.db import get_connection
    conn = get_connection()
    params = list(SUPPLY_CHAIN_CWES)
    ph = ",".join("?" * len(params))

    yearly_cwe = pd.read_sql_query(
        f"""
        SELECT year, cwe,
               COUNT(*) AS count,
               ROUND(AVG(base_score), 2) AS avg_score,
               SUM(CASE WHEN base_severity='CRITICAL' THEN 1 ELSE 0 END) AS n_critical,
               SUM(CASE WHEN base_severity='HIGH'     THEN 1 ELSE 0 END) AS n_high,
               SUM(CASE WHEN base_severity='MEDIUM'   THEN 1 ELSE 0 END) AS n_medium,
               SUM(CASE WHEN base_severity='LOW'      THEN 1 ELSE 0 END) AS n_low,
               SUM(CASE WHEN attack_vector='NETWORK'  THEN 1 ELSE 0 END) AS n_network
        FROM cves
        WHERE year IS NOT NULL AND cwe IN ({ph})
        GROUP BY year, cwe
        ORDER BY year, count DESC
        """,
        conn, params=params,
    )
    yearly = pd.read_sql_query(
        f"""
        SELECT year,
               COUNT(*) AS total,
               SUM(CASE WHEN base_severity='CRITICAL' THEN 1 ELSE 0 END) AS n_critical,
               SUM(CASE WHEN base_severity='HIGH'     THEN 1 ELSE 0 END) AS n_high,
               SUM(CASE WHEN base_severity='MEDIUM'   THEN 1 ELSE 0 END) AS n_medium,
               SUM(CASE WHEN base_severity='LOW'      THEN 1 ELSE 0 END) AS n_low
        FROM cves
        WHERE year IS NOT NULL AND cwe IN ({ph})
        GROUP BY year ORDER BY year
        """,
        conn, params=params,
    )
    conn.close()
    yearly_cwe["year"] = yearly_cwe["year"].astype(int)
    yearly["year"]     = yearly["year"].astype(int)
    return yearly_cwe, yearly

def _load_from_json() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Demo fallback: build aggregates from data/processed/temporal_features.csv."""
    path = Path("data/processed/temporal_features.csv")
    if not path.exists():
        raise FileNotFoundError("Run preprocess first.")
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)

    yearly_cwe = df[["year", "cwe", "count_t", "avg_score",
                     "pct_critical", "pct_high", "pct_network"]].copy()
    yearly_cwe = yearly_cwe.rename(columns={
        "count_t": "count",
        "pct_critical": "n_critical",
        "pct_high": "n_high",
    })

    yearly = (df.groupby("year")["count_t"]
               .sum().reset_index().rename(columns={"count_t": "total"}))
    for col in ("n_critical", "n_high", "n_medium", "n_low"):
        yearly[col] = 0
    return yearly_cwe, yearly

def plot_volume_by_year(yearly: pd.DataFrame):
    """
    Chart: Total supply-chain CVE count per year.

    What to observe
    ---------------
    An upward trend indicates that the aggregate supply-chain attack surface is
    expanding, either because more third-party components are being scrutinised,
    because attackers are exploiting more supply-chain vectors, or both.
    A plateau or decline may signal improved upstream patching practices.
    """
    years = sorted(yearly["year"].unique())
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(yearly["year"], yearly["total"], color="#1f77b4", width=0.6, zorder=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of CVEs")
    ax.set_title(
        "Supply-Chain CVE Volume by Year\n"
        "Each bar is the total count of CVEs mapped to supply-chain weakness categories"
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    _integer_year_axis(ax, years)
    _save("trend_sc_volume_by_year.png", fig)

def plot_cwe_per_year(yearly_cwe: pd.DataFrame):
    """
    Chart: Year-by-year CVE count for each tracked supply-chain CWE.

    What to observe
    ---------------
    Lines trending upward over consecutive years identify recurring weaknesses
    that are growing in prevalence, prime candidates for the 'emerging threat'
    classification. Lines that peak and then fall suggest a weakness that the
    ecosystem has started to patch out. Flat lines indicate persistent but stable
    threats that do not respond to remediation pressure.
    """
    cwes  = yearly_cwe.groupby("cwe")["count"].sum().nlargest(12).index.tolist()
    years = sorted(yearly_cwe["year"].unique())

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, cwe in enumerate(cwes):
        sub = (yearly_cwe[yearly_cwe["cwe"] == cwe]
               .sort_values("year")
               .set_index("year")["count"]
               .reindex(years, fill_value=0))
        ax.plot(years, sub.values, marker="o", ms=5, lw=2,
                label=cwe, color=CWE_COLORS[i % len(CWE_COLORS)])

    ax.set_xlabel("Year")
    ax.set_ylabel("CVE count")
    ax.set_title(
        "Supply-Chain CVE Trends by Weakness Category (Year-by-Year)\n"
        "Rising lines = growing risk; falling lines = declining or patched weakness"
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    _integer_year_axis(ax, years)
    _save("trend_sc_cwe_per_year.png", fig)

def plot_severity_by_year(yearly_cwe: pd.DataFrame):
    """
    Chart: Severity distribution of supply-chain CVEs per year.

    What to observe
    ---------------
    A growing share of CRITICAL (red) and HIGH (orange) over time indicates that
    supply-chain vulnerabilities are becoming more severe on average, even if
    total volume stays flat. This matters for prioritisation: a stable count
    with a shifting severity mix toward CRITICAL is actually a worsening posture.
    """
    yearly = (yearly_cwe.groupby("year")
              .agg(n_critical=("n_critical","sum"), n_high=("n_high","sum"),
                   total=("count","sum"))
              .reset_index()
              .sort_values("year"))
    yearly["n_other"] = yearly["total"] - yearly["n_critical"] - yearly["n_high"]
    yearly["n_other"] = yearly["n_other"].clip(lower=0)
    years = [int(y) for y in yearly["year"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    bot = np.zeros(len(years))
    for col, label, color in [
        ("n_critical", "CRITICAL", "#d62728"),
        ("n_high",     "HIGH",     "#ff7f0e"),
        ("n_other",    "MEDIUM/LOW","#1f77b4"),
    ]:
        vals = yearly[col].values
        ax.bar(years, vals, bottom=bot, label=label, color=color, width=0.6, zorder=3)
        bot += vals

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of CVEs")
    ax.set_title(
        "Severity Distribution of Supply-Chain CVEs by Year\n"
        "A rising CRITICAL share signals an increasingly severe threat landscape"
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    _integer_year_axis(ax, years)
    _save("trend_sc_severity_by_year.png", fig)

def plot_category_share(yearly_cwe: pd.DataFrame):
    """
    Chart: Proportional share of each supply-chain category per year.

    What to observe
    ---------------
    This stacked area chart reveals structural shifts in the composition of
    supply-chain risk. For example, if Memory Safety begins shrinking as a share
    while Code Execution grows, it suggests that attackers are moving from
    exploiting buffer overflows in system libraries toward weaponising
    deserialization and OS-command injection paths in application-layer packages.
    """
    yearly_cwe = yearly_cwe.copy()
    yearly_cwe["category"] = yearly_cwe["cwe"].apply(_cwe_to_category)
    pivot = (yearly_cwe.groupby(["year","category"])["count"]
             .sum().unstack(fill_value=0).sort_index())
    pivot.index = pivot.index.astype(int)

    cats  = list(SUPPLY_CHAIN_CATEGORIES.keys()) + ["Other"]
    cols  = [c for c in cats if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(13, 6))
    bot = np.zeros(len(pivot))
    for cat in cols:
        ax.bar(pivot.index, pivot[cat].values, bottom=bot,
               label=cat, color=CAT_COLORS.get(cat, "#aaa"), width=0.6, zorder=3)
        bot += pivot[cat].values

    ax.set_xlabel("Year")
    ax.set_ylabel("CVE count")
    ax.set_title(
        "Supply-Chain CVE Volume by Category and Year\n"
        "Tracks whether Memory Safety, Code Execution, or Injection dominates over time"
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    _integer_year_axis(ax, pivot.index.tolist())
    _save("trend_sc_category_share.png", fig)

def _growth_rate(yearly_cwe: pd.DataFrame, recent: int = 3) -> pd.DataFrame:
    max_yr  = int(yearly_cwe["year"].max())
    start_y = max_yr - recent
    end_  = yearly_cwe[yearly_cwe["year"] == max_yr][["cwe","count"]].rename(columns={"count":"end"})
    start_= yearly_cwe[yearly_cwe["year"] == start_y][["cwe","count"]].rename(columns={"count":"start"})
    m = end_.merge(start_, on="cwe", how="inner")
    m["cagr"] = ((m["end"] / m["start"].clip(lower=1)) ** (1/recent)) - 1
    return m.sort_values("cagr", ascending=False)

def _build_report(yearly_cwe, yearly, growth) -> str:
    max_yr  = int(yearly["year"].max())
    min_yr  = int(yearly["year"].min())
    total   = int(yearly["total"].sum())
    top3    = yearly_cwe.groupby("cwe")["count"].sum().nlargest(3).index.tolist()

    top_growing = growth[growth["start"] >= 5].head(5)
    declining   = growth[growth["start"] >= 5].tail(5).sort_values("cagr")

    recent3 = yearly[yearly["year"] >= max_yr - 2]
    pct_net  = 0.0
    try:
        pct_net = round(
            yearly_cwe[yearly_cwe["year"] >= max_yr-2]["n_network"].sum()
            / yearly_cwe[yearly_cwe["year"] >= max_yr-2]["count"].sum() * 100, 1
        )
    except Exception:
        pass
    pct_crit = round(
        yearly[yearly["year"] >= max_yr-2]["n_critical"].sum()
        / yearly[yearly["year"] >= max_yr-2]["total"].sum() * 100, 1
    ) if not yearly[yearly["year"] >= max_yr-2].empty else 0

    lines = [
        "=" * 64,
        "  VulnScope: Supply-Chain CVE Trend Analysis Report",
        "=" * 64,
        "",
        f"  Period         : {min_yr} – {max_yr}",
        f"  Total SC CVEs  : {total:,}",
        f"  Unique CWEs    : {yearly_cwe['cwe'].nunique()}",
        "",
        "Supply-chain vulnerabilities are defined here as CVEs mapped to CWE",
        "categories associated with third-party component weaknesses: memory",
        "safety bugs in C/C++ libraries, deserialization, OS-command injection,",
        "XXE/SSRF in upstream parsers, and access-control defects in shared",
        "libraries. These are the weakness types most commonly propagated",
        "through software dependency trees.",
        "",
        "### Most Recurring Supply-Chain CWEs",
        "",
    ]
    for cwe in top3:
        vol = int(yearly_cwe[yearly_cwe["cwe"] == cwe]["count"].sum())
        sc  = round(float(yearly_cwe[yearly_cwe["cwe"] == cwe]["avg_score"].mean()), 2)
        cat = _cwe_to_category(cwe)
        lines.append(f"  {cwe:<10} {cat:<25} {vol:>6,} CVEs  avg CVSS {sc}")

    lines += ["", "### Fastest Growing (3-year CAGR)", ""]
    for _, r in top_growing.iterrows():
        lines.append(
            f"  {r['cwe']:<10} +{r['cagr']*100:.0f}% CAGR  "
            f"({int(r['start'])} → {int(r['end'])} CVEs)"
        )

    lines += ["", "### Declining CWEs (3-year CAGR)", ""]
    for _, r in declining.iterrows():
        lines.append(
            f"  {r['cwe']:<10} {r['cagr']*100:+.0f}% CAGR  "
            f"({int(r['start'])} → {int(r['end'])} CVEs)"
        )

    lines += [
        "",
        "### Recent Threat Profile (last 3 years)",
        "",
        f"  {pct_net}% of recent supply-chain CVEs are network-exploitable.",
        f"  {pct_crit}% are rated CRITICAL severity.",
        "",
        "### Interpretation",
        "",
        "  Growing CWEs indicate weakness categories where upstream patching",
        "  and developer awareness have not kept pace with attacker discovery.",
        "  Declining CWEs may reflect successful ecosystem-wide remediation",
        "  (e.g. mandatory SAST rules, secure-by-default library updates).",
        "  High pct_network + HIGH/CRITICAL severity is the most dangerous",
        "  combination for supply-chain attacks: remote, no-auth exploitation",
        "  of a transitive dependency.",
        "",
        "=" * 64,
    ]
    return "\n".join(lines)

def analyse(demo: bool = False) -> dict:
    if demo:
        yearly_cwe, yearly = _load_from_json()
    else:
        yearly_cwe, yearly = _query_db()

    if yearly.empty:
        logger.error("No data. Run 'python main.py fetch' first.")
        return {}

    growth = _growth_rate(yearly_cwe)

    plot_volume_by_year(yearly)
    plot_cwe_per_year(yearly_cwe)
    plot_severity_by_year(yearly_cwe)
    plot_category_share(yearly_cwe)

    report = _build_report(yearly_cwe, yearly, growth)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(report)
    logger.info("Report: %s", REPORT_PATH)
    return {"yearly": yearly, "yearly_cwe": yearly_cwe, "growth": growth}
