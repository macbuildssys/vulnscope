"""
db.py
SQLite persistence layer for VulnScope.

Schema
------
cves        - one row per CVE with CVSS v3 fields parsed out
fetch_runs  - audit log of each download run

The database file lives at data/vulnscope.db (gitignored).
All reads return DataFrames; writes accept lists of raw NVD records.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from src.config import DATA_DIR

logger = logging.getLogger(__name__)
DB_PATH = DATA_DIR / "vulnscope.db"

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist yet."""
    conn = get_connection(db_path)
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cves (
                cve_id               TEXT PRIMARY KEY,
                published            TEXT,
                year                 INTEGER,
                cwe                  TEXT,
                attack_vector        TEXT,
                attack_complexity    TEXT,
                privileges_required  TEXT,
                user_interaction     TEXT,
                scope                TEXT,
                confidentiality      TEXT,
                integrity            TEXT,
                availability         TEXT,
                base_score           REAL,
                base_severity        TEXT,
                exploitability_score REAL,
                impact_score         REAL,
                cvss_version         TEXT,
                fetched_at           TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_cves_year
                ON cves (year);

            CREATE INDEX IF NOT EXISTS idx_cves_severity
                ON cves (base_severity);

            CREATE INDEX IF NOT EXISTS idx_cves_score
                ON cves (base_score);

            CREATE TABLE IF NOT EXISTS fetch_runs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at     TEXT NOT NULL,
                finished_at    TEXT,
                records_added  INTEGER DEFAULT 0,
                total_in_db    INTEGER DEFAULT 0,
                api_key_used   INTEGER DEFAULT 0
            );
        """)
    conn.close()
    logger.info("Database initialised at %s", db_path)

def _extract_cvss_v3(record: dict) -> Optional[dict]:
    metrics = record.get("cve", {}).get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key)
        if not entries:
            continue
        entry = entries[0]
        data = entry.get("cvssData", {})
        return {
            "attack_vector": data.get("attackVector"),
            "attack_complexity": data.get("attackComplexity"),
            "privileges_required": data.get("privilegesRequired"),
            "user_interaction": data.get("userInteraction"),
            "scope": data.get("scope"),
            "confidentiality": data.get("confidentialityImpact"),
            "integrity": data.get("integrityImpact"),
            "availability": data.get("availabilityImpact"),
            "base_score": data.get("baseScore"),
            "base_severity": data.get("baseSeverity"),
            "exploitability_score": entry.get("exploitabilityScore"),
            "impact_score": entry.get("impactScore"),
            "cvss_version": key,
        }
    return None

def _extract_cwe(record: dict) -> str:
    for weakness in record.get("cve", {}).get("weaknesses", []):
        for desc in weakness.get("description", []):
            val = desc.get("value", "")
            if val.upper().startswith("CWE-") and val.upper() != "CWE-NOINFO":
                return val.upper()
    return "CWE-OTHER"

def upsert_cves(records: list, conn: sqlite3.Connection) -> int:
    """
    Insert or replace CVE records into the cves table.
    Returns the number of rows written.
    """
    rows = []
    now = datetime.now(timezone.utc).isoformat()

    for rec in records:
        cve = rec.get("cve", {})
        cve_id = cve.get("id", "")
        published = cve.get("published", "")
        year = int(published[:4]) if published and len(published) >= 4 else None

        cvss = _extract_cvss_v3(rec)
        if cvss is None or cvss.get("base_score") is None:
            continue

        cwe = _extract_cwe(rec)

        rows.append((
            cve_id, published, year, cwe,
            cvss["attack_vector"], cvss["attack_complexity"],
            cvss["privileges_required"], cvss["user_interaction"],
            cvss["scope"], cvss["confidentiality"],
            cvss["integrity"], cvss["availability"],
            cvss["base_score"], cvss["base_severity"],
            cvss["exploitability_score"], cvss["impact_score"],
            cvss["cvss_version"], now,
        ))

    if rows:
        conn.executemany(
            """INSERT OR REPLACE INTO cves (
                cve_id, published, year, cwe,
                attack_vector, attack_complexity, privileges_required,
                user_interaction, scope, confidentiality, integrity, availability,
                base_score, base_severity, exploitability_score, impact_score,
                cvss_version, fetched_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )

    return len(rows)

def count_cves(db_path: Path = DB_PATH) -> int:
    conn = get_connection(db_path)
    count = conn.execute("SELECT COUNT(*) FROM cves").fetchone()[0]
    conn.close()
    return count

def load_as_dataframe(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Return the full cves table as a DataFrame for use in the ML pipeline.
    Columns mirror the raw NVD field names expected by preprocess.py.
    """
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            cve_id,
            year,
            cwe,
            attack_vector     AS attackVector,
            attack_complexity AS attackComplexity,
            privileges_required AS privilegesRequired,
            user_interaction  AS userInteraction,
            scope,
            confidentiality   AS confidentialityImpact,
            integrity         AS integrityImpact,
            availability      AS availabilityImpact,
            base_score        AS baseScore,
            base_severity     AS baseSeverity,
            exploitability_score AS exploitabilityScore,
            impact_score      AS impactScore
        FROM cves
        WHERE base_score IS NOT NULL
        """,
        conn,
    )
    conn.close()
    logger.info("Loaded %d CVE records from database.", len(df))
    return df

def db_summary(db_path: Path = DB_PATH) -> dict:
    """Return a summary dict of what is stored in the database."""
    conn = get_connection(db_path)
    total = conn.execute("SELECT COUNT(*) FROM cves").fetchone()[0]
    by_sev = conn.execute(
        "SELECT base_severity, COUNT(*) FROM cves GROUP BY base_severity ORDER BY COUNT(*) DESC"
    ).fetchall()
    score_stats = conn.execute(
        "SELECT MIN(base_score), AVG(base_score), MAX(base_score) FROM cves"
    ).fetchone()
    year_range = conn.execute(
        "SELECT MIN(year), MAX(year) FROM cves WHERE year IS NOT NULL"
    ).fetchone()
    last_fetch = conn.execute(
        "SELECT finished_at, records_added FROM fetch_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return {
        "total_cves": total,
        "by_severity": dict(by_sev),
        "score_min": score_stats[0],
        "score_avg": round(score_stats[1], 2) if score_stats[1] else None,
        "score_max": score_stats[2],
        "year_range": list(year_range) if year_range else None,
        "last_fetch": {"at": last_fetch[0], "added": last_fetch[1]} if last_fetch else None,
    }

def query_similar_cves(
    attack_vector: str,
    attack_complexity: str,
    privileges_required: str,
    predicted_score: float,
    cwe: str = None,
    score_tolerance: float = 1.0,
    limit: int = 5,
    db_path: Path = DB_PATH,
) -> list:
    """
    Return CVEs from the DB that share the same Attack Vector, Complexity,
    and Privileges Required, with a base score within *score_tolerance* of
    *predicted_score*. Optionally filter by CWE.
    Results are sorted by how close their score is to the predicted score.
    """
    if not db_path.exists():
        return []

    conn = get_connection(db_path)
    params = [
        attack_vector.upper(),
        attack_complexity.upper(),
        privileges_required.upper(),
        predicted_score - score_tolerance,
        predicted_score + score_tolerance,
    ]

    cwe_clause = ""
    if cwe and cwe.upper() != "CWE-OTHER":
        cwe_clause = "AND cwe = ?"
        params.append(cwe.upper())

    rows = conn.execute(
        f"""
        SELECT
            cve_id,
            base_score,
            base_severity,
            cwe,
            year,
            ABS(base_score - ?) AS score_delta
        FROM cves
        WHERE attack_vector        = ?
          AND attack_complexity    = ?
          AND privileges_required  = ?
          AND base_score BETWEEN ? AND ?
          {cwe_clause}
          AND base_score IS NOT NULL
        ORDER BY score_delta ASC, base_score DESC
        LIMIT ?
        """,
        [predicted_score] + params + [limit],
    ).fetchall()
    conn.close()

    return [
        {
            "cve_id":    r[0],
            "score":     r[1],
            "severity":  r[2],
            "cwe":       r[3] or "N/A",
            "year":      r[4],
        }
        for r in rows
    ]

def query_cwe_stats(cwe: str, db_path: Path = DB_PATH) -> dict:
    
    # Return prevalence and score statistics for a given CWE across the DB
    if not db_path.exists() or not cwe:
        return {}

    conn = get_connection(db_path)
    total_db = conn.execute("SELECT COUNT(*) FROM cves").fetchone()[0]

    row = conn.execute(
        """
        SELECT
            COUNT(*)                          AS cnt,
            ROUND(AVG(base_score), 2)         AS avg_score,
            ROUND(MIN(base_score), 1)         AS min_score,
            ROUND(MAX(base_score), 1)         AS max_score
        FROM cves
        WHERE cwe = ?
        """,
        (cwe.upper(),),
    ).fetchone()

    by_sev = conn.execute(
        """
        SELECT base_severity, COUNT(*) FROM cves
        WHERE cwe = ?
        GROUP BY base_severity
        ORDER BY COUNT(*) DESC
        """,
        (cwe.upper(),),
    ).fetchall()

    # Most recent notable CVEs for this CWE (highest scoring)
    top_cves = conn.execute(
        """
        SELECT cve_id, base_score, base_severity, year
        FROM cves
        WHERE cwe = ?
        ORDER BY base_score DESC, year DESC
        LIMIT 5
        """,
        (cwe.upper(),),
    ).fetchall()

    conn.close()

    if not row or row[0] == 0:
        return {"found": False, "cwe": cwe}

    cnt = row[0]
    return {
        "found":            True,
        "cwe":              cwe.upper(),
        "total_in_db":      cnt,
        "pct_of_all_cves":  round(cnt / total_db * 100, 2) if total_db else 0,
        "avg_score":        row[1],
        "min_score":        row[2],
        "max_score":        row[3],
        "severity_breakdown": dict(by_sev),
        "top_scoring_cves": [
            {"cve_id": r[0], "score": r[1], "severity": r[2], "year": r[3]}
            for r in top_cves
        ],
    }

def query_vector_prevalence(
    attack_vector: str,
    attack_complexity: str,
    privileges_required: str,
    user_interaction: str,
    db_path: Path = DB_PATH,
) -> dict:
    """
    Return what percentage of historical CVEs share each input vector component,
    and the score distribution for CVEs with this exact exploitability profile.
    """
    if not db_path.exists():
        return {}

    conn = get_connection(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM cves").fetchone()[0]
        if total == 0:
            return {}

        def pct(col, val):
            n = conn.execute(
                f"SELECT COUNT(*) FROM cves WHERE {col} = ?", (val.upper(),)
            ).fetchone()[0]
            return round(n / total * 100, 1)

        av_pct  = pct("attack_vector",       attack_vector)
        ac_pct  = pct("attack_complexity",   attack_complexity)
        pr_pct  = pct("privileges_required", privileges_required)
        ui_pct  = pct("user_interaction",    user_interaction)

        exact = conn.execute(
            """
            SELECT COUNT(*), ROUND(AVG(base_score), 2), ROUND(MIN(base_score), 1), ROUND(MAX(base_score), 1)
            FROM cves
            WHERE attack_vector       = ?
              AND attack_complexity   = ?
              AND privileges_required = ?
              AND user_interaction    = ?
            """,
            (attack_vector.upper(), attack_complexity.upper(),
             privileges_required.upper(), user_interaction.upper()),
        ).fetchone()

        sev_dist = conn.execute(
            """
            SELECT base_severity, COUNT(*) FROM cves
            WHERE attack_vector       = ?
              AND attack_complexity   = ?
              AND privileges_required = ?
              AND user_interaction    = ?
            GROUP BY base_severity
            ORDER BY COUNT(*) DESC
            """,
            (attack_vector.upper(), attack_complexity.upper(),
             privileges_required.upper(), user_interaction.upper()),
        ).fetchall()
    finally:
        conn.close()

    return {
        "pct_with_same_attack_vector":       av_pct,
        "pct_with_same_attack_complexity":   ac_pct,
        "pct_with_same_privileges_required": pr_pct,
        "pct_with_same_user_interaction":    ui_pct,
        "exact_profile_matches": {
            "count":     exact[0] if exact else 0,
            "avg_score": exact[1] if exact else None,
            "min_score": exact[2] if exact else None,
            "max_score": exact[3] if exact else None,
            "severity_distribution": dict(sev_dist) if sev_dist else {},
        },
    }
