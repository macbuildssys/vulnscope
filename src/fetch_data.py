"""
fetch_data.py
Download CVE records from the NVD REST API v2.0, store directly in SQLite.

Pagination strategy
-------------------
By default the fetch pages through the entire NVD dataset without any date
filter. This is the most reliable approach: NVD contains all years (1999-2026)
in its index and pagination returns them in full.

If --start-year / --end-year are supplied the range is split into 120-day
chunks, which is the maximum window the NVD API accepts for date-filtered
requests. Larger windows cause a 404.

Key delivery
------------
The API key is passed as a request header (apiKey) per the NVD API 2.0 spec.
Passing it as a URL query parameter causes a 404.

Resilience
----------
- Pre-flight check before the main loop.
- Fatal HTTP codes abort immediately without pointless retries.
- Each page is committed to the DB before moving on, so progress survives
  interruption. Re-running upserts rather than duplicating rows.
"""

import logging
import os
import time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from typing import Optional, Iterator, Tuple

import requests

from src.config import (
    NVD_API_BASE,
    NVD_RESULTS_PER_PAGE,
    NVD_RATE_LIMIT_SLEEP,
    NVD_RATE_LIMIT_SLEEP_KEYED,
    NVD_DEFAULT_START_YEAR,
    NVD_DEFAULT_END_YEAR,
)
from src.db import DB_PATH, get_connection, init_db, upsert_cves

logger = logging.getLogger(__name__)

_FATAL_STATUSES = {400, 401, 422}
NVD_MAX_DATE_WINDOW_DAYS = 119   # NVD enforces < 120 days per filtered request

def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    """Precedence: explicit arg > NVD_API_KEY env var > keystore."""
    if api_key:
        return api_key
    env_key = os.environ.get("NVD_API_KEY")
    if env_key:
        logger.info("Using NVD API key from NVD_API_KEY environment variable.")
        return env_key
    from src.keystore import get_api_key
    stored = get_api_key()
    if stored:
        logger.info("Using NVD API key from keystore (~/.config/vulnscope/config.json).")
        return stored
    return None

def _make_headers(api_key: Optional[str]) -> dict:
    headers = {"User-Agent": "VulnScope-ML/1.0 (https://github.com/macbuildssys/vulnscope)"}
    if api_key:
        headers["apiKey"] = api_key   # header, NOT query param, NVD spec requirement
    return headers

def _fmt_date(d: date) -> str:
    """NVD API date format: YYYY-MM-DDTHH:MM:SS.000"""
    return d.strftime("%Y-%m-%dT00:00:00.000")

def _fmt_date_end(d: date) -> str:
    return d.strftime("%Y-%m-%dT23:59:59.999")

def _date_windows(start_year: int, end_year: int) -> Iterator[Tuple[date, date]]:
    """
    Yield (window_start, window_end) pairs that cover start_year-01-01 to
    end_year-12-31 in chunks of at most NVD_MAX_DATE_WINDOW_DAYS days.
    """
    cursor = date(start_year, 1, 1)
    end    = date(end_year, 12, 31)
    while cursor <= end:
        window_end = min(cursor + timedelta(days=NVD_MAX_DATE_WINDOW_DAYS), end)
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)

def _preflight_check(api_key: Optional[str]) -> bool:
    """
    One small request without date filters to confirm the endpoint and key
    are both working before committing to a full download.
    """
    headers = _make_headers(api_key)
    params  = {"resultsPerPage": 1, "startIndex": 0}

    try:
        resp = requests.get(NVD_API_BASE, params=params, headers=headers, timeout=15)
    except requests.exceptions.RequestException as exc:
        logger.error("Pre-flight check failed (network): %s", exc)
        return False

    if resp.status_code == 200:
        total = resp.json().get("totalResults", "?")
        logger.info(
            "NVD API reachable. Total CVEs in NVD: %s",
            f"{total:,}" if isinstance(total, int) else total,
        )
        return True

    if resp.status_code in (403, 401):
        logger.error(
            "API key rejected (%s). It may not be activated yet (up to 24 h after registration).\n"
            "  Re-enter: python main.py init\n"
            "  Skip key: python main.py fetch --no-key",
            resp.status_code,
        )
        return False

    logger.error(
        "Unexpected HTTP %s from NVD pre-flight. Body: %s",
        resp.status_code, resp.text[:300],
    )
    return False

def _fetch_pages(
    headers: dict,
    sleep_secs: float,
    conn,
    run_id: int,
    max_cves: int,
    date_filter: Optional[Tuple[date, date]] = None,
) -> int:
    """
    Page through one query window (optionally date-filtered) and upsert
    records into the DB. Returns the number of records written.
    """
    written_total = 0
    start_index   = 0
    retries       = 0
    max_retries   = 3
    stop_at       = max_cves if max_cves > 0 else float("inf")

    while start_index < stop_at:
        page_size = NVD_RESULTS_PER_PAGE if stop_at == float('inf') else min(NVD_RESULTS_PER_PAGE, int(stop_at - start_index))
        params    = {"startIndex": start_index, "resultsPerPage": page_size}

        if date_filter:
            params["pubStartDate"] = _fmt_date(date_filter[0])
            params["pubEndDate"]   = _fmt_date_end(date_filter[1])

        try:
            resp = requests.get(NVD_API_BASE, params=params, headers=headers, timeout=30)
        except requests.exceptions.RequestException as exc:
            retries += 1
            logger.error("Network error (%d/%d): %s", retries, max_retries, exc)
            if retries >= max_retries:
                logger.error("Max retries reached. Stopping this window.")
                break
            time.sleep(30)
            continue

        if resp.status_code in _FATAL_STATUSES:
            logger.error("Fatal HTTP %s — aborting. URL: %s", resp.status_code, resp.url)
            break

        if resp.status_code == 403:
            logger.warning("Rate limited (403). Sleeping 35 s ...")
            time.sleep(35)
            continue

        if resp.status_code == 404:
            logger.error(
                "HTTP 404 on this request. If you used date filters, the window may exceed "
                "120 days — VulnScope chunks automatically, so this should not happen. "
                "URL: %s", resp.url,
            )
            break

        if resp.status_code == 503:
            logger.warning("NVD 503 (service unavailable). Sleeping 60 s ...")
            time.sleep(60)
            continue

        if resp.status_code != 200:
            retries += 1
            logger.error("HTTP %s (%d/%d). Sleeping 30 s ...", resp.status_code, retries, max_retries)
            if retries >= max_retries:
                break
            time.sleep(30)
            continue

        retries = 0
        payload = resp.json()
        batch   = payload.get("vulnerabilities", [])
        total_available = payload.get("totalResults", 0)

        if not batch:
            break

        written = upsert_cves(batch, conn)
        conn.commit()
        written_total += written
        start_index   += len(batch)

        effective_stop = min(stop_at, total_available) if stop_at != float("inf") else total_available
        pct = (start_index / effective_stop * 100) if effective_stop else 0

        window_label = (
            f"{date_filter[0]}:{date_filter[1]}" if date_filter else "all"
        )
        logger.info(
            "  [%5.1f%%]  idx %d-%d  +%d stored  run_total=%d  window=%s",
            pct, start_index - len(batch), start_index,
            written, written_total, window_label,
        )

        if start_index >= effective_stop:
            break

        time.sleep(sleep_secs)

    return written_total

def fetch_nvd_data(
    max_cves: int = 0,
    api_key: Optional[str] = None,
    db_path: Path = DB_PATH,
    no_key: bool = False,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> int:
    """
    Download CVE records from NVD and store them in SQLite.

    Parameters
    ----------
    max_cves:
        Upper bound on records (0 = unlimited, fetch everything).
    api_key:
        Overrides the stored/env key.
    no_key:
        Force unauthenticated mode (ignores any stored key).
    start_year / end_year:
        When both are supplied, restrict to CVEs published in that year range
        using 120-day chunked date windows. When omitted, plain pagination is
        used (faster and covers all years in NVD).
    """
    if no_key:
        api_key = None
        logger.info("Running without API key (rate limit: ~5 req / 30 s).")
    else:
        api_key = _resolve_api_key(api_key)
        if not api_key:
            from src.keystore import prompt_for_key
            api_key = prompt_for_key(allow_skip=True)

    if not _preflight_check(api_key):
        logger.error("Aborting fetch due to pre-flight failure.")
        return 0

    sleep_secs = NVD_RATE_LIMIT_SLEEP_KEYED if api_key else NVD_RATE_LIMIT_SLEEP
    headers    = _make_headers(api_key)

    init_db(db_path)
    conn = get_connection(db_path)

    started_at = datetime.now(timezone.utc).isoformat()
    cursor_row = conn.execute(
        "INSERT INTO fetch_runs (started_at, api_key_used) VALUES (?, ?)",
        (started_at, 1 if api_key else 0),
    )
    run_id = cursor_row.lastrowid
    conn.commit()

    use_date_filter = (start_year is not None and end_year is not None)
    mode = "keyed" if api_key else "unauthenticated"

    if use_date_filter:
        logger.info(
            "Fetch mode: date-filtered %d-%d, chunked into %d-day windows, %s, %.1f s/page",
            start_year, end_year, NVD_MAX_DATE_WINDOW_DAYS, mode, sleep_secs,
        )
    else:
        logger.info(
            "Fetch mode: full pagination (all years), %s, %.1f s/page",
            mode, sleep_secs,
        )

    total_written = 0
    try:
        if use_date_filter:
            windows = list(_date_windows(start_year, end_year))
            logger.info("Date range %d-%d splits into %d windows.", start_year, end_year, len(windows))
            for i, (win_start, win_end) in enumerate(windows, 1):
                logger.info("Window %d/%d: %s to %s", i, len(windows), win_start, win_end)
                written = _fetch_pages(
                    headers, sleep_secs, conn, run_id,
                    max_cves=max_cves,
                    date_filter=(win_start, win_end),
                )
                total_written += written
        else:
            total_written = _fetch_pages(
                headers, sleep_secs, conn, run_id,
                max_cves=max_cves,
                date_filter=None,
            )
    finally:
        finished_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE fetch_runs SET finished_at=?, records_added=?, total_in_db=? WHERE id=?",
            (finished_at, total_written, total_written, run_id),
        )
        conn.commit()
        conn.close()

    logger.info("Fetch complete. %d records written to %s", total_written, db_path)
    return total_written
