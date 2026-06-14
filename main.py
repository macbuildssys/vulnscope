#!/usr/bin/env python3
"""
VulnScope: Historical CVE Trend Analysis and Future Risk Forecasting

Workflow (three commands)
--------------------------
  python main.py fetch      Download CVE data from NVD into SQLite (one-time)
  python main.py build      Analyse trends + train models + evaluate
  python main.py predict    Forecast emerging threats for the next year (offline)

Additional
----------
  python main.py demo       Full pipeline on synthetic data (no network needed)
  python main.py db-status  Show what is in the database
  python main.py init       Set up / re-enter NVD API key
"""

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def cmd_init(args):
    import getpass, requests
    from src.keystore import save_api_key, get_api_key

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║       VulnScope. NVD API Key Setup.  ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    existing = get_api_key()
    if existing:
        masked = existing[:6] + "..." + existing[-4:]
        print(f"  Stored key: {masked}")
        if input("  Re-enter? [y/N] ").strip().lower() not in ("y", "yes"):
            _validate_key(existing)
            return

    print("  Get a free key at: https://nvd.nist.gov/developers/request-an-api-key")
    print("  (Keys activate within ~1 hour of registration)\n")

    while True:
        key = "".join(getpass.getpass("  Paste NVD API key (hidden): ").split())
        if len(key) < 20:
            print("  Too short. Try again."); continue

        status = _validate_key(key)
        if status == "valid":
            save_api_key(key)
            print("  Key saved. Run: python main.py fetch"); break
        elif status == "not_activated":
            if input("  Save anyway? [Y/n] ").strip().lower() in ("", "y", "yes"):
                save_api_key(key)
                print("  Saved. Use --no-key until it activates."); break
            break
        else:
            if input("  Try again? [Y/n] ").strip().lower() not in ("", "y", "yes"):
                break
    print()

def _validate_key(key):
    import requests
    try:
        r = requests.get(
            "https://services.nvd.nist.gov/rest/json/cves/2.0",
            params={"resultsPerPage": 1},
            headers={"apiKey": key, "User-Agent": "VulnScope-ML/1.0"},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"  Network error: {e}"); return "error"

    if r.status_code == 200:
        total = r.json().get("totalResults", "?")
        print(f"  Key valid. NVD has {total:,} CVEs."); return "valid"
    if r.status_code in (403, 404):
        print(f"  Key not yet active ({r.status_code}). Try again in ~1 hour."); return "not_activated"
    print(f"  Unexpected status {r.status_code}."); return "invalid"

def cmd_fetch(args):
    from src.fetch_data import fetch_nvd_data
    written = fetch_nvd_data(
        max_cves=args.max_cves,
        api_key=args.api_key,
        no_key=args.no_key,
        start_year=args.start_year or None,
        end_year=args.end_year or None,
    )
    print(f"\n  Done. {written:,} CVE records stored in the database.\n")

def cmd_db_status(args):
    from src.db import db_summary, DB_PATH
    if not DB_PATH.exists():
        print("\n  No database found. Run: python main.py fetch\n"); return
    s = db_summary()
    print(f"\n  Database : {DB_PATH}")
    print(f"  CVEs     : {s['total_cves']:,}")
    print(f"  Scores   : {s['score_min']} - {s['score_max']}  (mean {s['score_avg']})")
    if s["year_range"]:
        print(f"  Years    : {s['year_range'][0]} - {s['year_range'][1]}")
    print("  Severity:")
    for sev, n in sorted(s["by_severity"].items()):
        print(f"    {sev:<12} {n:>8,}")
    if s["last_fetch"]:
        print(f"  Last fetch: {s['last_fetch']['at']}  (+{s['last_fetch']['added']:,})")
    print()

def cmd_build(args):
    # Full build: trend analysis + feature engineering + train + evaluate
    if args.demo:
        _ensure_demo_data()
        logger.info("Build 1/4: Analysing trends ...")
        from src.analyse import analyse
        analyse()
        logger.info("Build 2/4: Feature engineering ...")
        from src.preprocess import preprocess_from_json
        df = preprocess_from_json()
    else:
        logger.info("Build 1/4: Analysing trends ...")
        from src.analyse import analyse
        analyse()
        logger.info("Build 2/4: Feature engineering ...")
        from src.preprocess import preprocess_from_db
        df = preprocess_from_db()

    logger.info("Build 3/4: Training models ...")
    from src.train import train
    artifacts = train(df=df)
    print(json.dumps(artifacts["cv_summary"], indent=2))

    logger.info("Build 4/4: Evaluating ...")
    from src.evaluate import evaluate
    metrics = evaluate()
    print(json.dumps(metrics, indent=2))

    logger.info("Build complete. Figures in results/figures/")

def cmd_predict(args):
    from src.predict import predict_threats
    result = predict_threats(forecast_year=args.year or None)
    print(json.dumps(result, indent=2))

def cmd_demo(args):
    logger.info("[DEMO] Generating synthetic CVE data ...")
    _ensure_demo_data()
    cmd_build(argparse.Namespace(demo=True))

def _ensure_demo_data():
    from pathlib import Path
    if not Path("data/raw/cves.json").exists():
        from src.demo_data import generate_demo_data
        generate_demo_data()

def build_parser():
    p = argparse.ArgumentParser(
        description="VulnScope: CVE Trend Analysis and Future Risk Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="Set up NVD API key")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("fetch", help="Download CVEs from NVD into SQLite")
    s.add_argument("--max-cves", type=int, default=0, dest="max_cves")
    s.add_argument("--start-year", type=int, default=None, dest="start_year")
    s.add_argument("--end-year",   type=int, default=None, dest="end_year")
    s.add_argument("--api-key", default=None, dest="api_key")
    s.add_argument("--no-key", action="store_true", dest="no_key")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("db-status", help="Show database contents")
    s.set_defaults(func=cmd_db_status)

    s = sub.add_parser("build", help="Trend analysis + train + evaluate (after fetch)")
    s.add_argument("--demo", action="store_true")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("predict", help="Forecast emerging threats (offline)")
    s.add_argument("--year", type=int, default=None,
                   help="Year to forecast (default: next year after training data)")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("demo", help="Full pipeline on synthetic data (no network)")
    s.set_defaults(func=cmd_demo)

    return p

def main():
    build_parser().parse_args().func(build_parser().parse_args())

if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
