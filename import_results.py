#!/usr/bin/env python3
"""
One-shot: import an existing results.json into the SQLite store so prior research
shows up in the dashboard. Idempotent-ish — re-running appends the records again,
so run it once (or clear prospector.db first).

Usage:
  python import_results.py                 # imports ./results.json
  python import_results.py my_results.json
"""

import json
import sys
from pathlib import Path

import db

HERE = Path(__file__).parent


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json"
    if not path.exists():
        sys.exit(f"No results file at {path}. Nothing to import.")

    records = json.loads(path.read_text())
    if not isinstance(records, list):
        sys.exit(f"{path} is not a JSON list of prospect records.")

    db.init_db()
    n = 0
    for rec in records:
        if not isinstance(rec, dict) or not rec.get("company"):
            continue
        db.insert_prospect(rec, run_id=None)  # run_id NULL = legacy/imported
        n += 1
    print(f"Imported {n} prospect(s) from {path} into {db.DB_PATH}.")


if __name__ == "__main__":
    main()
