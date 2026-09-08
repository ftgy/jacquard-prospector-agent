#!/usr/bin/env python3
"""
Backfill niche categories onto existing discovery runs.

Discovery runs now get filed under a location-agnostic niche category at launch
(so "real estate agencies in Barcelona" and "...in Marbella" share one list).
Runs that predate that feature have no category; this walks them oldest-first and
assigns each one, reusing a matching category or creating a new one — the same
logic new runs use (service.categorize_run).

Only 'discover' runs are categorized; the companies tab stays run-grouped. Runs
are processed oldest-first so the earliest search of a niche seeds its category
name and later ones reuse it. Idempotent: a run that already has a category is
skipped, so it's safe to re-run.

Usage:
    python scripts/categorize_runs.py            # categorize all uncategorized runs
    python scripts/categorize_runs.py --dry-run  # show what it would assign, write nothing
    python scripts/categorize_runs.py --limit 20 # only the first N uncategorized runs
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # find prospector/

from prospector import db
from prospector.config import make_client
from prospector.service import categorize_run, friendly_api_error


def uncategorized_discover_runs(limit: int | None) -> list[dict]:
    """Discovery runs with no category yet, OLDEST first (so early searches seed
    the category names later ones reuse)."""
    runs = [r for r in db.list_runs(kind="discover", limit=10_000)
            if r["category_id"] is None]
    runs.sort(key=lambda r: r["id"])          # list_runs is newest-first; flip it
    return runs[:limit] if limit else runs


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill niche categories onto runs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show assignments without writing them.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only categorize the first N uncategorized runs.")
    args = ap.parse_args()

    db.init_db()
    runs = uncategorized_discover_runs(args.limit)
    if not runs:
        print("Nothing to do — every discovery run already has a category.")
        return 0

    print(f"{len(runs)} uncategorized discovery run(s) to file"
          + (" (dry run)" if args.dry_run else "") + ":\n")

    try:
        client = make_client()
    except SystemExit as e:
        print(f"API client not configured: {e}", file=sys.stderr)
        return 1

    done = 0
    for run in runs:
        query = run["query"]
        if args.dry_run:
            # Don't create rows on a dry run — just show the LLM's name.
            from prospector.agent import categorize_niche
            try:
                name = categorize_niche(
                    client, query, [c["name"] for c in db.list_categories()])
            except Exception as e:
                print(f"  run {run['id']:>4}  {query!r}  -> ERROR: "
                      f"{friendly_api_error(e)}", file=sys.stderr)
                continue
            print(f"  run {run['id']:>4}  {query!r}  ->  {name!r}")
            continue

        category_id = categorize_run(client, query)  # find/creates, best-effort
        if category_id is None:
            print(f"  run {run['id']:>4}  {query!r}  ->  (could not categorize, "
                  "left uncategorized)", file=sys.stderr)
            continue
        db.set_run_category(run["id"], category_id)
        cat = db.get_category(category_id)
        print(f"  run {run['id']:>4}  {query!r}  ->  {cat['name']!r}")
        done += 1

    if not args.dry_run:
        print(f"\nFiled {done} run(s) into {len(db.list_categories())} "
              "categor(y/ies).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
