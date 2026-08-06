#!/usr/bin/env python3
"""
Populate the prospect database unattended, one small batch per run.

Designed to be driven by cron (a batch every few hours) so cost and load spread
out and a flaky endpoint hurts less. Each invocation:

  1. Takes a lock so two runs never overlap.
  2. Auto-suggests B2B niches for a location and picks one not researched before.
  3. Discovers companies for that niche, drops any already in the DB.
  4. Researches + qualifies the new ones, persisting as it goes.

Niche-suggestion and discovery double as a health gate: if the endpoint is down
they fail first, and the run exits BEFORE writing any error rows. A burst of
consecutive research failures mid-batch also aborts, so a proxy that dies partway
through doesn't fill the DB with junk.

Usage:
    python scripts/populate.py                     # one batch for the default location
    python scripts/populate.py --location "Valencia, Spain" --count 15
    python scripts/populate.py --niche "property management firms in Barcelona"
    python scripts/populate.py --dry-run           # suggest + discover, persist nothing

Location defaults to $POPULATE_LOCATION or "Spain". Exit 0 on success or a clean
skip (locked / endpoint down / nothing new); 1 on an unexpected error.
"""

import argparse
import fcntl
import logging
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # find prospector/

from prospector import db
from prospector.agent import discover_candidates, run_prospect, suggest_niches
from prospector.config import make_client
from prospector.icp import ICP
from prospector.service import friendly_api_error

log = logging.getLogger("populate")

PACE_SECONDS = 2.0          # gap between company research calls (be gentle)
MAX_CONSECUTIVE_ERRORS = 3  # abort the batch if this many in a row fail


def setup_logging(log_path: Path) -> None:
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def acquire_lock(lock_path: Path):
    """Return an open, flock'd file handle, or None if another run holds it."""
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def existing_companies() -> set:
    """Lowercased names already in the DB, for dedup."""
    return {(p.get("company") or "").strip().lower()
            for p in db.list_prospects()}


def researched_niches() -> set:
    """Niche queries already run (discover runs), so we don't repeat them."""
    return {(r.get("query") or "").strip().lower()
            for r in db.list_runs(kind="discover", limit=500)}


def pick_niche(client, location: str) -> str | None:
    """Auto-suggest niches for the location and choose one not done before."""
    suggestions = suggest_niches(client, location, ICP, count=15)
    niches = [s.get("niche", "").strip() for s in suggestions if s.get("niche")]
    if not niches:
        return None
    done = researched_niches()
    fresh = [n for n in niches if n.lower() not in done]
    if fresh:
        return random.choice(fresh)
    # Everything suggested has been run before; company-level dedup still guards
    # against duplicates, so re-run one to keep discovering new companies within it.
    log.info("All suggested niches already researched; reusing one.")
    return random.choice(niches)


def research_batch(client, run_id: int, candidates: list) -> dict:
    """Research + qualify each candidate, persisting as we go. Aborts early on a
    burst of consecutive failures (endpoint likely down)."""
    counts = {"ok": 0, "error": 0, "A": 0, "B": 0, "C": 0, "disqualified": 0}
    consecutive = 0
    for cand in candidates:
        company = cand["company"]
        try:
            rec = run_prospect(client, company, ICP, cand.get("hint", ""))
            consecutive = 0
            counts["ok"] += 1
            tier = rec.get("tier")
            if tier in counts:
                counts[tier] += 1
            log.info("  ✓ %s — tier %s (fit %s)", company, tier, rec.get("fit_score"))
        except Exception as e:
            rec = {"company": company, "error": friendly_api_error(e)}
            consecutive += 1
            counts["error"] += 1
            log.warning("  ✗ %s — %s", company, str(e).splitlines()[0][:120])
        db.insert_prospect(rec, run_id=run_id)
        db.bump_run_progress(run_id)
        if consecutive >= MAX_CONSECUTIVE_ERRORS:
            log.error("Aborting batch after %d consecutive failures — endpoint "
                      "likely down.", consecutive)
            break
        time.sleep(PACE_SECONDS)
    return counts


def run_once(location: str, count: int, forced_niche: str | None,
             dry_run: bool) -> int:
    try:
        client = make_client()
    except SystemExit as e:  # missing API key
        log.error("%s", e)
        return 1

    # 1. Choose a niche (also the first model call — a health gate).
    try:
        niche = forced_niche or pick_niche(client, location)
    except Exception as e:
        log.error("Niche step failed (endpoint down?): %s — skipping this run.",
                  friendly_api_error(e))
        return 0
    if not niche:
        log.error("No niche to work from; skipping.")
        return 0
    log.info("Niche: %s", niche)

    # 2. Discover companies (needs web search + model — the second health gate).
    try:
        candidates = discover_candidates(client, niche, ICP, count)
    except Exception as e:
        log.error("Discovery failed (endpoint down?): %s — skipping, nothing "
                  "written.", friendly_api_error(e))
        return 0
    if not candidates:
        log.info("Discovery found no companies for this niche; skipping.")
        return 0

    # 3. Drop companies already in the DB (and dups within the batch).
    known = existing_companies()
    seen, new = set(), []
    for c in candidates:
        key = c["company"].strip().lower()
        if key and key not in known and key not in seen:
            seen.add(key)
            new.append(c)
    log.info("Discovered %d, %d new after dedup.", len(candidates), len(new))
    if not new:
        log.info("Nothing new to research; skipping.")
        return 0
    if dry_run:
        for c in new:
            log.info("  [dry-run] would research: %s", c["company"])
        return 0

    # 4. Research + qualify, grouped under a run so the dashboard shows it.
    run_id = db.create_run("discover", niche, count)
    db.set_run_total(run_id, len(new))
    try:
        counts = research_batch(client, run_id, new)
        db.finish_run(run_id, "done")
    except Exception as e:  # unexpected — mark the run so it isn't stuck
        db.finish_run(run_id, "error", friendly_api_error(e))
        log.exception("Batch failed unexpectedly.")
        return 1

    log.info("Done: %d ok (A:%d B:%d C:%d DQ:%d), %d errored.",
             counts["ok"], counts["A"], counts["B"], counts["C"],
             counts["disqualified"], counts["error"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Populate the prospect DB, one batch.")
    ap.add_argument("--location",
                    default=os.environ.get("POPULATE_LOCATION", "Spain"),
                    help="Where to prospect (default: $POPULATE_LOCATION or 'Spain').")
    ap.add_argument("--count", type=int, default=15,
                    help="Companies to discover per batch (default: 15).")
    ap.add_argument("--niche", help="Research this exact niche, skip auto-suggest.")
    ap.add_argument("--log", default=str(ROOT / "populate.log"),
                    help="Log file (default: ./populate.log).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Suggest + discover only; persist nothing.")
    args = ap.parse_args()

    setup_logging(Path(args.log))

    lock = acquire_lock(ROOT / ".populate.lock")
    if lock is None:
        log.info("Another populate run is in progress; exiting.")
        return 0
    try:
        return run_once(args.location, args.count, args.niche, args.dry_run)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
