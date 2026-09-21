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
    python scripts/populate.py --loop              # daemon: batch after batch (below)

Location defaults to $POPULATE_LOCATION or "Spain". Exit 0 on success or a clean
skip (locked / endpoint down / nothing new); 1 on an unexpected error.

--loop keeps running batches for as long as the LiteLLM key has more than
$POPULATE_BUDGET_FLOOR (default 5) left, checking before every company, then
idles and re-checks hourly in case the budget is reset or raised. While the proxy
is unreachable (VPN off) it idles too. Meant for the prospector-research user
service; SIGTERM finishes the current company, closes the run, and exits.
"""

import argparse
import fcntl
import logging
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # find prospector/

from prospector import db
from prospector.budget import llm_budget
from prospector.agent import discover_candidates, run_prospect, suggest_niches
from prospector.config import load_env, make_client
from prospector.icp import ICP
from prospector.service import categorize_run, friendly_api_error

log = logging.getLogger("populate")

PACE_SECONDS = 2.0          # gap between company research calls (be gentle)
MAX_CONSECUTIVE_ERRORS = 3  # abort the batch if this many in a row fail

# --loop pacing. A batch that researched something is followed quickly by the
# next; one that researched nothing (endpoint down, model denied, nothing new)
# backs off so a broken setup doesn't burn budget on niche suggestions.
LOOP_PAUSE_SECONDS = 60
LOOP_BACKOFF_SECONDS = 15 * 60
LOOP_OFFLINE_SECONDS = 5 * 60      # proxy unreachable — likely the VPN is off
LOOP_BUDGET_SECONDS = 60 * 60      # at/below the floor — wait for a reset
DEFAULT_BUDGET_FLOOR = 5.0

_stop = threading.Event()  # set by SIGTERM/SIGINT in --loop mode


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


def budget_state(floor: float) -> str:
    """'ok' when the key has more than `floor` left, 'low' at or below it,
    'offline' when the proxy can't be reached or doesn't report a capped budget."""
    b = llm_budget(force=True)
    if not b.get("available") or b.get("remaining") is None:
        return "offline"
    return "ok" if b["remaining"] > floor else "low"


def research_batch(client, run_id: int, candidates: list,
                   should_stop=None) -> dict:
    """Research + qualify each candidate, persisting as we go. Aborts early on a
    burst of consecutive failures (endpoint likely down), or when `should_stop()`
    turns true before a company."""
    counts = {"ok": 0, "error": 0, "A": 0, "B": 0, "C": 0, "disqualified": 0}
    consecutive = 0
    for cand in candidates:
        if should_stop and should_stop():
            log.info("Stopping the batch early (budget floor or shutdown).")
            break
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
        if cand.get("website"):  # keep the discovered domain for future dedup
            rec.setdefault("website", cand["website"])
        db.insert_prospect(rec, run_id=run_id)
        db.bump_run_progress(run_id)
        if consecutive >= MAX_CONSECUTIVE_ERRORS:
            log.error("Aborting batch after %d consecutive failures — endpoint "
                      "likely down.", consecutive)
            break
        time.sleep(PACE_SECONDS)
    return counts


def run_once(location: str, count: int, forced_niche: str | None,
             dry_run: bool, should_stop=None) -> tuple[int, int]:
    """One batch. Returns (exit code, companies researched)."""
    try:
        client = make_client()
    except SystemExit as e:  # missing API key
        log.error("%s", e)
        return 1, 0

    # 1. Choose a niche (also the first model call — a health gate).
    try:
        niche = forced_niche or pick_niche(client, location)
    except Exception as e:
        log.error("Niche step failed (endpoint down?): %s — skipping this run.",
                  friendly_api_error(e))
        return 0, 0
    if not niche:
        log.error("No niche to work from; skipping.")
        return 0, 0
    log.info("Niche: %s", niche)

    # 2. Discover companies (needs web search + model — the second health gate).
    try:
        candidates = discover_candidates(client, niche, ICP, count)
    except Exception as e:
        log.error("Discovery failed (endpoint down?): %s — skipping, nothing "
                  "written.", friendly_api_error(e))
        return 0, 0
    if not candidates:
        log.info("Discovery found no companies for this niche; skipping.")
        return 0, 0

    # 3. Drop companies already in the DB by name or domain (and dups in the batch).
    new = db.filter_unresearched(candidates)
    log.info("Discovered %d, %d new after dedup.", len(candidates), len(new))
    if not new:
        log.info("Nothing new to research; skipping.")
        return 0, 0
    if dry_run:
        for c in new:
            log.info("  [dry-run] would research: %s", c["company"])
        return 0, 0

    # 4. Research + qualify, grouped under a run so the dashboard shows it. File
    #    the run under a niche category (best-effort) so the dashboard groups these
    #    unattended runs by niche just like the ones launched from the browser.
    category_id = categorize_run(client, niche)
    run_id = db.create_run("discover", niche, count, category_id=category_id)
    db.set_run_total(run_id, len(new))
    try:
        counts = research_batch(client, run_id, new, should_stop)
        db.finish_run(run_id, "done")
    except Exception as e:  # unexpected — mark the run so it isn't stuck
        db.finish_run(run_id, "error", friendly_api_error(e))
        log.exception("Batch failed unexpectedly.")
        return 1, 0
    except BaseException:  # Ctrl-C / kill mid-company: don't leave it "running"
        db.finish_run(run_id, "error", "Interrupted before the batch finished.")
        raise

    log.info("Done: %d ok (A:%d B:%d C:%d DQ:%d), %d errored.",
             counts["ok"], counts["A"], counts["B"], counts["C"],
             counts["disqualified"], counts["error"])
    return 0, counts["ok"]


def run_loop(location: str, count: int, floor: float) -> int:
    """Batch after batch while the budget stays above `floor` (see module doc)."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: _stop.set())
    log.info("Research daemon up: %s, %d per batch, stops at %.2f left.",
             location, count, floor)

    def should_stop() -> bool:
        return _stop.is_set() or budget_state(floor) != "ok"

    last = None
    while not _stop.is_set():
        state = budget_state(floor)
        if state != last:  # log transitions, not every idle re-check
            if state == "offline":
                log.info("Proxy unreachable or no budget reported; idling.")
            elif state == "low":
                log.info("Budget at or below %.2f; idling until it's topped up.",
                         floor)
            last = state
        if state == "offline":
            _stop.wait(LOOP_OFFLINE_SECONDS)
            continue
        if state == "low":
            _stop.wait(LOOP_BUDGET_SECONDS)
            continue

        _, researched = run_once(location, count, None, False, should_stop)
        _stop.wait(LOOP_PAUSE_SECONDS if researched else LOOP_BACKOFF_SECONDS)
    log.info("Research daemon stopping.")
    return 0


def main() -> int:
    load_env()  # so $POPULATE_* in .env reach the defaults below
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
    ap.add_argument("--loop", action="store_true",
                    help="Keep running batches until the budget floor (daemon).")
    ap.add_argument("--budget-floor", type=float,
                    default=float(os.environ.get("POPULATE_BUDGET_FLOOR",
                                                 DEFAULT_BUDGET_FLOOR)),
                    help="--loop stops at this much budget left (default: "
                         "$POPULATE_BUDGET_FLOOR or 5).")
    args = ap.parse_args()

    setup_logging(Path(args.log))

    lock = acquire_lock(ROOT / ".populate.lock")
    if lock is None:
        log.info("Another populate run is in progress; exiting.")
        return 0
    try:
        if args.loop:
            return run_loop(args.location, args.count, args.budget_floor)
        return run_once(args.location, args.count, args.niche, args.dry_run)[0]
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
