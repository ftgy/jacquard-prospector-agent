#!/usr/bin/env python3
"""
Draft + review outreach emails for prospects marked "to contact", unattended.

Designed for cron, like populate.py; the dashboard's "Draft now" button runs the
same job (service.draft_queued) under the same lock. Each invocation:

  1. Takes the draft lock so two runs (cron, terminal, or button) never overlap.
  2. Picks up to --limit queued, unsent prospects that have no draft yet, or whose
     review failed last time (those only get the review re-run, not a new draft).
  3. Drafts (EMAIL_PROVIDER), checks + reviews with Claude, finds a contact, and
     stores it all — the dashboard's Outreach tab shows the result for approval.

Nothing is sent. A burst of consecutive failures aborts the batch, so an endpoint
that's down doesn't burn through the queue.

Usage:
    python scripts/draft_queued.py              # up to 10 prospects
    python scripts/draft_queued.py --limit 3
    python scripts/draft_queued.py --dry-run    # list the worklist, call nothing

Exit 0 on success or a clean skip (locked / empty queue); 1 on setup errors.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # find prospector/

from prospector import db
from prospector.config import load_env
from prospector.service import acquire_draft_lock, draft_queued, release_draft_lock

log = logging.getLogger("draft_queued")


def setup_logging(log_path: Path) -> None:
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def run_once(limit: int, dry_run: bool) -> int:
    db.init_db()
    if dry_run:
        work = db.queued_needing_draft(limit)
        log.info("%d queued prospect(s) to process.", len(work))
        for w in work:
            log.info("  [dry-run] would %s: %s",
                     "re-review" if w["drafted"] else "draft", w["company"])
        return 0
    try:
        counts = draft_queued(limit, log=lambda level, msg: getattr(log, level)("  " + msg))
    except SystemExit as e:  # missing API key
        log.error("%s", e)
        return 1
    if counts["total"]:
        log.info("Done: %d ok, %d blocked, %d failed%s.", counts["ok"],
                 counts["blocked"], counts["failed"],
                 " (aborted)" if counts["aborted"] else "")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Draft emails for queued prospects.")
    ap.add_argument("--limit", type=int, default=10,
                    help="Max prospects per run (default: 10).")
    ap.add_argument("--log", default=str(ROOT / "draft_queued.log"),
                    help="Log file (default: ./draft_queued.log).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the worklist; call no model.")
    args = ap.parse_args()

    load_env()
    setup_logging(Path(args.log))
    lock = acquire_draft_lock()
    if lock is None:
        log.info("Another draft run is in progress; exiting.")
        return 0
    try:
        return run_once(args.limit, args.dry_run)
    finally:
        release_draft_lock(lock)


if __name__ == "__main__":
    sys.exit(main())
