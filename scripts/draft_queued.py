#!/usr/bin/env python3
"""
Draft + review outreach emails for prospects marked "to contact", unattended.

Designed for cron, like populate.py. Each invocation:

  1. Takes a lock so two runs never overlap.
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
import fcntl
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # find prospector/

from prospector import db
from prospector.config import load_env, make_client
from prospector.service import draft_email_for, friendly_api_error, review_email_for

log = logging.getLogger("draft_queued")

PACE_SECONDS = 2.0
MAX_CONSECUTIVE_ERRORS = 3


def setup_logging(log_path: Path) -> None:
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def acquire_lock(lock_path: Path):
    """Return an open, flock'd file handle, or None if another run holds it."""
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def run_once(limit: int, dry_run: bool) -> int:
    db.init_db()
    work = db.queued_needing_draft(limit)
    if not work:
        log.info("Queue empty — nothing to draft.")
        return 0
    log.info("%d queued prospect(s) to process.", len(work))
    if dry_run:
        for w in work:
            log.info("  [dry-run] would %s: %s",
                     "re-review" if w["drafted"] else "draft", w["company"])
        return 0

    try:
        client = make_client()
    except SystemExit as e:  # missing API key
        log.error("%s", e)
        return 1

    consecutive = 0
    for w in work:
        try:
            if w["drafted"]:
                email = review_email_for(w["id"], client=client)
            else:
                email = draft_email_for(w["id"], client=client)
        except Exception as e:
            consecutive += 1
            log.warning("  ✗ %s — %s", w["company"], friendly_api_error(e).splitlines()[0][:160])
            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                log.error("Aborting after %d consecutive failures.", consecutive)
                break
            continue

        review = email["review"]
        if review.get("error"):
            # The draft is stored; the review will be retried next run. Count it
            # toward the abort, since it usually means the Anthropic side is down.
            consecutive += 1
            log.warning("  ~ %s — drafted, review failed: %s", w["company"],
                        review["error"].splitlines()[0][:160])
        else:
            consecutive = 0
            contact = (email.get("contact") or {}).get("email") or "no contact"
            log.info("  ✓ %s — %d fix(es), %d rule(s) still broken, %s",
                     w["company"], len(review["issues"]), len(review["remaining"]),
                     contact)
        if consecutive >= MAX_CONSECUTIVE_ERRORS:
            log.error("Aborting after %d consecutive failures.", consecutive)
            break
        time.sleep(PACE_SECONDS)
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
    lock = acquire_lock(ROOT / ".draft_queued.lock")
    if lock is None:
        log.info("Another draft run is in progress; exiting.")
        return 0
    try:
        return run_once(args.limit, args.dry_run)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
