"""
Service layer: bridges the agent (agent.py) and persistence (db.py) for the web
server (server.py).

- run_batch(): qualify a list of prospects, persisting each result and updating
  run progress as it goes. One bad company doesn't kill the batch.
- start_run_async(): kick off a discovery-and/or-qualification run on a background
  thread and return immediately with a run id the frontend can poll.
- friendly_api_error(): translation of common API failures into actionable text.
"""

import fcntl
import threading
import time
from datetime import datetime, timezone

import anthropic

from . import db, gmailer
from .agent import (
    categorize_niche,
    discover_candidates,
    draft_email_body,
    draft_email_subject,
    draft_outreach_email,
    find_contact,
    review_outreach_email,
    run_prospect,
    suggest_niches,
)
from .config import (
    ROOT,
    email_review_enabled,
    get_output_language,
    get_review_model,
    make_client,
    using_proxy,
)
from .email_lint import lint_email
from .icp import ICP


def friendly_api_error(e: Exception) -> str:
    """Turn common API failures into something actionable instead of a traceback."""
    msg = str(e)
    if isinstance(e, anthropic.APIConnectionError):
        cause = repr(e.__cause__ or "")
        if "CERTIFICATE_VERIFY_FAILED" in cause:
            return ("TLS verification failed — this network likely runs an intercepting "
                    "proxy. Point Python at your system CA bundle:\n"
                    "  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt")
        return f"Could not reach the API (check your network): {msg}"
    if isinstance(e, anthropic.AuthenticationError):
        return "ANTHROPIC_API_KEY is invalid or revoked. Check .env / console.anthropic.com."
    if "credit balance is too low" in msg:
        return ("Your Anthropic account is out of credits. Add credits at\n"
                "  https://console.anthropic.com -> Plans & Billing\n"
                "(Or point ANTHROPIC_BASE_URL at your LiteLLM instance instead.)")
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited. Wait a moment and retry, or use a smaller count."
    if isinstance(e, anthropic.NotFoundError) and using_proxy():
        return (f"Endpoint or model not found on the proxy: {msg}\n"
                "Run `python scripts/check_setup.py` to list models it actually "
                "serves, then set PROSPECT_MODEL in .env.")
    return msg


def run_batch(client: anthropic.Anthropic, prospects: list[dict], icp: str = ICP,
              run_id: int | None = None, thorough: bool = False) -> list[dict]:
    """Research + qualify each prospect, persisting results as they land.

    `prospects` is a list of {'company', 'hint'}. Returns the list of result
    records (qualified verdicts or {'company', 'error'}). If run_id is given, each
    completed company bumps that run's progress counter. `thorough` deepens the
    research stage (see agent.run_prospect).
    """
    results = []
    for p in prospects:
        company = p["company"]
        try:
            record = run_prospect(client, company, icp, p.get("hint", ""),
                                  thorough=thorough)
        except Exception as e:  # one bad company shouldn't kill the batch
            record = {"company": company, "error": friendly_api_error(e)}
        if p.get("website"):  # keep the discovered domain for future dedup
            record.setdefault("website", p["website"])
        db.insert_prospect(record, run_id=run_id)
        if run_id is not None:
            db.bump_run_progress(run_id)
        results.append(record)
    return results


MAX_DISCOVERY_ATTEMPTS = 4     # times to re-ask discovery for fresh names
DISCOVERY_OVERSHOOT = 5        # ask for a few extra so filtering still leaves enough
MAX_EXCLUDE_HINTS = 60         # cap on the 'already have' list sent to the model


def _discover_unresearched(client: anthropic.Anthropic, niche: str,
                           count: int) -> list[dict]:
    """Discover up to `count` companies we haven't researched yet.

    A single discovery pass tends to re-find companies already in the DB, so a
    repeated search on the same niche can yield nothing new. This retries
    discovery, each pass telling the model which names to avoid (the ones we
    already have plus those found so far), until it has `count` fresh companies or
    stops making progress. Returns candidate dicts — possibly fewer than `count`,
    or empty if the niche is genuinely exhausted.
    """
    avoid = db.researched_company_names(limit=MAX_EXCLUDE_HINTS)
    fresh: list[dict] = []
    for _ in range(MAX_DISCOVERY_ATTEMPTS):
        need = count - len(fresh)
        if need <= 0:
            break
        exclude = avoid + [c["company"] for c in fresh]
        batch = discover_candidates(client, niche, ICP, need + DISCOVERY_OVERSHOOT,
                                    exclude=exclude)
        if not batch:
            break  # the model genuinely found nothing — stop hitting the API
        # A pass that only re-finds known companies isn't a dead end: the growing
        # exclusion list steers the next pass elsewhere, so keep going until we hit
        # the target or run out of attempts.
        fresh = db.filter_unresearched(fresh + batch)  # drop known + dups, keep order
    return fresh[:count]


def _execute_run(client: anthropic.Anthropic, run_id: int, kind: str,
                 query: str, count: int, thorough: bool = False) -> None:
    """Body of a run, executed on the worker thread. Persists via run_batch and
    marks the run done/error at the end."""
    try:
        if kind == "discover":
            candidates = _discover_unresearched(client, query, count)
            if not candidates:
                db.finish_run(run_id, "error",
                              "No new companies found for this niche — you may have "
                              "already researched the ones out there. Try a "
                              "different or broader niche.")
                return
            prospects = [{"company": c["company"], "hint": c.get("hint", ""),
                          "website": c.get("website", "")} for c in candidates]
            # Discovery skips companies already in the DB (by name or domain) so a
            # repeated search never re-researches them. The Research-companies tab
            # deliberately does NOT filter — naming a known company re-researches it.
            prospects = db.filter_unresearched(prospects)
            if not prospects:
                db.finish_run(run_id, "done")  # everything was already researched
                return
        else:  # 'companies' — query is a comma/newline separated list of names
            names = [n.strip() for n in query.replace("\n", ",").split(",")
                     if n.strip()]
            prospects = [{"company": n, "hint": ""} for n in names]

        if not prospects:
            db.finish_run(run_id, "error", "No companies to process.")
            return

        db.set_run_total(run_id, len(prospects))
        run_batch(client, prospects, ICP, run_id=run_id, thorough=thorough)
        db.finish_run(run_id, "done")
    except Exception as e:  # discovery itself failed, or something unexpected
        db.finish_run(run_id, "error", friendly_api_error(e))


def categorize_run(client: anthropic.Anthropic, query: str) -> int | None:
    """Resolve a discovery query to a niche category id, creating one if needed.

    Best-effort: a categorization hiccup returns None (the run just starts
    uncategorized and can be filed later) rather than blocking the search. Only
    discovery runs are categorized — the companies tab stays run-grouped.
    """
    try:
        existing = [c["name"] for c in db.list_categories()]
        name = categorize_niche(client, query, existing)
        if not name:
            return None
        return db.find_or_create_category(name)["id"]
    except Exception:
        return None


def start_run_async(kind: str, query: str, count: int = 10, thorough: bool = False,
                    client: anthropic.Anthropic | None = None) -> int:
    """Create a run row and launch it on a daemon thread. Returns the run id
    immediately so the caller (HTTP handler) can respond and the frontend can poll
    GET /api/runs/{id} for progress. Discovery runs are filed under a niche
    category first (a quick reasoning call) so results group by niche, not run."""
    if kind not in ("discover", "companies"):
        raise ValueError(f"unknown run kind: {kind!r}")
    client = client or make_client()
    category_id = categorize_run(client, query) if kind == "discover" else None
    run_id = db.create_run(kind, query, count, category_id=category_id)
    threading.Thread(
        target=_execute_run,
        args=(client, run_id, kind, query, count, thorough),
        daemon=True,
    ).start()
    return run_id


def suggest_niches_for(location: str, count: int = 8,
                       client: anthropic.Anthropic | None = None) -> list:
    """Suggest B2B niches for a location, ready to feed into a discovery run.

    Synchronous and quick (a single reasoning call) — unlike a research run, so
    the HTTP handler can return the niches directly.
    """
    client = client or make_client()
    return suggest_niches(client, location, ICP, count)


def next_draft_language(rec: dict) -> str:
    """The language a prospect's next draft is written in: the drawer's pick
    (draft_lang), else the current draft's language, else OUTPUT_LANGUAGE."""
    return (rec.get("draft_lang") or (rec.get("email") or {}).get("language")
            or get_output_language())


def draft_email_for(prospect_id: int, language: str | None = None,
                    client: anthropic.Anthropic | None = None) -> dict:
    """Draft, review, and store an outreach email for one prospect. Returns
    {'subject','body','language','review','contact'}.

    Synchronous. Drafts the email from the saved research (EMAIL_PROVIDER), runs
    the playbook checks + Claude review over it (see _review_draft), stores the
    reviewed text, then looks up where to send it (the contact search only runs
    once — a stored contact is reused across regenerations). `language` is
    'english' or 'spanish'; None uses next_draft_language.
    Raises LookupError if the prospect is gone, ValueError if it's a
    failed-research row.
    """
    rec = db.get_prospect(prospect_id)
    if rec is None:
        raise LookupError("prospect not found")
    if rec.get("error"):
        raise ValueError("This entry is a failed research record — there's nothing "
                         "to write an email from.")
    language = language or next_draft_language(rec)
    client = client or make_client()
    draft = draft_outreach_email(client, rec, ICP, language)
    email = _store_reviewed(prospect_id, rec, draft, language, client)
    email["contact"] = _resolve_contact(prospect_id, rec, client)
    return email


def save_email_edits(prospect_id: int, subject: str, body: str) -> dict:
    """Store hand edits to a prospect's drafted email (the drawer's autosave).

    Like redraft_subject_for, refreshes the stored review's `remaining` rule
    findings so the Pipeline status tracks the edited text. Returns
    {'subject', 'body', 'remaining'}. Raises LookupError if the prospect is
    gone, ValueError if there's no draft yet.
    """
    rec = db.get_prospect(prospect_id)
    if rec is None:
        raise LookupError("prospect not found")
    email = rec.get("email")
    if not email:
        raise ValueError("No drafted email yet — draft one first.")
    db.set_email_text(prospect_id, subject, body)
    remaining = lint_email(subject, body, email.get("language") or get_output_language())
    review = email.get("review")
    if review:
        db.set_email_review(prospect_id, {**review, "remaining": remaining})
    return {"subject": subject, "body": body, "remaining": remaining}


def redraft_subject_for(prospect_id: int, body: str | None = None,
                        subject: str | None = None,
                        client: anthropic.Anthropic | None = None) -> dict:
    """Write a new subject for a prospect's drafted email and store it.

    `body`/`subject` are what's in the editor (unsent edits included), else the
    stored draft. Only the subject is saved; the stored review's `remaining`
    rule findings are refreshed so the queue status tracks the new subject.
    Returns {'subject', 'remaining'}. Raises LookupError if the prospect is
    gone, ValueError if there's no draft yet.
    """
    rec = db.get_prospect(prospect_id)
    if rec is None:
        raise LookupError("prospect not found")
    email = rec.get("email")
    if not email or not (body or email.get("body")):
        raise ValueError("No drafted email yet — draft one first.")
    body = body if body is not None else email.get("body", "")
    current = subject if subject is not None else email.get("subject", "")
    language = email.get("language") or get_output_language()
    new = draft_email_subject(client or make_client(), rec, body, ICP, language, current)
    db.set_email_subject(prospect_id, new)
    remaining = lint_email(new, email.get("body", ""), language)
    review = email.get("review")
    if review:
        db.set_email_review(prospect_id, {**review, "remaining": remaining})
    return {"subject": new, "remaining": remaining}


def redraft_body_for(prospect_id: int, subject: str | None = None,
                     body: str | None = None,
                     client: anthropic.Anthropic | None = None) -> dict:
    """Write a new body for a prospect's drafted email, keep its subject, store both.

    `subject`/`body` are what's in the editor (unsaved edits included), else the
    stored draft; the body is passed so the new one differs from it. The new
    body gets the same playbook checks + review as a full draft, but only the
    body is taken from the review — the subject stays as given. Returns
    {'subject', 'body', 'review'}. Raises LookupError if the prospect is gone,
    ValueError if there's no draft yet.
    """
    rec = db.get_prospect(prospect_id)
    if rec is None:
        raise LookupError("prospect not found")
    email = rec.get("email")
    if not email:
        raise ValueError("No drafted email yet — draft one first.")
    subject = subject if subject is not None else email.get("subject", "")
    current = body if body is not None else email.get("body", "")
    language = email.get("language") or get_output_language()
    client = client or make_client()
    new = draft_email_body(client, rec, subject, ICP, language, current)
    final, review = _review_draft(client, rec, {"subject": subject, "body": new}, language)
    review["remaining"] = lint_email(subject, final["body"], language)
    db.set_email_text(prospect_id, subject, final["body"])
    db.set_email_review(prospect_id, review)
    return {"subject": subject, "body": final["body"], "review": review}


def review_email_for(prospect_id: int,
                     client: anthropic.Anthropic | None = None) -> dict:
    """Re-run the review on a prospect's stored draft, without redrafting.

    Reviews the ORIGINAL draft kept by the last review when there is one (so a
    retry after a failed review starts from what the drafter wrote), else the
    stored text. Backs the auto-draft retry for reviews that errored. Returns the
    same shape as draft_email_for, minus 'contact'.
    """
    rec = db.get_prospect(prospect_id)
    if rec is None:
        raise LookupError("prospect not found")
    email = rec.get("email")
    if not email:
        raise ValueError("No drafted email to review — generate one first.")
    original = (email.get("review") or {}).get("original") or {
        "subject": email.get("subject", ""), "body": email.get("body", "")}
    language = email.get("language") or get_output_language()
    return _store_reviewed(prospect_id, rec, original, language,
                           client or make_client())


def _store_reviewed(prospect_id: int, rec: dict, draft: dict, language: str,
                    client: anthropic.Anthropic) -> dict:
    """Review a draft and persist the result (final text + review record)."""
    final, review = _review_draft(client, rec, draft, language)
    db.set_prospect_email(prospect_id, final["subject"], final["body"], language)
    db.set_email_review(prospect_id, review)
    return {**final, "language": language, "review": review}


def _review_draft(client: anthropic.Anthropic, rec: dict, draft: dict,
                  language: str) -> tuple[dict, dict]:
    """Run the playbook checks and the Claude review over one draft.

    Returns (final {'subject','body'}, review record). The record keeps the
    drafter's original text, the deterministic findings before (`lint`) and after
    (`remaining`) review, and Claude's fixes (`issues`). A review failure never
    loses the draft: it's stored unreviewed with `error` set, so the queue shows it
    as needing attention and the auto-draft job retries just the review.
    """
    original = {"subject": draft.get("subject", ""), "body": draft.get("body", "")}
    lint = lint_email(original["subject"], original["body"], language)
    review = {
        "original": original, "lint": lint, "issues": [], "remaining": lint,
        "changed": False, "model": None, "error": None,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not email_review_enabled():
        review["skipped"] = True
        return original, review
    try:
        out = review_outreach_email(client, rec, original, ICP, language, lint)
    except Exception as e:  # keep the draft; mark it unreviewed
        review["error"] = friendly_api_error(e)
        return original, review
    final = {"subject": out["subject"], "body": out["body"]}
    review.update(
        issues=out["issues"], model=get_review_model(),
        remaining=lint_email(final["subject"], final["body"], language),
        changed=final != original,
    )
    return final, review


def _resolve_contact(prospect_id: int, rec: dict,
                     client: anthropic.Anthropic) -> dict | None:
    """Where to send the email. Reuse a stored contact, else search for one.

    Best-effort: a lookup failure never breaks email generation — it just means
    we return no contact yet. Retrying is a separate, explicit action
    (find_contact_for / the "Find contact" button).
    """
    existing = (rec.get("email") or {}).get("contact")
    if existing and existing.get("email"):
        return existing
    try:
        found = find_contact(client, rec)
    except Exception:
        return existing
    return _persist_found(prospect_id, found) if found else existing


def find_contact_for(prospect_id: int,
                     client: anthropic.Anthropic | None = None) -> dict | None:
    """Search the web for where to send outreach and store it. Returns the contact
    dict, or None if no real address was found.

    Unlike the search folded into email generation, this always runs a fresh
    search and lets errors propagate — it backs the on-demand "Find contact"
    button, so the caller can surface a failure and let the user retry.
    """
    rec = db.get_prospect(prospect_id)
    if rec is None:
        raise LookupError("prospect not found")
    found = find_contact(client or make_client(), rec)
    return _persist_found(prospect_id, found) if found else None


def _persist_found(prospect_id: int, found: dict) -> dict:
    """Store a find_contact() result, normalizing blank fields to NULL."""
    return db.set_prospect_contact(prospect_id, found["email"],
                                   found.get("phone") or None,
                                   found.get("website") or None,
                                   found.get("source_url") or None)


# --- Auto-draft the "to contact" queue ----------------------------------------
# One job, two triggers: scripts/draft_queued.py (cron / terminal) and the
# dashboard's "Draft now" button (start_draft_queued_async). Both take the same
# file lock, so a cron run and a button press never draft the same queue twice.

DRAFT_LOCK = ROOT / ".draft_queued.lock"
DRAFT_PACE_SECONDS = 2.0          # gap between prospects (be gentle)
DRAFT_MAX_CONSECUTIVE_ERRORS = 3  # abort the batch — the endpoint is likely down

# Progress of the dashboard-started job, polled by GET /api/outreach/draft-status.
# Only this process's job is tracked; a cron run shows up as the lock being held.
_draft_job: dict = {"running": False}
_draft_job_guard = threading.Lock()


def acquire_draft_lock():
    """Return an open, flock'd file handle, or None if a draft run holds it."""
    fh = open(DRAFT_LOCK, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def release_draft_lock(fh) -> None:
    fcntl.flock(fh, fcntl.LOCK_UN)
    fh.close()


def draft_queued(limit: int | None = 10, client: anthropic.Anthropic | None = None,
                 log=None, on_progress=None, pace: float = DRAFT_PACE_SECONDS,
                 ids: list[int] | None = None) -> dict:
    """Draft + review emails for up to `limit` queued prospects. Caller holds the lock.

    Never-drafted prospects get a full draft (draft_email_for); ones whose review
    failed last time only get the review re-run (review_email_for). limit=None
    drafts until nothing is left: after each pass it re-reads the queue (catching
    prospects marked meanwhile), skipping ones already tried this run so a
    failing prospect isn't retried in a loop. `log(level, msg)` receives one line
    per prospect; `on_progress(counts, current)` is called before each prospect
    and once at the end with current=None. A burst of consecutive failures
    (including failed reviews — usually the Anthropic side being down) aborts the
    run. `ids` drafts exactly those prospects instead (one pass, full redraft
    each — see db.prospects_to_draft), ignoring `limit`. Returns {'total','done',
    'ok','blocked','failed','aborted'}; 'blocked' drafts were stored but failed
    review or rule checks.
    """
    log = log or (lambda level, msg: None)
    tried: set[int] = set()

    def next_work() -> list[dict]:
        if ids is not None:
            return [] if tried else db.prospects_to_draft(ids)
        if limit is not None:
            return db.queued_needing_draft(limit)
        return [w for w in db.queued_needing_draft(None) if w["id"] not in tried]

    work = next_work()
    counts = {"total": len(work), "done": 0, "ok": 0, "blocked": 0, "failed": 0,
              "aborted": False}
    if not work:
        log("info", "Queue empty — nothing to draft.")
        return counts
    client = client or make_client()
    consecutive = 0
    while work:
        for i, w in enumerate(work):
            tried.add(w["id"])
            if on_progress:
                on_progress(counts, w["company"])
            try:
                if w["drafted"]:
                    email = review_email_for(w["id"], client=client)
                else:
                    email = draft_email_for(w["id"], client=client)
            except Exception as e:
                consecutive += 1
                counts["failed"] += 1
                log("warning", f"✗ {w['company']} — {friendly_api_error(e).splitlines()[0][:160]}")
            else:
                review = email["review"]
                if review.get("error"):
                    consecutive += 1
                    counts["blocked"] += 1
                    log("warning", f"~ {w['company']} — drafted, review failed: "
                                   f"{review['error'].splitlines()[0][:160]}")
                else:
                    consecutive = 0
                    counts["blocked" if review["remaining"] else "ok"] += 1
                    contact = (email.get("contact") or {}).get("email") or "no contact"
                    log("info", f"✓ {w['company']} — {len(review['issues'])} fix(es), "
                                f"{len(review['remaining'])} rule(s) still broken, {contact}")
            counts["done"] += 1
            if consecutive >= DRAFT_MAX_CONSECUTIVE_ERRORS:
                counts["aborted"] = True
                log("error", f"Aborting after {consecutive} consecutive failures.")
                break
            if i < len(work) - 1:
                time.sleep(pace)
        if counts["aborted"] or limit is not None:
            break
        work = next_work()          # anything marked while we were drafting
        if work:
            counts["total"] += len(work)
            time.sleep(pace)
    if on_progress:
        on_progress(counts, None)
    return counts


def start_draft_queued_async(limit: int | None = None,
                             client: anthropic.Anthropic | None = None,
                             ids: list[int] | None = None) -> dict:
    """Run draft_queued on a background thread for the dashboard — the whole
    queue, or just `ids` when given ("Draft selected"). Returns the
    initial job status. Raises RuntimeError if a draft run (this process's, or a
    cron/terminal one) is already going, SystemExit if the API key is missing."""
    with _draft_job_guard:
        if _draft_job.get("running"):
            raise RuntimeError("A draft run is already in progress.")
        if _send_job.get("running"):
            raise RuntimeError("Emails are being sent — wait for that to finish.")
        lock = acquire_draft_lock()
        if lock is None:
            raise RuntimeError("A draft run (cron or terminal) is already in progress.")
        try:
            client = client or make_client()
        except SystemExit:
            release_draft_lock(lock)
            raise
        _draft_job.clear()
        _draft_job.update(running=True, total=0, done=0, ok=0, blocked=0, failed=0,
                          aborted=False, current=None, error=None,
                          started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          finished_at=None)

    def progress(counts, current):
        _draft_job.update(counts, current=current)

    def work():
        try:
            draft_queued(limit, client=client, on_progress=progress, ids=ids)
        except Exception as e:  # unexpected — surface it instead of a stuck job
            _draft_job["error"] = friendly_api_error(e)
        finally:
            release_draft_lock(lock)
            _draft_job.update(running=False, current=None,
                              finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # Snapshot before starting: a fast job (empty queue) can finish before a
    # post-start read, making the "initial" status look already done.
    status = draft_job_status()
    threading.Thread(target=work, daemon=True).start()
    return status


def draft_job_status() -> dict:
    """The dashboard job's progress, plus whether another run holds the lock."""
    status = dict(_draft_job)
    if not status.get("running"):
        lock = acquire_draft_lock()
        status["locked_elsewhere"] = lock is None
        if lock is not None:
            release_draft_lock(lock)
    return status


# --- Gmail outreach: send + reply tracking -----------------------------------

def _ms_epoch_to_iso(ms: str | None) -> str:
    """Gmail's internalDate (ms since epoch, as a string) -> our UTC ISO format.

    Falls back to 'now' if Gmail ever hands back something unparseable, so a
    detected reply is never dropped for want of a timestamp.
    """
    try:
        return (datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
                .isoformat(timespec="seconds"))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


def send_outreach(prospect_id: int, subject: str | None = None,
                  body: str | None = None) -> dict:
    """Send a prospect's outreach email via Gmail and record the send.

    subject/body override the stored draft (the user's in-editor edits); when
    given they're persisted before sending so the stored copy matches what went
    out. Uses the stored contact address. Returns {'id', 'sent_at', 'thread_id'}.
    Raises LookupError if the prospect is gone, ValueError if there's no draft or
    no address, and gmailer.GmailNotConfigured if the account isn't connected.
    """
    rec = db.get_prospect(prospect_id)
    if rec is None:
        raise LookupError("prospect not found")
    email = rec.get("email")
    if not email:
        raise ValueError("No drafted email to send — generate one first.")
    contact = email.get("contact") or {}
    to = contact.get("email")
    if not to:
        raise ValueError("No contact address — use “Find contact” first.")

    subject = subject if subject is not None else email.get("subject", "")
    body = body if body is not None else email.get("body", "")
    if not body.strip():
        raise ValueError("The email body is empty — nothing to send.")
    # Persist the edited text so the stored draft matches what we actually send.
    db.set_prospect_email(prospect_id, subject, body,
                          email.get("language") or get_output_language())

    sent = gmailer.send_email(to, subject, body)
    db.mark_sent(prospect_id, sent["message_id"], sent["thread_id"],
                 subject=subject, body=body, contact_email=to)
    updated = db.get_prospect(prospect_id)
    return {
        "id": prospect_id,
        "sent_at": (updated.get("email") or {}).get("sent_at"),
        "thread_id": sent["thread_id"],
    }


SEND_PACE_SECONDS = 2.0          # gap between sends (be gentle with Gmail)
SEND_MAX_CONSECUTIVE_ERRORS = 3  # abort the batch — Gmail is likely down

# Progress of the dashboard's "Send all" job, polled by GET /api/outreach/send-status.
_send_job: dict = {"running": False}


def sendable(ids: list[int] | None = None) -> list[dict]:
    """The "Send all" worklist: pipeline rows whose status is 'ready' (reviewed,
    no broken rules, has a contact), limited to `ids` when given. Same order as
    the Pipeline table. Returns [{'id', 'company'}]."""
    wanted = set(ids) if ids is not None else None
    return [{"id": r["id"], "company": r["company"]} for r in db.active_contacts()
            if r["status"] == "ready" and (wanted is None or r["id"] in wanted)]


def send_ready(ids: list[int] | None = None, on_progress=None,
               pace: float = SEND_PACE_SECONDS) -> dict:
    """Send the stored draft of every ready prospect (or just the ready ones of
    `ids`) via Gmail, one by one. A prospect sent meanwhile (e.g. from the
    drawer) is skipped, never sent twice. `on_progress(counts, current)` is
    called before each send and once at the end with current=None. Stops after
    SEND_MAX_CONSECUTIVE_ERRORS failures in a row. Returns {'total', 'done',
    'sent', 'skipped', 'failed', 'aborted', 'last_error'}.
    """
    work = sendable(ids)
    counts = {"total": len(work), "done": 0, "sent": 0, "skipped": 0, "failed": 0,
              "aborted": False, "last_error": None}
    consecutive = 0
    for i, w in enumerate(work):
        if on_progress:
            on_progress(counts, w["company"])
        rec = db.get_prospect(w["id"])
        if rec is None or (rec.get("email") or {}).get("sent_at"):
            counts["skipped"] += 1
        else:
            try:
                send_outreach(w["id"])
            except Exception as e:
                consecutive += 1
                counts["failed"] += 1
                counts["last_error"] = f"{w['company']}: {friendly_api_error(e).splitlines()[0][:160]}"
            else:
                consecutive = 0
                counts["sent"] += 1
        counts["done"] += 1
        if consecutive >= SEND_MAX_CONSECUTIVE_ERRORS:
            counts["aborted"] = True
            break
        if i < len(work) - 1:
            time.sleep(pace)
    if on_progress:
        on_progress(counts, None)
    return counts


def start_send_ready_async(ids: list[int] | None = None) -> dict:
    """Run send_ready on a background thread for the dashboard's "Send all".
    Returns the initial job status. Raises RuntimeError if a send or draft job
    is already running, gmailer.GmailNotConfigured if Gmail isn't connected."""
    with _draft_job_guard:
        if _send_job.get("running"):
            raise RuntimeError("Emails are already being sent.")
        if _draft_job.get("running"):
            raise RuntimeError("A draft run is in progress — wait for it to finish.")
        gmailer.ensure_authorized()
        _send_job.clear()
        _send_job.update(running=True, total=0, done=0, sent=0, skipped=0, failed=0,
                         aborted=False, last_error=None, current=None, error=None,
                         started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         finished_at=None)

    def progress(counts, current):
        _send_job.update(counts, current=current)

    def work():
        try:
            send_ready(ids, on_progress=progress)
        except Exception as e:  # unexpected — surface it instead of a stuck job
            _send_job["error"] = friendly_api_error(e)
        finally:
            _send_job.update(running=False, current=None,
                             finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    status = send_job_status()   # snapshot first, as in start_draft_queued_async
    threading.Thread(target=work, daemon=True).start()
    return status


def send_job_status() -> dict:
    """The dashboard "Send all" job's progress."""
    return dict(_send_job)


def refresh_replies() -> dict:
    """Poll every sent-but-unanswered thread for a reply, recording any found.

    Returns {'checked', 'new_replies'}. Raises gmailer.GmailNotConfigured if the
    account isn't connected. A per-thread lookup failure is swallowed so one bad
    thread doesn't abort the sweep.
    """
    gmailer.ensure_authorized()       # fail clearly rather than silently no-op
    worklist = db.sent_awaiting_reply()
    me = gmailer.my_addresses()       # resolve once; reused for every thread
    new_replies = 0
    for item in worklist:
        try:
            reply_ms = gmailer.check_reply(item["thread_id"], me)
        except Exception:
            continue
        if reply_ms:
            db.mark_replied(item["id"], _ms_epoch_to_iso(reply_ms))
            new_replies += 1
    return {"checked": len(worklist), "new_replies": new_replies}
