"""
Service layer: bridges the agent (agent.py) and persistence (db.py) for the web
server (server.py).

- run_batch(): qualify a list of prospects, persisting each result and updating
  run progress as it goes. One bad company doesn't kill the batch.
- start_run_async(): kick off a discovery-and/or-qualification run on a background
  thread and return immediately with a run id the frontend can poll.
- friendly_api_error(): translation of common API failures into actionable text.
"""

import threading

import anthropic

from . import db
from .agent import (
    discover_candidates,
    draft_outreach_email,
    find_contact,
    run_prospect,
    suggest_niches,
)
from .config import get_output_language, make_client, using_proxy
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
              run_id: int | None = None) -> list[dict]:
    """Research + qualify each prospect, persisting results as they land.

    `prospects` is a list of {'company', 'hint'}. Returns the list of result
    records (qualified verdicts or {'company', 'error'}). If run_id is given, each
    completed company bumps that run's progress counter.
    """
    results = []
    for p in prospects:
        company = p["company"]
        try:
            record = run_prospect(client, company, icp, p.get("hint", ""))
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
                 query: str, count: int) -> None:
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
        else:  # 'companies' — query is a comma/newline separated list of names
            names = [n.strip() for n in query.replace("\n", ",").split(",")
                     if n.strip()]
            prospects = [{"company": n, "hint": ""} for n in names]

        if not prospects:
            db.finish_run(run_id, "error", "No companies to process.")
            return

        # Skip companies already in the DB (by name or domain) so a re-run never
        # re-researches them.
        prospects = db.filter_unresearched(prospects)
        if not prospects:
            db.finish_run(run_id, "done")  # everything was already researched
            return

        db.set_run_total(run_id, len(prospects))
        run_batch(client, prospects, ICP, run_id=run_id)
        db.finish_run(run_id, "done")
    except Exception as e:  # discovery itself failed, or something unexpected
        db.finish_run(run_id, "error", friendly_api_error(e))


def start_run_async(kind: str, query: str, count: int = 10,
                    client: anthropic.Anthropic | None = None) -> int:
    """Create a run row and launch it on a daemon thread. Returns the run id
    immediately so the caller (HTTP handler) can respond and the frontend can poll
    GET /api/runs/{id} for progress."""
    if kind not in ("discover", "companies"):
        raise ValueError(f"unknown run kind: {kind!r}")
    client = client or make_client()
    run_id = db.create_run(kind, query, count)
    threading.Thread(
        target=_execute_run,
        args=(client, run_id, kind, query, count),
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


def draft_email_for(prospect_id: int, language: str | None = None,
                    client: anthropic.Anthropic | None = None) -> dict:
    """Draft an outreach email for one stored prospect. Returns {'subject','body',
    'language','contact'}.

    Synchronous. Drafts the email from the saved research, then looks up where to
    send it (the contact search only runs once — a stored contact is reused across
    regenerations). `language` is 'english' or 'spanish'; None follows the global
    config.OUTPUT_LANGUAGE. Raises LookupError if the prospect is gone, ValueError
    if it's a failed-research row.
    """
    rec = db.get_prospect(prospect_id)
    if rec is None:
        raise LookupError("prospect not found")
    if rec.get("error"):
        raise ValueError("This entry is a failed research record — there's nothing "
                         "to write an email from.")
    language = language or get_output_language()
    client = client or make_client()
    email = draft_outreach_email(client, rec, ICP, language)
    db.set_prospect_email(prospect_id, email.get("subject", ""),
                          email.get("body", ""), language)
    email["language"] = language
    email["contact"] = _resolve_contact(prospect_id, rec, client)
    return email


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
