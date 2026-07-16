"""
Service layer: bridges the agent (agent.py) and persistence (db.py) so the CLI
(main.py) and the web server (server.py) share one code path.

- run_batch(): qualify a list of prospects, persisting each result and updating
  run progress as it goes. Same "one bad company doesn't kill the batch" behavior
  the CLI has always had.
- start_run_async(): kick off a discovery-and/or-qualification run on a background
  thread and return immediately with a run id the frontend can poll.
- friendly_api_error(): shared translation of common API failures (moved here from
  main.py so both entrypoints use it).
"""

import threading
from typing import Callable

import anthropic

from . import db
from .agent import discover_candidates, run_prospect
from .config import make_client, using_proxy
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
        return "Rate limited. Wait a moment and retry, or lower --count."
    if isinstance(e, anthropic.NotFoundError) and using_proxy():
        return (f"Endpoint or model not found on the proxy: {msg}\n"
                "Run `python check_setup.py` to list models it actually serves, "
                "then set PROSPECT_MODEL in .env.")
    return msg


def run_batch(client: anthropic.Anthropic, prospects: list[dict], icp: str = ICP,
              run_id: int | None = None, persist: bool = True,
              on_progress: Callable[[dict], None] | None = None) -> list[dict]:
    """Research + qualify each prospect, persisting results as they land.

    `prospects` is a list of {'company', 'hint'}. Returns the list of result
    records (qualified verdicts or {'company', 'error'}). If run_id is given, each
    completed company bumps that run's progress counter. Set persist=False to skip
    the SQLite store entirely (CLI --no-db). on_progress(record) is invoked after
    each company for callers that want to stream/log.
    """
    results = []
    for p in prospects:
        company = p["company"]
        try:
            record = run_prospect(client, company, icp, p.get("hint", ""))
        except Exception as e:  # one bad company shouldn't kill the batch
            record = {"company": company, "error": friendly_api_error(e)}
        if persist:
            db.insert_prospect(record, run_id=run_id)
            if run_id is not None:
                db.bump_run_progress(run_id)
        if on_progress:
            on_progress(record)
        results.append(record)
    return results


def _execute_run(client: anthropic.Anthropic, run_id: int, kind: str,
                 query: str, count: int) -> None:
    """Body of a run, executed on the worker thread. Persists via run_batch and
    marks the run done/error at the end."""
    try:
        if kind == "discover":
            candidates = discover_candidates(client, query, ICP, count)
            if not candidates:
                db.finish_run(run_id, "error",
                              "Discovery found no companies. Try a broader niche.")
                return
            prospects = [{"company": c["company"], "hint": c.get("hint", "")}
                         for c in candidates]
        else:  # 'companies' — query is a comma/newline separated list of names
            names = [n.strip() for n in query.replace("\n", ",").split(",")
                     if n.strip()]
            prospects = [{"company": n, "hint": ""} for n in names]

        if not prospects:
            db.finish_run(run_id, "error", "No companies to process.")
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
