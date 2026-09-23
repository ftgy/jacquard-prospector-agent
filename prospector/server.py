#!/usr/bin/env python3
"""
FastAPI backend for the prospector dashboard.

Serves a JSON API over the SQLite store (db.py) plus the single-page dashboard
(static/index.html). Research runs launched from the browser execute on background
threads (service.start_run_async); the page polls GET /api/runs/{id} for progress.

Run it:
  python server.py                 # http://localhost:8000
  uvicorn server:app --reload      # dev autoreload
"""

import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .config import (
    describe_target,
    get_auto_draft_buffer,
    get_auto_queue_interval,
    get_auto_queue_target,
    load_env,
    make_client,
    scheduler_enabled,
)

HERE = Path(__file__).parent
STATIC = HERE / "static"

# Loose sanity check on a hand-typed address — enough to catch a typo'd or
# half-pasted one, not a spec-complete validation.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_env()
    db.init_db()
    # Fail fast & loud if the API key/client is misconfigured, but don't block
    # read-only browsing of existing data — just warn.
    try:
        make_client()
        print(f"Prospector ready — {describe_target()}")
    except SystemExit as e:
        print(f"[warn] API client not configured: {e}\n"
              "       Browsing works; launching new runs will fail until fixed.")
    from . import service
    target = get_auto_queue_target()
    if target and scheduler_enabled():
        buffer = get_auto_draft_buffer()
        service.start_auto_queue(target, get_auto_queue_interval(), buffer)
        print(f"Auto-queue on: topping the scheduler up to {target} "
              f"every {get_auto_queue_interval():g} min, {buffer} ready in reserve")
    yield
    service.stop_auto_queue()


app = FastAPI(title="Prospector", lifespan=lifespan)


# --- API models --------------------------------------------------------------

class RunRequest(BaseModel):
    kind: str = Field(..., pattern="^(discover|companies)$")
    query: str = Field(..., min_length=1)
    count: int = Field(10, ge=1, le=50)
    thorough: bool = False  # deeper research pass (Research-companies tab opt-in)


class NicheRequest(BaseModel):
    location: str = Field(..., min_length=1)
    count: int = Field(8, ge=1, le=20)


class NotesRequest(BaseModel):
    notes: str = Field("", max_length=5000)


class CategoryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class RunCategoryRequest(BaseModel):
    # Refile a run: give an existing category_id (null un-files), or a name to
    # find/create. `name` wins when both are present.
    category_id: int | None = None
    name: str | None = Field(None, max_length=120)


class EmailRequest(BaseModel):
    # Omitted / null -> follow the global config.OUTPUT_LANGUAGE.
    language: str | None = Field(None, pattern="^(english|spanish)$")


class ApprovalRequest(BaseModel):
    approved: bool


class LanguageRequest(BaseModel):
    language: Literal["english", "spanish"]


class QueueRequest(BaseModel):
    queued: bool


class DraftRequest(BaseModel):
    # Prospect ids to draft ("Draft selected") or queue ("Queue selected").
    # Omitted -> the whole queue / every ready draft.
    ids: list[int] | None = Field(None, max_length=500)


class EmailEditRequest(BaseModel):
    # The drawer's edited subject/body (autosaved).
    subject: str = Field(..., max_length=500)
    body: str = Field(..., max_length=20000)


class SendRequest(BaseModel):
    # The edited subject/body to send. Omitted -> send the stored draft as-is.
    subject: str | None = Field(None, max_length=500)
    body: str | None = Field(None, max_length=20000)


class ContactEditRequest(BaseModel):
    # The drawer's hand-edited contact details. An empty email clears the contact.
    email: str = Field("", max_length=320)
    phone: str | None = Field(None, max_length=120)
    website: str | None = Field(None, max_length=500)


class ScheduleRequest(BaseModel):
    # When to send, ISO 8601 with its offset ("2026-09-28T10:00:00+02:00"); a
    # naive value is read as UTC. Omitted -> the scheduler's next paced slot.
    send_at: str | None = Field(None, max_length=64)


# --- prospects ---------------------------------------------------------------

@app.get("/api/prospects")
def api_prospects(tier: str | None = None, min_score: int | None = None,
                  q: str | None = None, sort: str = "fit"):
    return db.list_prospects(tier=tier, min_score=min_score, q=q, sort=sort)


@app.get("/api/prospects/{prospect_id}")
def api_prospect(prospect_id: int):
    from .service import next_draft_language
    rec = db.get_prospect(prospect_id)
    if not rec:
        raise HTTPException(404, "prospect not found")
    return {**rec, "next_language": next_draft_language(rec)}


@app.delete("/api/prospects/{prospect_id}")
def api_delete_prospect(prospect_id: int):
    if not db.delete_prospect(prospect_id):
        raise HTTPException(404, "prospect not found")
    return {"deleted": prospect_id}


@app.put("/api/prospects/{prospect_id}/notes")
def api_set_notes(prospect_id: int, req: NotesRequest):
    if not db.set_prospect_notes(prospect_id, req.notes):
        raise HTTPException(404, "prospect not found")
    return {"id": prospect_id, "notes": req.notes.strip() or None}


@app.put("/api/prospects/{prospect_id}/approval")
def api_set_approval(prospect_id: int, req: ApprovalRequest):
    """Approve the draft by hand so it's ready to send despite review findings
    (the drawer's status selector), or go back to the review's verdict.
    Returns the prospect's resulting Pipeline status."""
    if not db.set_draft_approved(prospect_id, req.approved):
        raise HTTPException(404, "prospect not found")
    return {"id": prospect_id, "approved": req.approved,
            "status": db.get_prospect(prospect_id)["draft_status"]}


@app.put("/api/prospects/{prospect_id}/language")
def api_set_language(prospect_id: int, req: LanguageRequest):
    """Pick the language the prospect's next draft is written in (the drawer's
    EN/ES switch). The current draft is left as it is until it's redrafted."""
    if not db.set_draft_language(prospect_id, req.language):
        raise HTTPException(404, "prospect not found")
    return {"id": prospect_id, "language": req.language}


@app.put("/api/prospects/{prospect_id}/queue")
def api_set_queued(prospect_id: int, req: QueueRequest):
    """Mark (or unmark) a prospect "to contact" — the auto-draft job
    (scripts/draft_queued.py) writes and reviews emails for marked prospects."""
    if not db.set_queued(prospect_id, req.queued):
        raise HTTPException(404, "prospect not found")
    return {"id": prospect_id, "queued": req.queued}


@app.post("/api/prospects/{prospect_id}/email")
def api_draft_email(prospect_id: int, req: EmailRequest | None = None):
    """Draft a cold outreach email from a prospect's research. Synchronous.

    Optional body {language: english|spanish}; omitted follows the global
    config.OUTPUT_LANGUAGE.
    """
    from .service import draft_email_for, friendly_api_error
    language = (req or EmailRequest()).language
    try:
        return draft_email_for(prospect_id, language)
    except LookupError:
        raise HTTPException(404, "prospect not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except SystemExit as e:  # make_client() with no API key
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, friendly_api_error(e))


@app.post("/api/prospects/{prospect_id}/followup")
def api_draft_followup(prospect_id: int):
    """Draft a follow-up to the emails already sent to a prospect. Synchronous.
    It's stored as the prospect's pending draft (back in To do) and goes out as a
    reply in the same Gmail thread when sent."""
    from .service import draft_followup_for, friendly_api_error
    try:
        return draft_followup_for(prospect_id)
    except LookupError:
        raise HTTPException(404, "prospect not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except SystemExit as e:  # make_client() with no API key
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, friendly_api_error(e))


@app.delete("/api/prospects/{prospect_id}/followup")
def api_discard_followup(prospect_id: int):
    """Drop a pending follow-up draft; the prospect goes back to Sent."""
    from .service import discard_followup_for
    try:
        discard_followup_for(prospect_id)
    except LookupError:
        raise HTTPException(404, "prospect not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": prospect_id, "followup": False}


@app.put("/api/prospects/{prospect_id}/email")
def api_save_email(prospect_id: int, req: EmailEditRequest):
    """Save hand edits to the drafted email (the drawer autosaves as you type).
    Returns {subject, body, remaining} — the rule checks re-run on the edit."""
    from .service import save_email_edits
    try:
        return save_email_edits(prospect_id, req.subject, req.body)
    except LookupError:
        raise HTTPException(404, "prospect not found")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/prospects/{prospect_id}/email/subject")
def api_redraft_subject(prospect_id: int, req: SendRequest | None = None):
    """Write a new subject for the drafted email, keeping the body. Synchronous.

    Optional body {subject, body}: the editor's current text, so the subject
    fits unsaved edits. Stores and returns the new subject.
    """
    from .service import redraft_subject_for, friendly_api_error
    req = req or SendRequest()
    try:
        return redraft_subject_for(prospect_id, req.body, req.subject)
    except LookupError:
        raise HTTPException(404, "prospect not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except SystemExit as e:  # make_client() with no API key
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, friendly_api_error(e))


@app.post("/api/prospects/{prospect_id}/email/body")
def api_redraft_body(prospect_id: int, req: SendRequest | None = None):
    """Write a new body for the drafted email, keeping the subject. Synchronous.

    Optional body {subject, body}: the editor's current text (the new body is
    written for that subject and differs from that body). The new body is
    reviewed like a full draft. Returns {subject, body, review}.
    """
    from .service import redraft_body_for, friendly_api_error
    req = req or SendRequest()
    try:
        return redraft_body_for(prospect_id, req.subject, req.body)
    except LookupError:
        raise HTTPException(404, "prospect not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except SystemExit as e:  # make_client() with no API key
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, friendly_api_error(e))


@app.post("/api/prospects/{prospect_id}/contact")
def api_find_contact(prospect_id: int):
    """Search the web for where to send outreach. Synchronous.

    Runs a fresh search and overwrites any stored contact — backs the on-demand
    "Find contact" button so a failed lookup can be retried. Returns
    {id, contact} where contact is null if no real address was found.
    """
    from .service import find_contact_for, friendly_api_error
    try:
        return {"id": prospect_id, "contact": find_contact_for(prospect_id)}
    except LookupError:
        raise HTTPException(404, "prospect not found")
    except SystemExit as e:  # make_client() with no API key
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, friendly_api_error(e))


@app.put("/api/prospects/{prospect_id}/contact")
def api_edit_contact(prospect_id: int, req: ContactEditRequest):
    """Hand-edit where the outreach goes (the drawer's contact form), for when
    the search got it wrong or found nothing. An empty email clears the contact.
    Returns {id, contact}, shaped like the search's answer."""
    email = req.email.strip()
    if email and not EMAIL_RE.match(email):
        raise HTTPException(400, f"“{email}” doesn't look like an email address")
    try:
        contact = db.edit_prospect_contact(prospect_id, email, req.phone, req.website)
    except LookupError:
        raise HTTPException(404, "prospect not found")
    return {"id": prospect_id, "contact": contact}


@app.post("/api/prospects/{prospect_id}/send")
def api_send_outreach(prospect_id: int, req: SendRequest | None = None):
    """Send the prospect's outreach email via Gmail and record the send.

    Optional body {subject, body} sends (and persists) the edited text; omitted
    sends the stored draft. Requires a found contact and a connected Gmail account
    (see docs/gmail-setup.md). Returns {id, sent_at, thread_id}.
    """
    from .gmailer import GmailNotConfigured
    from .service import friendly_api_error, send_outreach
    req = req or SendRequest()
    try:
        return send_outreach(prospect_id, subject=req.subject, body=req.body)
    except LookupError:
        raise HTTPException(404, "prospect not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except GmailNotConfigured as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, friendly_api_error(e))


@app.post("/api/prospects/{prospect_id}/schedule")
def api_schedule_outreach(prospect_id: int, req: ScheduleRequest | None = None):
    """Queue this prospect's draft on the scheduler: at `send_at` if given,
    else in the next paced slot.

    The draft is frozen at this point: editing it is refused until the schedule
    is cancelled. Returns {id, scheduled_at, job_id}.
    """
    from .scheduler import SchedulerUnavailable
    from .service import schedule_outreach
    try:
        return schedule_outreach(prospect_id, (req or ScheduleRequest()).send_at)
    except LookupError:
        raise HTTPException(404, "prospect not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except SchedulerUnavailable as e:
        raise HTTPException(502, str(e))


@app.delete("/api/prospects/{prospect_id}/schedule")
def api_unschedule_outreach(prospect_id: int):
    """Withdraw a queued send, making the draft editable and sendable again.

    502 if the scheduler can't be reached or the email has already gone out —
    either way we must not show it as cancelled.
    """
    from .scheduler import SchedulerUnavailable
    from .service import unschedule_outreach
    try:
        return unschedule_outreach(prospect_id)
    except LookupError:
        raise HTTPException(404, "prospect not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except SchedulerUnavailable as e:
        raise HTTPException(502, str(e))


@app.get("/api/scheduler/status")
def api_scheduler_status():
    """Whether scheduling is available, for the Outreach tab's controls."""
    from . import scheduler
    return scheduler.status()


@app.post("/api/scheduler/reconcile")
def api_reconcile_scheduled():
    """Ask the scheduler what became of every queued send and record it.

    This is what turns a scheduled email into a normal sent one. Never fails on
    an unreachable scheduler — those jobs are simply reported as still pending.
    """
    from .service import reconcile_scheduled
    return reconcile_scheduled()


@app.get("/api/gmail/status")
def api_gmail_status():
    """Whether the app can send as you, and which account, for the Outreach tab."""
    from . import gmailer
    if not gmailer.authorized():
        return {"connected": False, "email": None}
    return {"connected": True, "email": gmailer.sender_address()}


@app.get("/api/llm/budget")
def api_llm_budget(refresh: bool = False):
    """How much of the LiteLLM key's (shared, all-model) budget is left."""
    from . import budget
    return budget.llm_budget(force=refresh)


@app.get("/api/outreach")
def api_outreach():
    """Send/reply statistics for the Outreach tab."""
    return db.outreach_stats()


@app.post("/api/outreach/draft-queued")
def api_draft_queued(req: DraftRequest | None = None):
    """Start drafting emails for the "to contact" queue in the background (the
    same job as scripts/draft_queued.py, but with no limit: it runs until the
    queue is empty). Optional body {ids: [...]} drafts just those prospects.
    Returns the job status; poll
    GET /api/outreach/draft-status. 409 if a draft run is already going."""
    from .service import start_draft_queued_async
    try:
        ids = (req or DraftRequest()).ids
        return start_draft_queued_async(ids=ids) if ids else start_draft_queued_async()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except SystemExit as e:  # make_client() with no API key
        raise HTTPException(400, str(e))


@app.get("/api/outreach/draft-status")
def api_draft_status():
    """Progress of the dashboard-started draft job."""
    from .service import draft_job_status
    return draft_job_status()


@app.post("/api/outreach/queue-ready")
def api_queue_ready(req: DraftRequest | None = None):
    """Hand every draft whose status is "ready" to the scheduler, in the
    background; it picks each one's send slot. Optional body {ids: [...]} limits
    it to those prospects (others are skipped). Returns the job status; poll
    GET /api/outreach/queue-status. 409 if a queue or draft job is already
    running, 400 if the scheduler isn't set up."""
    from .scheduler import SchedulerUnavailable
    from .service import start_queue_ready_async
    try:
        ids = (req or DraftRequest()).ids
        return start_queue_ready_async(ids=ids) if ids else start_queue_ready_async()
    except SchedulerUnavailable as e:  # a RuntimeError too — catch it first
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/api/outreach/queue-status")
def api_queue_status():
    """Progress of the dashboard-started "Queue all" job."""
    from .service import queue_job_status
    return queue_job_status()


@app.get("/api/outreach/auto-queue")
def api_auto_queue_status():
    """The auto-queue loop: whether it's on, its target, and its last pass."""
    from .service import auto_queue_status
    return auto_queue_status()


@app.get("/api/outreach/blocked")
def api_outreach_blocked():
    """Unsent drafts that failed the review or the rule checks, with reasons."""
    return db.blocked_drafts()


@app.get("/api/outreach/active")
def api_outreach_active():
    """The pipeline: queued-but-unsent prospects plus everyone already emailed,
    with draft status and last send."""
    return db.active_contacts()


@app.post("/api/outreach/refresh-replies")
def api_refresh_replies():
    """Poll sent-but-unanswered threads for replies; record any found."""
    from .gmailer import GmailNotConfigured
    from .service import friendly_api_error, refresh_replies
    try:
        return refresh_replies()
    except GmailNotConfigured as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, friendly_api_error(e))


@app.get("/api/stats")
def api_stats():
    return db.stats()


@app.get("/api/results/{kind}")
def api_results(kind: str):
    """Results for one kind. Discovery groups by niche category (across runs);
    companies stays grouped by the query (run) that produced them."""
    if kind == "discover":
        return db.categorized_results()
    if kind == "companies":
        return db.grouped_results(kind)
    raise HTTPException(404, "unknown kind")


# --- categories --------------------------------------------------------------

@app.get("/api/categories")
def api_categories():
    """All niche categories — backs the 'move to category' picker."""
    return db.list_categories()


@app.put("/api/categories/{category_id}")
def api_rename_category(category_id: int, req: CategoryRequest):
    """Rename a category; renaming onto an existing name merges the two."""
    try:
        cat = db.rename_category(category_id, req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if cat is None:
        raise HTTPException(404, "category not found")
    return cat


@app.delete("/api/categories/{category_id}")
def api_delete_category(category_id: int):
    """Delete a category with every search and prospect filed under it."""
    if not db.delete_category(category_id):
        raise HTTPException(404, "category not found")
    return {"deleted": category_id}


@app.put("/api/runs/{run_id}/category")
def api_set_run_category(run_id: int, req: RunCategoryRequest):
    """Refile one search under a different category (by id, or a new name).

    Backs fixing a miscategorized search and manual merges. Body carries either
    category_id (an existing category, or null to un-file) or name (find/create).
    """
    if req.name is not None and req.name.strip():
        try:
            category_id = db.find_or_create_category(req.name)["id"]
        except ValueError as e:
            raise HTTPException(400, str(e))
    else:
        category_id = req.category_id
        if category_id is not None and db.get_category(category_id) is None:
            raise HTTPException(404, "category not found")
    if not db.set_run_category(run_id, category_id):
        raise HTTPException(404, "run not found")
    return {"id": run_id, "category_id": category_id}


# --- runs --------------------------------------------------------------------

@app.post("/api/runs")
def api_create_run(req: RunRequest):
    # Import here so browsing works even if the agent stack can't be built.
    from .service import start_run_async
    try:
        run_id = start_run_async(req.kind, req.query, req.count,
                                 thorough=req.thorough)
    except SystemExit as e:  # make_client() with no API key
        raise HTTPException(400, str(e))
    return {"run_id": run_id}


@app.post("/api/niches")
def api_niches(req: NicheRequest):
    """Suggest niches for a city — a fast, synchronous reasoning call (no run)."""
    from .service import friendly_api_error, suggest_niches_for
    try:
        niches = suggest_niches_for(req.location, req.count)
    except SystemExit as e:  # make_client() with no API key
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, friendly_api_error(e))
    return {"niches": niches}


@app.get("/api/runs")
def api_runs():
    return db.list_runs()


@app.get("/api/runs/{run_id}")
def api_run(run_id: int):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@app.delete("/api/runs/{run_id}")
def api_delete_run(run_id: int):
    """Delete a whole search (run) and all the prospects it produced."""
    if not db.delete_run(run_id):
        raise HTTPException(404, "run not found")
    return {"deleted": run_id}


# --- static / dashboard ------------------------------------------------------

@app.get("/")
def index():
    # no-cache so dashboard edits always load fresh (the browser revalidates).
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


# Serve any other static assets (none required today, but future-proof).
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
