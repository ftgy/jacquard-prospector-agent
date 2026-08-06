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

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .config import describe_target, load_env, make_client

HERE = Path(__file__).parent
STATIC = HERE / "static"


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
    yield


app = FastAPI(title="Prospector", lifespan=lifespan)


# --- API models --------------------------------------------------------------

class RunRequest(BaseModel):
    kind: str = Field(..., pattern="^(discover|companies)$")
    query: str = Field(..., min_length=1)
    count: int = Field(10, ge=1, le=50)


class NicheRequest(BaseModel):
    location: str = Field(..., min_length=1)
    count: int = Field(8, ge=1, le=20)


class NotesRequest(BaseModel):
    notes: str = Field("", max_length=5000)


class EmailRequest(BaseModel):
    language: str = Field("english", pattern="^(english|spanish)$")


# --- prospects ---------------------------------------------------------------

@app.get("/api/prospects")
def api_prospects(tier: str | None = None, min_score: int | None = None,
                  q: str | None = None, sort: str = "fit"):
    return db.list_prospects(tier=tier, min_score=min_score, q=q, sort=sort)


@app.get("/api/prospects/{prospect_id}")
def api_prospect(prospect_id: int):
    rec = db.get_prospect(prospect_id)
    if not rec:
        raise HTTPException(404, "prospect not found")
    return rec


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


@app.post("/api/prospects/{prospect_id}/email")
def api_draft_email(prospect_id: int, req: EmailRequest | None = None):
    """Draft a cold outreach email from a prospect's research. Synchronous.

    Optional body {language: english|spanish}; defaults to english.
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


@app.get("/api/stats")
def api_stats():
    return db.stats()


@app.get("/api/results/{kind}")
def api_results(kind: str):
    """Prospects grouped by the query (run) that produced them, for one kind."""
    if kind not in ("discover", "companies"):
        raise HTTPException(404, "unknown kind")
    return db.grouped_results(kind)


# --- runs --------------------------------------------------------------------

@app.post("/api/runs")
def api_create_run(req: RunRequest):
    # Import here so browsing works even if the agent stack can't be built.
    from .service import start_run_async
    try:
        run_id = start_run_async(req.kind, req.query, req.count)
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
