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

import db
from config import describe_target, load_env, make_client

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


@app.get("/api/stats")
def api_stats():
    return db.stats()


# --- runs --------------------------------------------------------------------

@app.post("/api/runs")
def api_create_run(req: RunRequest):
    # Import here so browsing works even if the agent stack can't be built.
    from service import start_run_async
    try:
        run_id = start_run_async(req.kind, req.query, req.count)
    except SystemExit as e:  # make_client() with no API key
        raise HTTPException(400, str(e))
    return {"run_id": run_id}


@app.get("/api/runs")
def api_runs():
    return db.list_runs()


@app.get("/api/runs/{run_id}")
def api_run(run_id: int):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


# --- static / dashboard ------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


# Serve any other static assets (none required today, but future-proof).
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
