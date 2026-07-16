"""
SQLite persistence for prospects and runs.

One file (prospector.db) next to this module. Queryable/sortable fields get real
columns; the nested structures a prospect record carries (pain_points, sources,
etc.) are stored as JSON text and re-inflated by row_to_record() into exactly the
dict shape agent.run_prospect() returns — so the CLI, the JSON API, and the
frontend all speak the same records.

Threads: the web server runs research on background threads while HTTP handlers
read concurrently. Each call opens its own short-lived connection
(check_same_thread=False) and WAL journaling keeps readers unblocked by the
writer. sqlite3's module-level access is serialized, so a connection-per-call keeps
this simple and safe for our low write volume.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Keep the store at the project root (one level up from this package), so it sits
# beside .env / results.json regardless of where the package lives.
DB_PATH = Path(__file__).resolve().parent.parent / "prospector.db"

# Prospect fields kept as JSON text columns (nested / list-shaped).
_JSON_FIELDS = ("pain_points", "buying_signals", "red_flags", "sources")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if absent. Idempotent — safe to call on every startup."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                kind        TEXT NOT NULL,              -- 'discover' | 'companies'
                query       TEXT NOT NULL,
                count       INTEGER,
                status      TEXT NOT NULL DEFAULT 'running',  -- running|done|error
                total       INTEGER DEFAULT 0,
                completed   INTEGER DEFAULT 0,
                error       TEXT,
                created_at  TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS prospects (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id           INTEGER REFERENCES runs(id) ON DELETE SET NULL,
                company          TEXT NOT NULL,
                fit_score        INTEGER,
                tier             TEXT,
                confidence       TEXT,
                one_line         TEXT,
                outreach_angle   TEXT,
                research_summary TEXT,
                pain_points      TEXT,   -- JSON
                buying_signals   TEXT,   -- JSON
                red_flags        TEXT,   -- JSON
                sources          TEXT,   -- JSON
                error            TEXT,
                created_at       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_prospects_run ON prospects(run_id);
            CREATE INDEX IF NOT EXISTS idx_prospects_score ON prospects(fit_score);
            """
        )


# --- runs --------------------------------------------------------------------

def create_run(kind: str, query: str, count: int | None) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO runs (kind, query, count, status, created_at) "
            "VALUES (?, ?, ?, 'running', ?)",
            (kind, query, count, _now()),
        )
        return cur.lastrowid


def set_run_total(run_id: int, total: int) -> None:
    with _connect() as conn:
        conn.execute("UPDATE runs SET total=? WHERE id=?", (total, run_id))


def bump_run_progress(run_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET completed=completed+1 WHERE id=?", (run_id,)
        )


def finish_run(run_id: int, status: str = "done", error: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET status=?, error=?, finished_at=? WHERE id=?",
            (status, error, _now(), run_id),
        )


def get_run(run_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None


def list_runs(kind: str | None = None, limit: int = 25) -> list[dict]:
    sql = "SELECT * FROM runs"
    params: list = []
    if kind:
        sql += " WHERE kind = ?"
        params.append(kind)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


# --- prospects ---------------------------------------------------------------

def insert_prospect(record: dict, run_id: int | None = None) -> int:
    """Persist one prospect record (the dict run_prospect returns, or an error
    record {'company', 'error'}). Returns the new row id."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO prospects (
                run_id, company, fit_score, tier, confidence, one_line,
                outreach_angle, research_summary, pain_points, buying_signals,
                red_flags, sources, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                record.get("company", ""),
                record.get("fit_score"),
                record.get("tier"),
                record.get("confidence"),
                record.get("one_line"),
                record.get("outreach_angle"),
                record.get("research_summary"),
                json.dumps(record.get("pain_points", [])),
                json.dumps(record.get("buying_signals", [])),
                json.dumps(record.get("red_flags", [])),
                json.dumps(record.get("sources", [])),
                record.get("error"),
                _now(),
            ),
        )
        return cur.lastrowid


def row_to_record(row: sqlite3.Row, full: bool = True) -> dict:
    """Inflate a prospects row back into a record dict.

    full=False returns just the summary fields the table needs (skips the heavy
    research_summary and nested JSON), keeping the list endpoint lightweight.
    """
    rec = {
        "id": row["id"],
        "run_id": row["run_id"],
        "company": row["company"],
        "fit_score": row["fit_score"],
        "tier": row["tier"],
        "confidence": row["confidence"],
        "one_line": row["one_line"],
        "outreach_angle": row["outreach_angle"],
        "error": row["error"],
        "created_at": row["created_at"],
    }
    if full:
        rec["research_summary"] = row["research_summary"]
        for field in _JSON_FIELDS:
            rec[field] = json.loads(row[field]) if row[field] else []
    return rec


def list_prospects(tier: str | None = None, min_score: int | None = None,
                   q: str | None = None, sort: str = "fit",
                   run_id: int | None = None, ungrouped: bool = False) -> list[dict]:
    """Filtered/sorted summary list for the table. sort: 'fit' | 'recent' | 'company'.

    run_id restricts to one run's prospects; ungrouped=True restricts to prospects
    with no run (imported/legacy). The two are mutually exclusive — ungrouped wins.
    """
    where, params = [], []
    if ungrouped:
        where.append("run_id IS NULL")
    elif run_id is not None:
        where.append("run_id = ?")
        params.append(run_id)
    if tier:
        where.append("tier = ?")
        params.append(tier)
    if min_score is not None:
        where.append("fit_score >= ?")
        params.append(min_score)
    if q:
        where.append("company LIKE ?")
        params.append(f"%{q}%")

    order = {
        "fit": "fit_score DESC NULLS LAST, id DESC",
        "recent": "id DESC",
        "company": "company COLLATE NOCASE ASC",
    }.get(sort, "fit_score DESC NULLS LAST, id DESC")

    sql = "SELECT * FROM prospects"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {order}"

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [row_to_record(r, full=False) for r in rows]


def get_prospect(prospect_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM prospects WHERE id=?", (prospect_id,)
        ).fetchone()
        return row_to_record(row, full=True) if row else None


def delete_prospect(prospect_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM prospects WHERE id=?", (prospect_id,))
        return cur.rowcount > 0


def grouped_results(kind: str) -> dict:
    """Results grouped by the run (query) that produced them, for one kind.

    Returns {"groups": [{"run": {...}, "prospects": [...]}, ...],  # newest run first
             "ungrouped": [...]}  where ungrouped are prospects with no run
    (imported/legacy). Prospect lists are summary rows, sorted by fit.
    """
    runs = list_runs(kind=kind, limit=200)
    groups = [{"run": r, "prospects": list_prospects(run_id=r["id"])} for r in runs]
    return {"groups": groups, "ungrouped": list_prospects(ungrouped=True)}


def stats() -> dict:
    """Header tiles: total, per-tier counts, average fit (excludes error rows)."""
    with _connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM prospects WHERE error IS NULL"
        ).fetchone()[0]
        avg = conn.execute(
            "SELECT AVG(fit_score) FROM prospects WHERE fit_score IS NOT NULL"
        ).fetchone()[0]
        tier_rows = conn.execute(
            "SELECT tier, COUNT(*) c FROM prospects WHERE tier IS NOT NULL "
            "GROUP BY tier"
        ).fetchall()
    by_tier = {r["tier"]: r["c"] for r in tier_rows}
    return {
        "total": total,
        "avg_fit": round(avg, 1) if avg is not None else None,
        "by_tier": {t: by_tier.get(t, 0) for t in ("A", "B", "C", "disqualified")},
    }
