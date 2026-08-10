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
from urllib.parse import urlsplit

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
                domain           TEXT,   -- normalized website domain, for dedup
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
                notes            TEXT,   -- user-authored, free text
                email_subject    TEXT,   -- last generated outreach email
                email_body       TEXT,
                email_at         TEXT,   -- when that email was generated
                email_lang       TEXT,   -- language it was written in
                contact_email    TEXT,   -- where to send it (found on email gen)
                contact_phone    TEXT,
                contact_website  TEXT,
                contact_source   TEXT,   -- URL the email was found on
                contact_at       TEXT,   -- when the contact was looked up
                error            TEXT,
                created_at       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_prospects_run ON prospects(run_id);
            CREATE INDEX IF NOT EXISTS idx_prospects_score ON prospects(fit_score);
            """
        )
        # Migrate DBs created before newer columns existed.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(prospects)")}
        for col in ("domain", "notes", "email_subject", "email_body", "email_at",
                    "email_lang", "contact_email", "contact_phone", "contact_website",
                    "contact_source", "contact_at"):
            if col not in cols:
                conn.execute(f"ALTER TABLE prospects ADD COLUMN {col} TEXT")


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


def delete_run(run_id: int) -> bool:
    """Delete a run and every prospect it produced. Returns False if no such run.

    Prospects are removed first (rather than relying on the FK's SET NULL, which
    would orphan them as 'imported') so deleting a search really discards it.
    """
    with _connect() as conn:
        conn.execute("DELETE FROM prospects WHERE run_id=?", (run_id,))
        cur = conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
        return cur.rowcount > 0


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
                run_id, company, domain, fit_score, tier, confidence, one_line,
                outreach_angle, research_summary, pain_points, buying_signals,
                red_flags, sources, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                record.get("company", ""),
                normalize_domain(record.get("website")) or None,
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


def normalize_company(name: str | None) -> str:
    """Canonical key for matching a company by name (case/whitespace-insensitive).

    One source of truth for when two company names count as 'the same' while
    deciding whether we've already researched one.
    """
    return (name or "").strip().lower()


def normalize_domain(url: str | None) -> str:
    """Canonical key for matching a company by website.

    Strips scheme, userinfo, port, path and a leading 'www.', lowercasing the
    rest: 'https://www.Acme.com/contact' -> 'acme.com'. Empty string if there's
    nothing usable. Not a full public-suffix parse — just enough to spot the same
    site under two URLs.
    """
    s = (url or "").strip().lower()
    if not s:
        return ""
    if "//" not in s:            # bare host/path -> give urlsplit a netloc to find
        s = "//" + s
    host = urlsplit(s).netloc.split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def known_keys() -> tuple[set[str], set[str]]:
    """(company names, website domains) already SUCCESSFULLY researched, for dedup.

    Backs 'skip companies we already know' for both interactive runs (service) and
    the unattended populate script. Domains come from a prospect's discovered site
    (the `domain` column) and from any official site found during contact lookup
    (`contact_website`). Error rows are excluded on purpose, so a company whose
    research previously failed — usually a transient endpoint hiccup — can be tried
    again instead of being skipped forever.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT company, domain, contact_website FROM prospects "
            "WHERE error IS NULL"
        ).fetchall()
    names: set[str] = set()
    domains: set[str] = set()
    for r in rows:
        name = normalize_company(r["company"])
        if name:
            names.add(name)
        for d in (r["domain"], r["contact_website"]):
            dom = normalize_domain(d)
            if dom:
                domains.add(dom)
    return names, domains


def researched_company_names(limit: int = 200) -> list[str]:
    """Original-case names of successfully researched companies, newest first.

    Feeds discovery's 'don't return these' hint so a repeated search is steered
    toward different companies. Authoritative dedup still happens in
    filter_unresearched — this is only a prompt nudge, hence the cap.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT company FROM prospects "
            "WHERE error IS NULL AND company <> '' "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [r["company"] for r in rows]


def filter_unresearched(items: list[dict], *, company_key: str = "company",
                        website_key: str = "website") -> list[dict]:
    """Drop items already researched — by company name OR website domain — and
    de-duplicate within the list itself. Input order is preserved.

    Matching either key skips the item, so the same company found under a slightly
    different name (but the same site), or the same name reached via a different
    URL, is only researched once. Items with neither a name nor a domain are
    dropped as unidentifiable.
    """
    names, domains = known_keys()
    seen_names: set[str] = set()
    seen_domains: set[str] = set()
    kept: list[dict] = []
    for it in items:
        name = normalize_company(it.get(company_key))
        dom = normalize_domain(it.get(website_key))
        if not name and not dom:
            continue
        if name and (name in names or name in seen_names):
            continue
        if dom and (dom in domains or dom in seen_domains):
            continue
        if name:
            seen_names.add(name)
        if dom:
            seen_domains.add(dom)
        kept.append(it)
    return kept


def _contact_dict(row: sqlite3.Row) -> dict | None:
    """The stored 'where to send it' details, or None if none were found."""
    if not row["contact_email"]:
        return None
    return {
        "email": row["contact_email"],
        "phone": row["contact_phone"],
        "website": row["contact_website"],
        "source": row["contact_source"],
        "found_at": row["contact_at"],
    }


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
        rec["notes"] = row["notes"]
        rec["email"] = (
            {"subject": row["email_subject"], "body": row["email_body"],
             "generated_at": row["email_at"], "language": row["email_lang"],
             "contact": _contact_dict(row)}
            if row["email_subject"] else None
        )
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


def set_prospect_notes(prospect_id: int, notes: str | None) -> bool:
    """Store the user's free-text note for a prospect. Empty/blank clears it."""
    notes = (notes or "").strip() or None
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE prospects SET notes=? WHERE id=?", (notes, prospect_id)
        )
        return cur.rowcount > 0


def set_prospect_email(prospect_id: int, subject: str, body: str,
                       language: str = "english") -> bool:
    """Persist the last generated outreach email (language + when) for a prospect."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE prospects SET email_subject=?, email_body=?, email_at=?, "
            "email_lang=? WHERE id=?",
            (subject, body, _now(), language, prospect_id),
        )
        return cur.rowcount > 0


def set_prospect_contact(prospect_id: int, email: str, phone: str | None = None,
                         website: str | None = None,
                         source: str | None = None) -> dict:
    """Persist the contact details found for a prospect. Returns the stored dict."""
    ts = _now()
    with _connect() as conn:
        conn.execute(
            "UPDATE prospects SET contact_email=?, contact_phone=?, "
            "contact_website=?, contact_source=?, contact_at=? WHERE id=?",
            (email, phone, website, source, ts, prospect_id),
        )
    return {"email": email, "phone": phone, "website": website,
            "source": source, "found_at": ts}


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
