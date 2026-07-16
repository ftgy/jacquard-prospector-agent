"""
Shared fixtures.

Every test runs against a throwaway SQLite file, never the real prospector.db:
the `temp_db` fixture (autouse) repoints prospector.db.DB_PATH at a tmp path and
initializes the schema. Because db, service, and server all read that one module
global at query time, patching it once covers the whole stack — no test touches
the live database or the network.
"""

import copy

import pytest


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the store at a fresh temp DB and set a dummy API key.

    The dummy key keeps config.make_client() from raising during the server
    lifespan (it only constructs the client object — no network call).
    """
    from prospector import db

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    return db


# A qualified verdict, shaped exactly like agent.run_prospect() returns.
_SAMPLE = {
    "company": "Acme Robotics",
    "fit_score": 82,
    "tier": "A",
    "confidence": "high",
    "one_line": "Strong fit with clear automatable back-office pain.",
    "pain_points": [
        {"pain": "Manual CV screening", "evidence": "Hundreds of CVs per role.",
         "agent_solution": "A triage agent ranks CVs against the spec."},
    ],
    "buying_signals": ["Hiring an ops manager", "Recently raised a seed round"],
    "red_flags": ["Small permanent team"],
    "outreach_angle": "Open on their intern-turnover problem.",
    "research_summary": "Acme is a 40-person agency ... (long text).",
    "sources": [{"title": "Acme site", "url": "https://acme.example"}],
}


@pytest.fixture
def sample_record():
    """A deep copy so a test mutating it can't leak into another."""
    return copy.deepcopy(_SAMPLE)


def make_record(company, tier="A", fit=80, **over):
    """Build a minimal-but-complete verdict record for a given company."""
    rec = copy.deepcopy(_SAMPLE)
    rec.update(company=company, tier=tier, fit_score=fit)
    rec.update(over)
    return rec
