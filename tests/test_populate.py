"""The unattended populate driver: budget gating and never leaving a run stuck."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "populate", Path(__file__).resolve().parent.parent / "scripts" / "populate.py")
populate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(populate)


@pytest.fixture(autouse=True)
def no_pacing(monkeypatch):
    monkeypatch.setattr(populate, "PACE_SECONDS", 0)


@pytest.mark.parametrize("budget, state", [
    ({"available": True, "remaining": 5.01}, "ok"),
    ({"available": True, "remaining": 5.0}, "low"),
    ({"available": True, "remaining": 1.2}, "low"),
    ({"available": True, "remaining": None}, "offline"),  # uncapped key
    ({"available": False, "reason": "Proxy unreachable"}, "offline"),
])
def test_budget_state(monkeypatch, budget, state):
    monkeypatch.setattr(populate, "llm_budget", lambda force=False: budget)
    assert populate.budget_state(5.0) == state


def test_batch_stops_when_asked(monkeypatch, temp_db):
    researched = []

    def fake_prospect(client, company, icp, hint=""):
        researched.append(company)
        return {"company": company, "tier": "B", "fit_score": 60}

    monkeypatch.setattr(populate, "run_prospect", fake_prospect)
    run_id = temp_db.create_run("discover", "niche", 3)
    cands = [{"company": c} for c in ("One", "Two", "Three")]

    counts = populate.research_batch(None, run_id, cands,
                                     should_stop=lambda: len(researched) >= 2)

    assert researched == ["One", "Two"]
    assert counts["ok"] == 2


def test_interrupted_batch_closes_its_run(monkeypatch, temp_db):
    monkeypatch.setattr(populate, "make_client", lambda: None)
    monkeypatch.setattr(populate, "categorize_run", lambda client, niche: None)
    monkeypatch.setattr(populate, "discover_candidates",
                        lambda client, niche, icp, count: [{"company": "Acme"}])

    def killed(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(populate, "run_prospect", killed)

    with pytest.raises(KeyboardInterrupt):
        populate.run_once("Spain", 5, "some niche", False)

    run = temp_db.list_runs(kind="discover")[0]
    assert run["status"] == "error"
    assert run["finished_at"]
