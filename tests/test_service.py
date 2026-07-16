"""Service layer: batch persistence, run orchestration, error translation.

The agent's model calls (run_prospect / discover_candidates) are monkeypatched,
so these tests exercise the orchestration and persistence without any network.
"""

import httpx
import pytest

import anthropic

from prospector import db, service
from tests.conftest import make_record


class FakeClient:
    """Stand-in for anthropic.Anthropic — never actually called by the fakes."""


def test_friendly_api_error_credit_balance():
    msg = service.friendly_api_error(Exception("your credit balance is too low"))
    assert "out of credits" in msg


def test_friendly_api_error_generic_passthrough():
    assert service.friendly_api_error(Exception("weird boom")) == "weird boom"


def test_friendly_api_error_tls_hint():
    err = anthropic.APIConnectionError(message="down", request=httpx.Request("POST", "https://x"))
    err.__cause__ = Exception("CERTIFICATE_VERIFY_FAILED: self signed")
    msg = service.friendly_api_error(err)
    assert "SSL_CERT_FILE" in msg


def test_run_batch_persists_and_returns(monkeypatch):
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint: make_record(company))
    prospects = [{"company": "Acme", "hint": ""}, {"company": "Globex", "hint": ""}]
    results = service.run_batch(FakeClient(), prospects)

    assert [r["company"] for r in results] == ["Acme", "Globex"]
    stored = {r["company"] for r in db.list_prospects()}
    assert stored == {"Acme", "Globex"}


def test_run_batch_one_failure_does_not_kill_the_batch(monkeypatch):
    def flaky(client, company, icp, hint):
        if company == "BadCo":
            raise anthropic.RateLimitError(
                message="slow down",
                response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                body=None)
        return make_record(company)

    monkeypatch.setattr(service, "run_prospect", flaky)
    prospects = [{"company": "Acme", "hint": ""},
                 {"company": "BadCo", "hint": ""},
                 {"company": "Globex", "hint": ""}]
    results = service.run_batch(FakeClient(), prospects)

    assert len(results) == 3
    bad = [r for r in results if r["company"] == "BadCo"][0]
    assert "error" in bad
    # The failure is persisted as an error row, and the others still made it.
    assert len(db.list_prospects()) == 3


def test_run_batch_bumps_run_progress(monkeypatch):
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint: make_record(company))
    run_id = db.create_run("companies", "Acme, Globex", 2)
    db.set_run_total(run_id, 2)
    service.run_batch(FakeClient(), [{"company": "Acme", "hint": ""},
                                     {"company": "Globex", "hint": ""}], run_id=run_id)
    assert db.get_run(run_id)["completed"] == 2


def test_execute_run_companies_marks_done(monkeypatch):
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint: make_record(company))
    run_id = db.create_run("companies", "Acme, Globex", 2)
    service._execute_run(FakeClient(), run_id, "companies", "Acme, Globex", 2)

    run = db.get_run(run_id)
    assert run["status"] == "done"
    assert run["total"] == 2
    assert run["completed"] == 2
    assert {r["company"] for r in db.list_prospects()} == {"Acme", "Globex"}


def test_execute_run_discover_uses_candidates(monkeypatch):
    monkeypatch.setattr(service, "discover_candidates",
                        lambda client, niche, icp, count: [
                            {"company": "Found One", "hint": "h1"},
                            {"company": "Found Two", "hint": "h2"}])
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint: make_record(company))
    run_id = db.create_run("discover", "agencies", 2)
    service._execute_run(FakeClient(), run_id, "discover", "agencies", 2)

    assert db.get_run(run_id)["status"] == "done"
    assert {r["company"] for r in db.list_prospects()} == {"Found One", "Found Two"}


def test_execute_run_discover_empty_is_error(monkeypatch):
    monkeypatch.setattr(service, "discover_candidates",
                        lambda client, niche, icp, count: [])
    run_id = db.create_run("discover", "nothing here", 5)
    service._execute_run(FakeClient(), run_id, "discover", "nothing here", 5)

    run = db.get_run(run_id)
    assert run["status"] == "error"
    assert "no companies" in run["error"].lower()


def test_execute_run_catches_discovery_exception(monkeypatch):
    def boom(client, niche, icp, count):
        raise Exception("discovery exploded")
    monkeypatch.setattr(service, "discover_candidates", boom)
    run_id = db.create_run("discover", "agencies", 3)
    service._execute_run(FakeClient(), run_id, "discover", "agencies", 3)

    run = db.get_run(run_id)
    assert run["status"] == "error"
    assert "exploded" in run["error"]


def test_suggest_niches_for_passes_through(monkeypatch):
    captured = {}

    def fake_suggest(client, location, icp, count):
        captured.update(location=location, count=count, icp=icp)
        return [{"niche": "agencies in X", "why": "w", "local_angle": "l"}]

    monkeypatch.setattr(service, "suggest_niches", fake_suggest)
    out = service.suggest_niches_for("Bilbao", 5, client=FakeClient())

    assert out[0]["niche"] == "agencies in X"
    assert captured["location"] == "Bilbao" and captured["count"] == 5
    assert captured["icp"] is service.ICP  # qualifies against the configured ICP


def test_start_run_async_rejects_unknown_kind():
    with pytest.raises(ValueError):
        service.start_run_async("bogus", "x", 1, client=FakeClient())


def test_start_run_async_creates_run_and_thread(monkeypatch):
    """A specific-companies run, driven to completion, ends up 'done' with rows."""
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint: make_record(company))
    run_id = service.start_run_async("companies", "Acme, Globex", 2, client=FakeClient())

    # The worker is a daemon thread; give it a moment, then assert the outcome.
    _wait_until(lambda: db.get_run(run_id)["status"] == "done")
    run = db.get_run(run_id)
    assert run["status"] == "done"
    assert {r["company"] for r in db.list_prospects()} == {"Acme", "Globex"}


def _wait_until(pred, timeout=5.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met before timeout")
