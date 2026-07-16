"""HTTP API: prospects/stats/runs endpoints and the served dashboard.

Uses FastAPI's TestClient. The store is the temp DB (conftest); the one route
that would spawn real work — POST /api/runs — has service.start_run_async
stubbed so no thread or model call happens.
"""

import pytest
from fastapi.testclient import TestClient

from prospector import db
from prospector.server import app
from tests.conftest import make_record


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_index_serves_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>Prospector" in r.text


def test_stats_endpoint(client):
    db.insert_prospect(make_record("A1", tier="A", fit=80))
    db.insert_prospect(make_record("B1", tier="B", fit=70))
    s = client.get("/api/stats").json()
    assert s["total"] == 2
    assert s["by_tier"]["A"] == 1
    assert s["avg_fit"] == 75.0


def test_list_prospects_and_filters(client):
    db.insert_prospect(make_record("Alpha", tier="A", fit=90))
    db.insert_prospect(make_record("Beta", tier="B", fit=60))

    assert len(client.get("/api/prospects").json()) == 2
    assert [r["company"] for r in client.get("/api/prospects?tier=A").json()] == ["Alpha"]
    assert [r["company"] for r in client.get("/api/prospects?min_score=70").json()] == ["Alpha"]
    assert [r["company"] for r in client.get("/api/prospects?q=bet").json()] == ["Beta"]


def test_get_prospect_detail(client, sample_record):
    pid = db.insert_prospect(sample_record)
    r = client.get(f"/api/prospects/{pid}")
    assert r.status_code == 200
    body = r.json()
    assert body["company"] == "Acme Robotics"
    assert body["pain_points"][0]["pain"] == "Manual CV screening"
    assert body["sources"][0]["url"] == "https://acme.example"


def test_get_prospect_404(client):
    assert client.get("/api/prospects/9999").status_code == 404


def test_delete_prospect(client):
    pid = db.insert_prospect(make_record("Temp"))
    assert client.delete(f"/api/prospects/{pid}").status_code == 200
    assert client.get(f"/api/prospects/{pid}").status_code == 404
    assert client.delete(f"/api/prospects/{pid}").status_code == 404


def test_create_run_validation(client):
    assert client.post("/api/runs", json={"kind": "bogus", "query": "x"}).status_code == 422
    assert client.post("/api/runs", json={"kind": "discover", "query": ""}).status_code == 422
    assert client.post("/api/runs", json={"kind": "discover", "query": "x", "count": 999}).status_code == 422


def test_create_run_starts_and_returns_id(client, monkeypatch):
    from prospector import service
    calls = {}

    def fake_start(kind, query, count):
        calls.update(kind=kind, query=query, count=count)
        return 4242

    monkeypatch.setattr(service, "start_run_async", fake_start)
    r = client.post("/api/runs", json={"kind": "discover", "query": "agencies", "count": 5})
    assert r.status_code == 200
    assert r.json() == {"run_id": 4242}
    assert calls == {"kind": "discover", "query": "agencies", "count": 5}


def test_niches_validation(client):
    assert client.post("/api/niches", json={"location": ""}).status_code == 422
    assert client.post("/api/niches", json={"location": "X", "count": 99}).status_code == 422


def test_niches_success(client, monkeypatch):
    from prospector import service
    monkeypatch.setattr(service, "suggest_niches_for",
                        lambda location, count: [
                            {"niche": f"agencies in {location}", "why": "w", "local_angle": "l"}])
    r = client.post("/api/niches", json={"location": "Girona", "count": 6})
    assert r.status_code == 200
    assert r.json()["niches"][0]["niche"] == "agencies in Girona"


def test_niches_api_error_returns_502(client, monkeypatch):
    from prospector import service

    def boom(location, count):
        raise Exception("model exploded")
    monkeypatch.setattr(service, "suggest_niches_for", boom)
    r = client.post("/api/niches", json={"location": "Girona"})
    assert r.status_code == 502
    assert "exploded" in r.json()["detail"]


def test_get_run_status(client):
    run_id = db.create_run("companies", "Acme", 1)
    db.set_run_total(run_id, 1)
    db.bump_run_progress(run_id)
    body = client.get(f"/api/runs/{run_id}").json()
    assert body["status"] == "running"
    assert body["completed"] == 1
    assert body["total"] == 1


def test_get_run_404(client):
    assert client.get("/api/runs/9999").status_code == 404


def test_list_runs(client):
    db.create_run("discover", "first", 1)
    db.create_run("discover", "second", 1)
    runs = client.get("/api/runs").json()
    assert [r["query"] for r in runs][:2] == ["second", "first"]
