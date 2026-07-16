"""Persistence layer: schema, prospect CRUD, filtering/sorting, stats, runs."""

from prospector import db
from tests.conftest import make_record


def test_init_db_is_idempotent():
    db.init_db()  # a second call must not raise or wipe data
    db.insert_prospect(make_record("Acme"))
    db.init_db()
    assert len(db.list_prospects()) == 1


def test_insert_and_get_inflates_json(sample_record):
    pid = db.insert_prospect(sample_record)
    rec = db.get_prospect(pid)
    # JSON columns come back as real lists, not strings.
    assert isinstance(rec["pain_points"], list)
    assert rec["pain_points"][0]["agent_solution"].startswith("A triage agent")
    assert rec["sources"][0]["url"] == "https://acme.example"
    assert rec["buying_signals"] == sample_record["buying_signals"]
    assert rec["research_summary"] == sample_record["research_summary"]


def test_get_missing_prospect_returns_none():
    assert db.get_prospect(9999) is None


def test_list_is_summary_only():
    """The table endpoint stays light: no heavy research_summary / nested JSON."""
    db.insert_prospect(make_record("Acme"))
    row = db.list_prospects()[0]
    assert "company" in row and "fit_score" in row
    assert "research_summary" not in row
    assert "pain_points" not in row


def test_filter_by_tier():
    db.insert_prospect(make_record("A Co", tier="A", fit=90))
    db.insert_prospect(make_record("B Co", tier="B", fit=70))
    db.insert_prospect(make_record("DQ Co", tier="disqualified", fit=10))
    assert [r["company"] for r in db.list_prospects(tier="B")] == ["B Co"]
    assert len(db.list_prospects(tier="disqualified")) == 1


def test_filter_by_min_score():
    db.insert_prospect(make_record("Low", fit=40))
    db.insert_prospect(make_record("High", fit=85))
    got = [r["company"] for r in db.list_prospects(min_score=50)]
    assert got == ["High"]


def test_search_by_company_substring():
    db.insert_prospect(make_record("Barcelona Realty"))
    db.insert_prospect(make_record("Madrid Motors"))
    got = [r["company"] for r in db.list_prospects(q="barce")]  # case-insensitive
    assert got == ["Barcelona Realty"]


def test_sort_by_fit_desc_is_default():
    db.insert_prospect(make_record("Mid", fit=60))
    db.insert_prospect(make_record("Top", fit=95))
    db.insert_prospect(make_record("Low", fit=30))
    assert [r["company"] for r in db.list_prospects()] == ["Top", "Mid", "Low"]


def test_sort_by_company():
    db.insert_prospect(make_record("Zeta"))
    db.insert_prospect(make_record("alpha"))
    got = [r["company"] for r in db.list_prospects(sort="company")]
    assert got == ["alpha", "Zeta"]  # NOCASE collation


def test_error_record_persists_and_is_excluded_from_stats():
    db.insert_prospect(make_record("Good", tier="A", fit=80))
    db.insert_prospect({"company": "Broken", "error": "rate limited"})
    s = db.stats()
    assert s["total"] == 1  # error rows don't count toward total
    # but the error row is still retrievable for display
    broken = [r for r in db.list_prospects() if r["company"] == "Broken"][0]
    assert broken["error"] == "rate limited"
    assert db.get_prospect(broken["id"])["error"] == "rate limited"


def test_stats_counts_and_average():
    db.insert_prospect(make_record("A1", tier="A", fit=80))
    db.insert_prospect(make_record("A2", tier="A", fit=90))
    db.insert_prospect(make_record("B1", tier="B", fit=70))
    s = db.stats()
    assert s["total"] == 3
    assert s["by_tier"] == {"A": 2, "B": 1, "C": 0, "disqualified": 0}
    assert s["avg_fit"] == 80.0  # (80+90+70)/3


def test_stats_empty_db():
    s = db.stats()
    assert s["total"] == 0
    assert s["avg_fit"] is None
    assert s["by_tier"]["A"] == 0


def test_delete_prospect():
    pid = db.insert_prospect(make_record("Temp"))
    assert db.delete_prospect(pid) is True
    assert db.get_prospect(pid) is None
    assert db.delete_prospect(pid) is False  # already gone


# --- runs --------------------------------------------------------------------

def test_run_lifecycle():
    run_id = db.create_run("discover", "recruiting agencies", 5)
    run = db.get_run(run_id)
    assert run["status"] == "running"
    assert run["completed"] == 0

    db.set_run_total(run_id, 3)
    db.bump_run_progress(run_id)
    db.bump_run_progress(run_id)
    run = db.get_run(run_id)
    assert run["total"] == 3
    assert run["completed"] == 2

    db.finish_run(run_id, "done")
    run = db.get_run(run_id)
    assert run["status"] == "done"
    assert run["finished_at"] is not None


def test_finish_run_with_error():
    run_id = db.create_run("companies", "Acme", 1)
    db.finish_run(run_id, "error", "discovery failed")
    run = db.get_run(run_id)
    assert run["status"] == "error"
    assert run["error"] == "discovery failed"


def test_get_missing_run_returns_none():
    assert db.get_run(9999) is None


def test_list_runs_newest_first():
    r1 = db.create_run("discover", "first", 1)
    r2 = db.create_run("discover", "second", 1)
    runs = db.list_runs()
    assert [r["id"] for r in runs][:2] == [r2, r1]


def test_prospect_linked_to_run():
    run_id = db.create_run("companies", "Acme", 1)
    pid = db.insert_prospect(make_record("Acme"), run_id=run_id)
    assert db.get_prospect(pid)["run_id"] == run_id


def test_list_runs_filter_by_kind():
    db.create_run("discover", "agencies", 3)
    db.create_run("companies", "Acme", 1)
    db.create_run("discover", "law firms", 5)
    kinds = [r["kind"] for r in db.list_runs(kind="discover")]
    assert kinds == ["discover", "discover"]
    assert len(db.list_runs(kind="companies")) == 1


def test_list_prospects_by_run_id():
    r1 = db.create_run("discover", "n1", 1)
    r2 = db.create_run("discover", "n2", 1)
    db.insert_prospect(make_record("InRun1"), run_id=r1)
    db.insert_prospect(make_record("InRun2"), run_id=r2)
    db.insert_prospect(make_record("Loose"))  # no run
    assert [p["company"] for p in db.list_prospects(run_id=r1)] == ["InRun1"]
    assert [p["company"] for p in db.list_prospects(ungrouped=True)] == ["Loose"]


# --- grouped results ---------------------------------------------------------

def test_grouped_results_by_query():
    r1 = db.create_run("discover", "recruiting in BCN", 2)
    db.insert_prospect(make_record("Kulturo", fit=70), run_id=r1)
    db.insert_prospect(make_record("Talent Co", fit=90), run_id=r1)
    r2 = db.create_run("discover", "law firms in VLC", 1)
    db.insert_prospect(make_record("Lex", fit=60), run_id=r2)
    db.insert_prospect(make_record("Imported One"))  # ungrouped

    data = db.grouped_results("discover")
    # newest run first, each carries its query and its prospects (sorted by fit)
    assert data["groups"][0]["run"]["query"] == "law firms in VLC"
    assert data["groups"][1]["run"]["query"] == "recruiting in BCN"
    bcn = data["groups"][1]["prospects"]
    assert [p["company"] for p in bcn] == ["Talent Co", "Kulturo"]  # 90 before 70
    assert [p["company"] for p in data["ungrouped"]] == ["Imported One"]


def test_grouped_results_excludes_other_kinds():
    rd = db.create_run("discover", "a niche", 1)
    db.insert_prospect(make_record("NicheCo"), run_id=rd)
    rc = db.create_run("companies", "Acme", 1)
    db.insert_prospect(make_record("Acme"), run_id=rc)

    disc = db.grouped_results("discover")
    assert [g["run"]["query"] for g in disc["groups"]] == ["a niche"]
    comp = db.grouped_results("companies")
    assert [g["run"]["query"] for g in comp["groups"]] == ["Acme"]
