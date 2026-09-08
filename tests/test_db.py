"""Persistence layer: schema, prospect CRUD, filtering/sorting, stats, runs."""

import pytest

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


def test_notes_roundtrip_and_clear():
    pid = db.insert_prospect(make_record("Noted"))
    assert db.get_prospect(pid)["notes"] is None  # none by default
    assert db.set_prospect_notes(pid, "  called on 8/4, keen  ") is True
    assert db.get_prospect(pid)["notes"] == "called on 8/4, keen"  # trimmed
    # blank clears it back to NULL
    assert db.set_prospect_notes(pid, "   ") is True
    assert db.get_prospect(pid)["notes"] is None


def test_set_notes_missing_prospect():
    assert db.set_prospect_notes(9999, "x") is False


def test_notes_absent_from_summary_list():
    pid = db.insert_prospect(make_record("Noted"))
    db.set_prospect_notes(pid, "some note")
    assert "notes" not in db.list_prospects()[0]  # summary rows stay light


def test_email_roundtrip():
    pid = db.insert_prospect(make_record("Mailed"))
    assert db.get_prospect(pid)["email"] is None  # none by default
    assert db.set_prospect_email(pid, "Quick idea", "Hi there…") is True
    email = db.get_prospect(pid)["email"]
    assert email["subject"] == "Quick idea"
    assert email["body"] == "Hi there…"
    assert email["generated_at"]  # timestamped
    # regenerating overwrites the stored draft
    db.set_prospect_email(pid, "New subject", "New body")
    assert db.get_prospect(pid)["email"]["subject"] == "New subject"


def test_set_email_missing_prospect():
    assert db.set_prospect_email(9999, "s", "b") is False


def test_contact_roundtrip():
    pid = db.insert_prospect(make_record("Mailed"))
    db.set_prospect_email(pid, "s", "b")
    assert db.get_prospect(pid)["email"]["contact"] is None  # none until looked up
    stored = db.set_prospect_contact(pid, "hola@acme.es", "900 111 222",
                                     "acme.es", "https://acme.es/contacto")
    assert stored["email"] == "hola@acme.es" and stored["found_at"]
    contact = db.get_prospect(pid)["email"]["contact"]
    assert contact["email"] == "hola@acme.es"
    assert contact["phone"] == "900 111 222"
    assert contact["website"] == "acme.es"
    assert contact["source"] == "https://acme.es/contacto"


def test_email_absent_from_summary_list():
    pid = db.insert_prospect(make_record("Mailed"))
    db.set_prospect_email(pid, "s", "b")
    assert "email" not in db.list_prospects()[0]  # summary rows stay light


# --- outreach: send + reply tracking -----------------------------------------

def test_mark_sent_records_ids_and_surfaces_on_record():
    pid = db.insert_prospect(make_record("Sent Co"))
    db.set_prospect_email(pid, "s", "b")
    assert db.mark_sent(pid, "msg1", "thr1") is True
    email = db.get_prospect(pid)["email"]
    assert email["sent_at"] and email["replied_at"] is None


def test_mark_sent_missing_prospect():
    assert db.mark_sent(9999, "m", "t") is False


def test_sent_awaiting_reply_lists_only_unanswered_with_thread():
    a = db.insert_prospect(make_record("A"))
    b = db.insert_prospect(make_record("B"))
    c = db.insert_prospect(make_record("C"))
    db.mark_sent(a, "ma", "ta")             # awaiting
    db.mark_sent(b, "mb", "tb")
    db.mark_replied(b, "2026-09-07T10:00:00+00:00")  # answered -> excluded
    # c never sent -> excluded
    worklist = db.sent_awaiting_reply()
    assert [w["id"] for w in worklist] == [a]
    assert worklist[0]["thread_id"] == "ta"


def test_mark_replied_clears_from_worklist_and_shows_on_record():
    pid = db.insert_prospect(make_record("Replier"))
    db.set_prospect_email(pid, "s", "b")
    db.mark_sent(pid, "m", "t")
    db.mark_replied(pid, "2026-09-07T12:34:56+00:00")
    assert db.sent_awaiting_reply() == []
    assert db.get_prospect(pid)["email"]["replied_at"] == "2026-09-07T12:34:56+00:00"


def test_resending_clears_prior_reply():
    pid = db.insert_prospect(make_record("Resend"))
    db.set_prospect_email(pid, "s", "b")
    db.mark_sent(pid, "m1", "t1")
    db.mark_replied(pid, "2026-09-07T12:00:00+00:00")
    db.mark_sent(pid, "m2", "t2")           # a fresh send resets reply tracking
    assert db.get_prospect(pid)["email"]["replied_at"] is None
    assert db.sent_awaiting_reply()[0]["thread_id"] == "t2"


def test_outreach_stats_counts_totals_and_reply_rate():
    for i in range(4):
        pid = db.insert_prospect(make_record(f"Co{i}"))
        db.mark_sent(pid, f"m{i}", f"t{i}")
    # one of the four replied
    first = db.sent_awaiting_reply()[0]["id"]
    db.mark_replied(first, "2026-09-07T09:00:00+00:00")

    s = db.outreach_stats()
    assert s["total_sent"] == 4
    assert s["total_replied"] == 1
    assert s["reply_rate"] == 0.25
    assert s["sent_today"] == 4        # marked just now, local day
    assert s["awaiting_reply"] == 3
    assert len(s["series"]) == 14      # default trailing window
    assert s["series"][-1]["sent"] == 4  # today is the last bucket
    assert len(s["recent"]) == 4


def test_outreach_stats_empty_has_no_reply_rate():
    s = db.outreach_stats()
    assert s["total_sent"] == 0
    assert s["reply_rate"] is None
    assert s["recent"] == []


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


def test_delete_run_removes_run_and_its_prospects():
    r1 = db.create_run("discover", "n1", 2)
    p1 = db.insert_prospect(make_record("A"), run_id=r1)
    p2 = db.insert_prospect(make_record("B"), run_id=r1)
    r2 = db.create_run("discover", "n2", 1)
    p3 = db.insert_prospect(make_record("C"), run_id=r2)
    loose = db.insert_prospect(make_record("Loose"))  # no run

    assert db.delete_run(r1) is True
    assert db.get_run(r1) is None
    assert db.get_prospect(p1) is None and db.get_prospect(p2) is None
    # unrelated run and ungrouped prospects are untouched
    assert db.get_run(r2) is not None
    assert db.get_prospect(p3) is not None
    assert db.get_prospect(loose) is not None
    assert db.delete_run(r1) is False  # already gone


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


def test_normalize_domain_variants():
    assert db.normalize_domain("https://www.Acme.com/contact") == "acme.com"
    assert db.normalize_domain("acme.com") == "acme.com"
    assert db.normalize_domain("www.acme.com") == "acme.com"
    assert db.normalize_domain("http://acme.com:8080/x?y=1") == "acme.com"
    assert db.normalize_domain("  ") == ""
    assert db.normalize_domain(None) == ""


def test_known_keys_reports_names_and_domains_excluding_errors():
    db.insert_prospect(make_record("Acme", website="https://acme.com"))
    db.insert_prospect({"company": "FailCo", "website": "failco.com",
                        "error": "boom"})  # error rows don't count as known
    names, domains = db.known_keys()
    assert "acme" in names and "acme.com" in domains
    assert "failco" not in names and "failco.com" not in domains


def test_filter_unresearched_matches_name_or_domain_and_dedups_batch():
    db.insert_prospect(make_record("Acme", website="https://acme.com"))
    incoming = [
        {"company": "Acme Corp", "website": "acme.com/about"},  # known domain
        {"company": "acme", "website": "other.com"},            # known name
        {"company": "Globex", "website": "globex.io"},          # new
        {"company": "Globex Ltd", "website": "globex.io"},      # dup domain in batch
        {"company": "Globex", "website": "elsewhere.com"},      # dup name in batch
    ]
    kept = db.filter_unresearched(incoming)
    assert [k["company"] for k in kept] == ["Globex"]


# --- categories --------------------------------------------------------------

def test_find_or_create_category_dedups_by_slug():
    a = db.find_or_create_category("Real Estate Agencies")
    b = db.find_or_create_category("real estate  agencies")  # case/space differ
    assert a["id"] == b["id"]                                # one row
    assert db.find_or_create_category("Dental clinics")["id"] != a["id"]


def test_find_or_create_category_rejects_blank():
    with pytest.raises(ValueError):
        db.find_or_create_category("   ")


def test_set_run_category_and_list():
    cat = db.find_or_create_category("Recruiting agencies")
    rid = db.create_run("discover", "recruiters in Girona", 2)
    assert db.get_run(rid)["category_id"] is None
    assert db.set_run_category(rid, cat["id"]) is True
    assert db.get_run(rid)["category_id"] == cat["id"]
    assert [c["name"] for c in db.list_categories()] == ["Recruiting agencies"]


def test_rename_category_plain():
    cat = db.find_or_create_category("Recruters")  # typo
    out = db.rename_category(cat["id"], "Recruiting agencies")
    assert out["name"] == "Recruiting agencies"
    assert db.get_category(cat["id"])["name"] == "Recruiting agencies"


def test_rename_category_onto_existing_name_merges():
    keep = db.find_or_create_category("Recruiting agencies")
    dupe = db.find_or_create_category("Staffing firms")
    rid = db.create_run("discover", "staffing in Reus", 1, category_id=dupe["id"])

    out = db.rename_category(dupe["id"], "recruiting agencies")  # same slug as keep
    assert out["id"] == keep["id"]                       # merged into the survivor
    assert db.get_category(dupe["id"]) is None           # the dupe is gone
    assert db.get_run(rid)["category_id"] == keep["id"]  # its run moved over


def test_delete_category_cascades_runs_and_prospects():
    cat = db.find_or_create_category("Law firms")
    rid = db.create_run("discover", "law firms in VLC", 1, category_id=cat["id"])
    pid = db.insert_prospect(make_record("Lex"), run_id=rid)
    loose = db.insert_prospect(make_record("Loose"))  # unrelated, no run

    assert db.delete_category(cat["id"]) is True
    assert db.get_category(cat["id"]) is None
    assert db.get_run(rid) is None
    assert db.get_prospect(pid) is None
    assert db.get_prospect(loose) is not None           # untouched
    assert db.delete_category(cat["id"]) is False        # already gone


def test_list_prospects_by_run_ids():
    r1 = db.create_run("discover", "n1", 1)
    r2 = db.create_run("discover", "n2", 1)
    db.insert_prospect(make_record("A", fit=50), run_id=r1)
    db.insert_prospect(make_record("B", fit=90), run_id=r2)
    db.insert_prospect(make_record("Loose"))  # no run
    got = [p["company"] for p in db.list_prospects(run_ids=[r1, r2])]
    assert got == ["B", "A"]                    # both runs, fit-sorted
    assert db.list_prospects(run_ids=[]) == []  # empty set matches nothing


def test_categorized_results_folds_runs_and_keeps_ungrouped():
    cat = db.find_or_create_category("Recruiting agencies")
    bcn = db.create_run("discover", "recruiters in Barcelona", 1, category_id=cat["id"])
    mrb = db.create_run("discover", "recruiters in Marbella", 1, category_id=cat["id"])
    db.insert_prospect(make_record("BCN Talent", fit=70), run_id=bcn)
    db.insert_prospect(make_record("Marbella Hire", fit=90), run_id=mrb)
    db.insert_prospect(make_record("Imported"))  # ungrouped (no run)

    data = db.categorized_results()
    assert len(data["categories"]) == 1
    group = data["categories"][0]
    assert group["category"]["id"] == cat["id"]
    assert len(group["runs"]) == 2
    # both searches' companies fold into one fit-sorted list
    assert [p["company"] for p in group["prospects"]] == ["Marbella Hire", "BCN Talent"]
    assert [p["company"] for p in data["ungrouped"]] == ["Imported"]


def test_categorized_results_hides_empty_categories_and_buckets_uncategorized():
    db.find_or_create_category("Empty niche")  # no runs -> hidden
    rid = db.create_run("discover", "widgets in Vic", 1)  # discover, no category
    db.insert_prospect(make_record("Widgetco"), run_id=rid)

    data = db.categorized_results()
    assert data["categories"] == []
    assert [r["id"] for r in data["uncategorized"]["runs"]] == [rid]
    assert [p["company"] for p in data["uncategorized"]["prospects"]] == ["Widgetco"]
