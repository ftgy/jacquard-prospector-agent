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
                        lambda client, company, icp, hint, thorough=False: make_record(company))
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
                        lambda client, company, icp, hint, thorough=False: make_record(company))
    run_id = db.create_run("companies", "Acme, Globex", 2)
    db.set_run_total(run_id, 2)
    service.run_batch(FakeClient(), [{"company": "Acme", "hint": ""},
                                     {"company": "Globex", "hint": ""}], run_id=run_id)
    assert db.get_run(run_id)["completed"] == 2


def test_execute_run_companies_marks_done(monkeypatch):
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint, thorough=False: make_record(company))
    run_id = db.create_run("companies", "Acme, Globex", 2)
    service._execute_run(FakeClient(), run_id, "companies", "Acme, Globex", 2)

    run = db.get_run(run_id)
    assert run["status"] == "done"
    assert run["total"] == 2
    assert run["completed"] == 2
    assert {r["company"] for r in db.list_prospects()} == {"Acme", "Globex"}


def test_execute_run_companies_reresearches_and_keeps_old(monkeypatch):
    # Acme is already in the DB. The Research-companies tab does NOT filter known
    # companies (unlike discovery): naming Acme re-researches it, and the fresh
    # record is kept alongside the old one (no records are deleted).
    db.insert_prospect(make_record("Acme"))
    researched = []
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint, thorough=False: researched.append(company)
                        or make_record(company))
    run_id = db.create_run("companies", "Acme, Globex", 2)
    service._execute_run(FakeClient(), run_id, "companies", "Acme, Globex", 2)

    run = db.get_run(run_id)
    assert run["status"] == "done"
    assert researched == ["Acme", "Globex"]  # both researched — Acme not skipped
    assert run["total"] == 2 and run["completed"] == 2
    # The old Acme record is preserved, so both the old and new rows now exist.
    assert [r["company"] for r in db.list_prospects()].count("Acme") == 2


def test_execute_run_discover_skips_known_domain_under_new_name(monkeypatch):
    # A prospect already stored with acme.com; discovery re-finds the same site
    # under a different display name. It must be skipped on the domain match.
    db.insert_prospect(make_record("Acme", website="https://www.acme.com"))
    monkeypatch.setattr(service, "discover_candidates",
                        lambda client, niche, icp, count, exclude=None: [
                            {"company": "Acme Corporation", "hint": "h",
                             "website": "acme.com/about"},
                            {"company": "Globex", "hint": "h", "website": "globex.io"}])
    researched = []
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint, thorough=False: researched.append(company)
                        or make_record(company))
    run_id = db.create_run("discover", "widgets", 2)
    service._execute_run(FakeClient(), run_id, "discover", "widgets", 2)

    assert researched == ["Globex"]                 # Acme skipped by domain
    # Globex's discovered domain was persisted, so a later run skips it by domain
    # even under a different name.
    assert db.filter_unresearched([{"company": "Globex Inc", "website": "globex.io"}]) == []


def test_execute_run_retries_previously_errored_company(monkeypatch):
    # An error row is NOT a successful research, so the company stays eligible.
    db.insert_prospect({"company": "Acme", "error": "boom"})
    researched = []
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint, thorough=False: researched.append(company)
                        or make_record(company))
    run_id = db.create_run("companies", "Acme", 1)
    service._execute_run(FakeClient(), run_id, "companies", "Acme", 1)

    assert researched == ["Acme"]


def test_execute_run_discover_uses_candidates(monkeypatch):
    monkeypatch.setattr(service, "discover_candidates",
                        lambda client, niche, icp, count, exclude=None: [
                            {"company": "Found One", "hint": "h1"},
                            {"company": "Found Two", "hint": "h2"}])
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint, thorough=False: make_record(company))
    run_id = db.create_run("discover", "agencies", 2)
    service._execute_run(FakeClient(), run_id, "discover", "agencies", 2)

    assert db.get_run(run_id)["status"] == "done"
    assert {r["company"] for r in db.list_prospects()} == {"Found One", "Found Two"}


def test_execute_run_discover_empty_is_error(monkeypatch):
    monkeypatch.setattr(service, "discover_candidates",
                        lambda client, niche, icp, count, exclude=None: [])
    run_id = db.create_run("discover", "nothing here", 5)
    service._execute_run(FakeClient(), run_id, "discover", "nothing here", 5)

    run = db.get_run(run_id)
    assert run["status"] == "error"
    assert "no new companies" in run["error"].lower()


def test_discover_unresearched_retries_until_enough_fresh(monkeypatch):
    # First pass returns companies we already have; later passes must surface new
    # ones, and each pass should be told what to avoid.
    db.insert_prospect(make_record("Acme", website="https://acme.com"))
    batches = [
        [{"company": "Acme", "hint": "", "website": "acme.com"}],        # all known
        [{"company": "Globex", "hint": "", "website": "globex.io"},
         {"company": "Acme", "hint": "", "website": "acme.com"}],        # 1 new
        [{"company": "Initech", "hint": "", "website": "initech.com"}],  # 1 new
    ]
    seen_excludes = []

    def fake_discover(client, niche, icp, count, exclude=None):
        seen_excludes.append(list(exclude or []))
        return batches.pop(0) if batches else []

    monkeypatch.setattr(service, "discover_candidates", fake_discover)
    fresh = service._discover_unresearched(FakeClient(), "widgets", 2)

    assert [c["company"] for c in fresh] == ["Globex", "Initech"]
    # Acme (already researched) is in the very first avoid list.
    assert any("Acme" in ex for ex in seen_excludes)
    # Once Globex is collected, the next pass is told to avoid it too.
    assert "Globex" in seen_excludes[-1]


def test_execute_run_catches_discovery_exception(monkeypatch):
    def boom(client, niche, icp, count, exclude=None):
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


def test_draft_email_for_uses_stored_record(monkeypatch):
    captured = {}

    def fake_draft(client, record, icp, language):
        captured.update(company=record["company"], icp=icp, language=language)
        return {"subject": "s", "body": "b"}

    monkeypatch.setattr(service, "draft_outreach_email", fake_draft)
    monkeypatch.setattr(service, "find_contact", lambda client, rec: None)
    pid = db.insert_prospect(make_record("Acme"))
    out = service.draft_email_for(pid, language="spanish", client=FakeClient())

    assert {k: out[k] for k in ("subject", "body", "language", "contact")} == \
        {"subject": "s", "body": "b", "language": "spanish", "contact": None}
    assert captured["company"] == "Acme"
    assert captured["icp"] is service.ICP
    assert captured["language"] == "spanish"
    # the draft is persisted with its language, so reopening shows it
    stored = db.get_prospect(pid)["email"]
    assert stored["subject"] == "s"
    assert stored["language"] == "spanish"


def test_draft_email_for_stores_reviewed_text_and_original(monkeypatch):
    monkeypatch.setattr(service, "draft_outreach_email",
                        lambda client, rec, icp, lang: {"subject": "s", "body": "usted b"})
    monkeypatch.setattr(service, "find_contact", lambda client, rec: None)
    seen = {}

    def fake_review(client, rec, draft, icp, language, lint):
        seen["lint"] = lint
        return {"issues": [{"rule": "vosotros", "detail": "usted -> vosotros"}],
                "subject": "s", "body": "vosotros b"}

    monkeypatch.setattr(service, "review_outreach_email", fake_review)
    pid = db.insert_prospect(make_record("Acme"))
    out = service.draft_email_for(pid, language="spanish", client=FakeClient())

    assert out["body"] == "vosotros b"
    # the deterministic findings were handed to the reviewer
    assert any("usted" in i["detail"] for i in seen["lint"])
    stored = db.get_prospect(pid)["email"]
    assert stored["body"] == "vosotros b"
    review = stored["review"]
    assert review["original"] == {"subject": "s", "body": "usted b"}
    assert review["changed"] is True and review["error"] is None
    assert len(review["issues"]) == 1
    # "remaining" is re-linted on the reviewed text (usted is gone)
    assert not any("usted" in i["detail"] for i in review["remaining"])


def test_draft_email_for_keeps_draft_when_review_fails(monkeypatch):
    monkeypatch.setattr(service, "draft_outreach_email",
                        lambda client, rec, icp, lang: {"subject": "s", "body": "b"})
    monkeypatch.setattr(service, "find_contact", lambda client, rec: None)

    def boom(*a, **k):
        raise RuntimeError("proxy down")

    monkeypatch.setattr(service, "review_outreach_email", boom)
    pid = db.insert_prospect(make_record("Acme"))
    service.draft_email_for(pid, client=FakeClient())

    stored = db.get_prospect(pid)["email"]
    assert stored["body"] == "b"
    assert stored["review"]["error"] == "proxy down"
    assert db.queued_needing_draft() == []            # not queued -> not picked up
    db.set_queued(pid, True)
    assert db.queued_needing_draft() == [{"id": pid, "company": "Acme", "drafted": True}]

    # the retry reviews the stored original, without redrafting
    monkeypatch.setattr(service, "draft_outreach_email",
                        lambda *a: pytest.fail("should not redraft"))
    monkeypatch.setattr(service, "review_outreach_email",
                        lambda client, rec, draft, icp, language, lint:
                        {"issues": [], "subject": draft["subject"], "body": draft["body"]})
    out = service.review_email_for(pid, client=FakeClient())
    assert out["review"]["error"] is None
    assert db.queued_needing_draft() == []


def test_draft_email_for_review_can_be_disabled(monkeypatch):
    monkeypatch.setenv("EMAIL_REVIEW", "off")
    monkeypatch.setattr(service, "draft_outreach_email",
                        lambda client, rec, icp, lang: {"subject": "s", "body": "b"})
    monkeypatch.setattr(service, "find_contact", lambda client, rec: None)
    monkeypatch.setattr(service, "review_outreach_email",
                        lambda *a, **k: pytest.fail("review should be skipped"))
    pid = db.insert_prospect(make_record("Acme"))
    out = service.draft_email_for(pid, language="spanish", client=FakeClient())
    assert out["review"]["skipped"] is True
    assert out["review"]["remaining"]              # "b" breaks the fixed-line rules


def test_draft_email_for_finds_and_persists_contact(monkeypatch):
    monkeypatch.setattr(service, "draft_outreach_email",
                        lambda client, rec, icp, lang: {"subject": "s", "body": "b"})
    monkeypatch.setattr(service, "find_contact", lambda client, rec: {
        "email": "hola@acme.es", "phone": "900 111 222",
        "website": "acme.es", "source_url": "https://acme.es/contacto"})
    pid = db.insert_prospect(make_record("Acme"))

    out = service.draft_email_for(pid, client=FakeClient())
    assert out["contact"]["email"] == "hola@acme.es"
    # persisted, and reused (no second search) on regenerate
    stored = db.get_prospect(pid)["email"]["contact"]
    assert stored["email"] == "hola@acme.es"
    assert stored["phone"] == "900 111 222"

    def boom(client, rec):
        raise AssertionError("should reuse stored contact, not search again")

    monkeypatch.setattr(service, "find_contact", boom)
    again = service.draft_email_for(pid, client=FakeClient())
    assert again["contact"]["email"] == "hola@acme.es"


def test_draft_email_for_survives_contact_lookup_failure(monkeypatch):
    monkeypatch.setattr(service, "draft_outreach_email",
                        lambda client, rec, icp, lang: {"subject": "s", "body": "b"})

    def boom(client, rec):
        raise RuntimeError("search API down")

    monkeypatch.setattr(service, "find_contact", boom)
    pid = db.insert_prospect(make_record("Acme"))

    out = service.draft_email_for(pid, client=FakeClient())
    assert out["subject"] == "s"        # email still returned
    assert out["contact"] is None       # just no contact yet


def test_find_contact_for_persists_result(monkeypatch):
    monkeypatch.setattr(service, "find_contact", lambda client, rec: {
        "email": "hola@acme.es", "phone": "", "website": "acme.es",
        "source_url": "https://acme.es/contacto"})
    pid = db.insert_prospect(make_record("Acme"))

    contact = service.find_contact_for(pid, client=FakeClient())
    assert contact["email"] == "hola@acme.es"
    assert contact["phone"] is None            # blank normalized to NULL
    assert contact["website"] == "acme.es"
    # readable straight off the prospect (even before any email is drafted)
    db.set_prospect_email(pid, "s", "b")
    assert db.get_prospect(pid)["email"]["contact"]["email"] == "hola@acme.es"


def test_find_contact_for_none_when_nothing_found(monkeypatch):
    monkeypatch.setattr(service, "find_contact", lambda client, rec: None)
    pid = db.insert_prospect(make_record("Acme"))
    assert service.find_contact_for(pid, client=FakeClient()) is None


def test_find_contact_for_missing_prospect_raises():
    with pytest.raises(LookupError):
        service.find_contact_for(9999, client=FakeClient())


def test_draft_email_for_missing_prospect_raises():
    with pytest.raises(LookupError):
        service.draft_email_for(9999, client=FakeClient())


def test_draft_email_for_rejects_error_record():
    pid = db.insert_prospect({"company": "Broken", "error": "rate limited"})
    with pytest.raises(ValueError):
        service.draft_email_for(pid, client=FakeClient())


def test_start_run_async_rejects_unknown_kind():
    with pytest.raises(ValueError):
        service.start_run_async("bogus", "x", 1, client=FakeClient())


def test_start_run_async_creates_run_and_thread(monkeypatch):
    """A specific-companies run, driven to completion, ends up 'done' with rows."""
    monkeypatch.setattr(service, "run_prospect",
                        lambda client, company, icp, hint, thorough=False: make_record(company))
    run_id = service.start_run_async("companies", "Acme, Globex", 2, client=FakeClient())

    # The worker is a daemon thread; give it a moment, then assert the outcome.
    _wait_until(lambda: db.get_run(run_id)["status"] == "done")
    run = db.get_run(run_id)
    assert run["status"] == "done"
    assert {r["company"] for r in db.list_prospects()} == {"Acme", "Globex"}


def test_categorize_run_reuses_matching_category(monkeypatch):
    """The LLM's name is find-or-created, so the same niche returns one id."""
    monkeypatch.setattr(service, "categorize_niche",
                        lambda client, query, existing: "Real estate agencies")
    first = service.categorize_run(FakeClient(), "real estate in Barcelona")
    second = service.categorize_run(FakeClient(), "estate agents in Marbella")
    assert first is not None and first == second
    assert db.get_category(first)["name"] == "Real estate agencies"


def test_categorize_run_swallows_failure(monkeypatch):
    """A categorization error never blocks a search — it just goes uncategorized."""
    def boom(client, query, existing):
        raise RuntimeError("model down")
    monkeypatch.setattr(service, "categorize_niche", boom)
    assert service.categorize_run(FakeClient(), "widgets in Vic") is None


def test_start_run_async_discover_files_under_category(monkeypatch):
    """A discovery run lands in the category the LLM assigns it."""
    monkeypatch.setattr(service, "categorize_niche",
                        lambda client, query, existing: "Recruiting agencies")
    monkeypatch.setattr(service, "_discover_unresearched",
                        lambda client, niche, count: [])  # short-circuit the worker
    run_id = service.start_run_async("discover", "recruiters in Girona", 2,
                                     client=FakeClient())
    run = db.get_run(run_id)
    assert run["category_id"] is not None
    assert db.get_category(run["category_id"])["name"] == "Recruiting agencies"


# --- Gmail outreach: send + reply tracking -----------------------------------

def test_send_outreach_persists_edits_and_marks_sent(monkeypatch):
    from prospector import gmailer
    monkeypatch.setattr(gmailer, "send_email",
                        lambda to, subject, body: {"message_id": "m", "thread_id": "th"})
    pid = db.insert_prospect(make_record("Acme"))
    db.set_prospect_email(pid, "orig", "orig body", "spanish")
    db.set_prospect_contact(pid, "hola@acme.es")

    out = service.send_outreach(pid, subject="new subj", body="new body")
    assert out["thread_id"] == "th" and out["sent_at"]
    email = db.get_prospect(pid)["email"]
    assert email["subject"] == "new subj" and email["body"] == "new body"
    assert email["sent_at"] and email["language"] == "spanish"


def test_send_outreach_requires_contact(monkeypatch):
    pid = db.insert_prospect(make_record("Acme"))
    db.set_prospect_email(pid, "s", "b")
    with pytest.raises(ValueError):
        service.send_outreach(pid)


def test_send_outreach_missing_prospect():
    with pytest.raises(LookupError):
        service.send_outreach(9999)


def test_refresh_replies_records_found(monkeypatch):
    from prospector import gmailer
    monkeypatch.setattr(gmailer, "ensure_authorized", lambda: None)
    monkeypatch.setattr(gmailer, "my_addresses", lambda: {"me@feina.dev"})
    # first thread replied, second not
    replies = {"ta": "1757230800000", "tb": None}
    monkeypatch.setattr(gmailer, "check_reply", lambda tid, me=None: replies[tid])

    a = db.insert_prospect(make_record("A")); db.mark_sent(a, "ma", "ta")
    b = db.insert_prospect(make_record("B")); db.mark_sent(b, "mb", "tb")

    out = service.refresh_replies()
    assert out == {"checked": 2, "new_replies": 1}
    # a is now answered and drops off the worklist; b still awaits a reply.
    assert db.sent_awaiting_reply() == [{"id": b, "thread_id": "tb"}]


def test_refresh_replies_swallows_per_thread_errors(monkeypatch):
    from prospector import gmailer
    monkeypatch.setattr(gmailer, "ensure_authorized", lambda: None)
    monkeypatch.setattr(gmailer, "my_addresses", lambda: {"me@feina.dev"})

    def boom(tid, me=None):
        raise RuntimeError("thread fetch failed")
    monkeypatch.setattr(gmailer, "check_reply", boom)

    pid = db.insert_prospect(make_record("A")); db.mark_sent(pid, "m", "t")
    out = service.refresh_replies()                 # must not raise
    assert out == {"checked": 1, "new_replies": 0}


def _wait_until(pred, timeout=5.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met before timeout")


def test_draft_queued_drafts_new_and_rereviews_failed(monkeypatch):
    fresh = db.insert_prospect(make_record("Fresh"))
    failed = db.insert_prospect(make_record("Failed"))
    for pid in (fresh, failed):
        db.set_queued(pid, True)
    db.set_prospect_email(failed, "s", "b", "spanish")
    db.set_email_review(failed, {"issues": [], "remaining": [], "error": "down"})
    calls = []

    def fake_draft(pid, client=None):
        calls.append(("draft", pid))
        return {"review": {"issues": [], "remaining": [], "error": None}, "contact": None}

    def fake_review(pid, client=None):
        calls.append(("review", pid))
        return {"review": {"issues": [], "remaining": [{"rule": "x", "detail": "y"}],
                           "error": None}}

    monkeypatch.setattr(service, "draft_email_for", fake_draft)
    monkeypatch.setattr(service, "review_email_for", fake_review)
    seen = []
    counts = service.draft_queued(client=FakeClient(), pace=0,
                                  on_progress=lambda c, cur: seen.append(cur))
    assert calls == [("draft", fresh), ("review", failed)]
    assert counts == {"total": 2, "done": 2, "ok": 1, "blocked": 1, "failed": 0,
                      "aborted": False}
    assert seen == ["Fresh", "Failed", None]


def test_draft_queued_aborts_after_consecutive_failures(monkeypatch):
    for n in range(5):
        db.set_queued(db.insert_prospect(make_record(f"C{n}")), True)

    def boom(pid, client=None):
        raise RuntimeError("down")

    monkeypatch.setattr(service, "draft_email_for", boom)
    counts = service.draft_queued(client=FakeClient(), pace=0)
    assert counts["failed"] == service.DRAFT_MAX_CONSECUTIVE_ERRORS
    assert counts["aborted"] is True


def test_draft_queued_without_limit_runs_until_queue_empty(monkeypatch):
    ids = [db.insert_prospect(make_record(f"C{n}")) for n in range(12)]
    for pid in ids:
        db.set_queued(pid, True)
    late = db.insert_prospect(make_record("Late"))
    calls = []

    def fake_draft(pid, client=None):
        calls.append(pid)
        if len(calls) == 1:
            db.set_queued(late, True)          # marked while the run is going
        if pid == ids[3]:
            raise RuntimeError("one bad draft")  # stays undrafted: must not loop
        db.set_prospect_email(pid, "s", "b", "spanish")
        return {"review": {"issues": [], "remaining": [], "error": None}}

    monkeypatch.setattr(service, "draft_email_for", fake_draft)
    counts = service.draft_queued(None, client=FakeClient(), pace=0)
    assert sorted(calls) == sorted(ids + [late])   # past the old 10 cap, each once
    assert counts == {"total": 13, "done": 13, "ok": 12, "blocked": 0, "failed": 1,
                      "aborted": False}


def test_draft_queued_with_ids_redrafts_just_those(monkeypatch):
    a, b, c = (db.insert_prospect(make_record(n)) for n in "ABC")
    sent = db.insert_prospect(make_record("Sent"))
    for pid in (a, b, c, sent):
        db.set_queued(pid, True)
    db.set_prospect_email(b, "old", "body", "spanish")   # already drafted: redrafted anyway
    db.mark_sent(sent, "m", "t")
    calls = []

    def fake_draft(pid, client=None):
        calls.append(pid)
        return {"review": {"issues": [], "remaining": [], "error": None}}

    monkeypatch.setattr(service, "draft_email_for", fake_draft)
    monkeypatch.setattr(service, "review_email_for", lambda *a, **k: pytest.fail("review-only"))
    counts = service.draft_queued(None, client=FakeClient(), pace=0, ids=[b, sent, a])
    assert calls == [a, b]                  # sent one skipped; C not picked; one pass
    assert counts["total"] == counts["done"] == counts["ok"] == 2


def test_start_draft_queued_async_refuses_when_locked():
    lock = service.acquire_draft_lock()
    try:
        assert service.draft_job_status()["locked_elsewhere"] is True
        with pytest.raises(RuntimeError):
            service.start_draft_queued_async(client=FakeClient())
    finally:
        service.release_draft_lock(lock)
    assert service.draft_job_status()["locked_elsewhere"] is False


def test_start_draft_queued_async_runs_job(monkeypatch):
    import time as _time
    monkeypatch.setattr(service, "draft_queued",
                        lambda limit, client=None, on_progress=None, ids=None:
                        on_progress({"total": 0, "done": 0, "ok": 0, "blocked": 0,
                                     "failed": 0, "aborted": False}, None))
    status = service.start_draft_queued_async(client=FakeClient())
    assert status["running"] is True
    for _ in range(50):
        if not service.draft_job_status()["running"]:
            break
        _time.sleep(0.02)
    final = service.draft_job_status()
    assert final["running"] is False and final["error"] is None
    assert final["locked_elsewhere"] is False   # lock released


def test_redraft_subject_keeps_body_and_refreshes_lint(monkeypatch):
    pid = db.insert_prospect(make_record("Acme"))
    db.set_prospect_email(pid, "Automatizar el enrutado", "Hola,\n\nCuerpo.", "spanish")
    db.set_email_review(pid, {"issues": [], "error": None,
                              "remaining": [{"rule": "subject", "detail": "x"}]})
    seen = {}

    def fake(client, rec, body, icp, language, current):
        seen.update(body=body, language=language, current=current)
        return "¿Sigue alguien asignando leads a mano?"

    monkeypatch.setattr(service, "draft_email_subject", fake)
    out = service.redraft_subject_for(pid, body="Editado.", client=object())
    assert out["subject"] == "¿Sigue alguien asignando leads a mano?"
    # the editor's body steers the subject; the old subject is passed to avoid repeats
    assert seen == {"body": "Editado.", "language": "spanish",
                    "current": "Automatizar el enrutado"}
    email = db.get_prospect(pid)["email"]
    assert email["subject"] == out["subject"]
    assert email["body"] == "Hola,\n\nCuerpo."          # stored body untouched
    assert email["review"]["remaining"] == out["remaining"]
    assert not any(f["rule"] == "subject" for f in out["remaining"])  # old subject flag gone


def test_redraft_subject_needs_a_draft():
    pid = db.insert_prospect(make_record("Acme"))
    with pytest.raises(ValueError):
        service.redraft_subject_for(pid, client=object())
    with pytest.raises(LookupError):
        service.redraft_subject_for(9999, client=object())


# --- "Send all": send every ready draft ---------------------------------------

def _ready(name, contact="hola@x.es"):
    """A queued prospect with a reviewed, rule-clean draft (status 'ready')."""
    pid = db.insert_prospect(make_record(name))
    db.set_queued(pid, True)
    db.set_prospect_email(pid, f"subj {name}", f"body {name}", "spanish")
    db.set_email_review(pid, {"issues": [], "remaining": [], "error": None})
    if contact:
        db.set_prospect_contact(pid, contact)
    return pid


def _fake_gmail(monkeypatch, fail_for=()):
    from prospector import gmailer
    sent = []

    def send_email(to, subject, body):
        if subject in fail_for:
            raise RuntimeError("gmail down")
        sent.append(subject)
        return {"message_id": "m", "thread_id": "t"}

    monkeypatch.setattr(gmailer, "send_email", send_email)
    return sent


def test_send_ready_sends_only_ready_drafts(monkeypatch):
    sent = _fake_gmail(monkeypatch)
    a = _ready("A")
    _ready("NoContact", contact=None)
    blocked = _ready("Blocked")
    db.set_email_review(blocked, {"issues": [], "remaining": [{"rule": "x"}], "error": None})
    counts = service.send_ready(pace=0)
    assert sent == ["subj A"]
    assert counts["total"] == counts["sent"] == 1 and counts["failed"] == 0
    assert db.get_prospect(a)["email"]["sent_at"]


def test_send_ready_limits_to_ids(monkeypatch):
    sent = _fake_gmail(monkeypatch)
    a, b, c = _ready("A"), _ready("B"), _ready("C")
    service.send_ready([c, a], pace=0)
    assert sorted(sent) == ["subj A", "subj C"]


def test_send_ready_never_sends_twice(monkeypatch):
    sent = _fake_gmail(monkeypatch)
    a, b = _ready("A"), _ready("B")

    def progress(counts, current):
        if current == "A":
            db.mark_sent(b, "m0", "t0")      # B sent from the drawer meanwhile
    counts = service.send_ready(pace=0, on_progress=progress)
    assert sent == ["subj A"]
    assert counts["sent"] == 1 and counts["skipped"] == 1


def test_send_ready_aborts_after_consecutive_failures(monkeypatch):
    names = [f"C{n}" for n in range(5)]
    _fake_gmail(monkeypatch, fail_for={f"subj {n}" for n in names})
    for n in names:
        _ready(n)
    counts = service.send_ready(pace=0)
    assert counts["failed"] == service.SEND_MAX_CONSECUTIVE_ERRORS
    assert counts["aborted"] is True and "gmail down" in counts["last_error"]


def test_start_send_ready_async_needs_gmail(monkeypatch):
    from prospector import gmailer

    def not_connected():
        raise gmailer.GmailNotConfigured("connect Gmail")
    monkeypatch.setattr(gmailer, "ensure_authorized", not_connected)
    with pytest.raises(gmailer.GmailNotConfigured):
        service.start_send_ready_async()
    assert not service.send_job_status().get("running")


def test_start_send_ready_async_runs_job(monkeypatch):
    from prospector import gmailer
    monkeypatch.setattr(gmailer, "ensure_authorized", lambda: None)
    sent = _fake_gmail(monkeypatch)
    monkeypatch.setattr(service, "SEND_PACE_SECONDS", 0)
    a = _ready("A")
    status = service.start_send_ready_async([a])
    assert status["running"] is True
    _wait_until(lambda: not service.send_job_status()["running"])
    final = service.send_job_status()
    assert final["sent"] == 1 and final["error"] is None and sent == ["subj A"]


def test_save_email_edits_stores_text_and_refreshes_lint():
    pid = db.insert_prospect(make_record("Acme"))
    db.set_prospect_email(pid, "Automatizar el enrutado", "Hola,\n\nCuerpo.", "spanish")
    db.set_email_review(pid, {"issues": [], "error": None,
                              "remaining": [{"rule": "subject", "detail": "x"}]})
    drafted_at = db.get_prospect(pid)["email"]["generated_at"]
    out = service.save_email_edits(pid, "¿Sigue alguien asignando leads a mano?", "Editado.")
    email = db.get_prospect(pid)["email"]
    assert email["subject"] == "¿Sigue alguien asignando leads a mano?"
    assert email["body"] == "Editado."
    assert email["language"] == "spanish" and email["generated_at"] == drafted_at
    assert email["review"]["remaining"] == out["remaining"]
    assert not any(f["rule"] == "subject" for f in out["remaining"])


def test_save_email_edits_needs_a_draft():
    pid = db.insert_prospect(make_record("Acme"))
    with pytest.raises(ValueError):
        service.save_email_edits(pid, "s", "b")
    with pytest.raises(LookupError):
        service.save_email_edits(9999, "s", "b")


def test_draft_email_for_language_follows_pipeline_pick(monkeypatch):
    seen = []
    monkeypatch.setattr(service, "draft_outreach_email",
                        lambda client, record, icp, language: seen.append(language)
                        or {"subject": "s", "body": "b"})
    monkeypatch.setattr(service, "find_contact", lambda client, rec: None)
    monkeypatch.setattr(service, "email_review_enabled", lambda: False)
    monkeypatch.setenv("OUTPUT_LANGUAGE", "spanish")
    pid = db.insert_prospect(make_record("Acme"))
    service.draft_email_for(pid, client=FakeClient())            # global default
    db.set_draft_language(pid, "english")
    service.draft_email_for(pid, client=FakeClient())            # the pick wins
    assert seen == ["spanish", "english"]
    assert db.get_prospect(pid)["email"]["language"] == "english"


def test_redraft_body_keeps_subject_and_stores_reviewed_body(monkeypatch):
    pid = db.insert_prospect(make_record("Acme"))
    db.set_prospect_email(pid, "stored subj", "stored body", "spanish")
    seen = {}

    def fake_body(client, rec, subject, icp, language, current):
        seen.update(subject=subject, language=language, current=current)
        return "usted b"

    def fake_review(client, rec, draft, icp, language, lint):
        seen["reviewed"] = draft
        # the reviewer also touches the subject: that change must not stick
        return {"issues": [{"rule": "vosotros", "detail": "usted -> vosotros"}],
                "subject": "reviewer subj", "body": "vosotros b"}

    monkeypatch.setattr(service, "draft_email_body", fake_body)
    monkeypatch.setattr(service, "review_outreach_email", fake_review)
    out = service.redraft_body_for(pid, subject="editor subj", body="editor body",
                                   client=FakeClient())
    # written for the editor's subject, told to differ from the editor's body
    assert seen["subject"] == "editor subj" and seen["current"] == "editor body"
    assert seen["language"] == "spanish"
    assert seen["reviewed"] == {"subject": "editor subj", "body": "usted b"}
    assert out["subject"] == "editor subj" and out["body"] == "vosotros b"
    stored = db.get_prospect(pid)["email"]
    assert (stored["subject"], stored["body"]) == ("editor subj", "vosotros b")
    assert stored["review"]["original"] == {"subject": "editor subj", "body": "usted b"}
    assert not any("usted" in i["detail"] for i in stored["review"]["remaining"])


def test_redraft_body_needs_a_draft():
    pid = db.insert_prospect(make_record("Acme"))
    with pytest.raises(ValueError):
        service.redraft_body_for(pid, client=FakeClient())
    with pytest.raises(LookupError):
        service.redraft_body_for(9999, client=FakeClient())


def test_send_ready_includes_approved_drafts(monkeypatch):
    sent = _fake_gmail(monkeypatch)
    pid = _ready("A")
    db.set_email_review(pid, {"issues": [], "remaining": [{"rule": "x"}], "error": None})
    service.send_ready(pace=0)
    assert sent == []                                  # blocked: not sent
    db.set_draft_approved(pid, True)
    service.send_ready(pace=0)
    assert sent == ["subj A"]


def test_redraft_body_clears_approval(monkeypatch):
    pid = db.insert_prospect(make_record("Acme"))
    db.set_prospect_email(pid, "s", "b", "spanish")
    db.set_draft_approved(pid, True)
    monkeypatch.setattr(service, "draft_email_body", lambda *a: "nuevo")
    monkeypatch.setattr(service, "email_review_enabled", lambda: False)
    service.redraft_body_for(pid, client=FakeClient())
    assert db.get_prospect(pid)["approved_at"] is None
