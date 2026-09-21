"""Scheduled sends: handing a draft to the scheduler service, and reconciling.

The scheduler client is monkeypatched throughout — these tests cover what the
dashboard does with a job, never the HTTP call itself (that's the scheduler
repo's own suite).
"""

from datetime import datetime, timedelta, timezone

import pytest

from prospector import db, scheduler, service
from tests.conftest import make_record


def soon(**delta) -> str:
    return (datetime.now(timezone.utc) + timedelta(**delta)).isoformat(timespec="seconds")


class FakeScheduler:
    """Stand-in for the scheduler service: remembers what was queued."""

    def __init__(self):
        self.jobs = {}
        self.next_id = 1
        self.unavailable = False

    def queue(self, **kw):
        if self.unavailable:
            raise scheduler.SchedulerUnavailable("box is down")
        job = {"id": self.next_id, "status": "pending", "recipient": kw["to"],
               "subject": kw["subject"], "body": kw["body"],
               "thread_id": kw.get("thread_id"),
               "skip_if_replied": kw.get("skip_if_replied", False),
               "send_at": kw["send_at"], "gmail_message_id": None,
               "gmail_thread_id": None, "error": None, "finished_at": None}
        self.jobs[self.next_id] = job
        self.next_id += 1
        return job

    def get(self, job_id):
        if self.unavailable:
            raise scheduler.SchedulerUnavailable("box is down")
        return self.jobs.get(job_id)

    def cancel(self, job_id):
        if self.unavailable:
            raise scheduler.SchedulerUnavailable("box is down")
        job = self.jobs.get(job_id)
        if job is None:
            raise scheduler.SchedulerUnavailable("no such job")
        if job["status"] != "pending":
            raise scheduler.SchedulerUnavailable(
                f"Job is already {job['status']} — too late to cancel.")
        job["status"] = "canceled"
        return job

    # Test helpers — what the service would later see on the job.
    def complete(self, job_id, message_id="m1", thread_id="th1", at=None):
        self.jobs[job_id].update(status="sent", gmail_message_id=message_id,
                                 gmail_thread_id=thread_id,
                                 finished_at=at or soon(minutes=-5))

    def finish(self, job_id, status, error):
        self.jobs[job_id].update(status=status, error=error,
                                 finished_at=soon(minutes=-5))


@pytest.fixture
def fake(monkeypatch):
    f = FakeScheduler()
    monkeypatch.setattr(scheduler, "queue", f.queue)
    monkeypatch.setattr(scheduler, "get", f.get)
    monkeypatch.setattr(scheduler, "cancel", f.cancel)
    return f


def drafted(company="Acme", contact="hola@acme.es") -> int:
    """A prospect with a draft and an address — ready to schedule."""
    pid = db.insert_prospect(make_record(company))
    db.set_prospect_email(pid, "Un asunto", "Un cuerpo.", "spanish")
    if contact:
        db.set_prospect_contact(pid, contact)
    return pid


# --- scheduling --------------------------------------------------------------

def test_schedule_queues_the_draft_and_records_it(fake):
    pid = drafted()
    out = service.schedule_outreach(pid, soon(days=2))

    job = fake.jobs[out["job_id"]]
    assert job["recipient"] == "hola@acme.es"
    assert job["subject"] == "Un asunto" and job["body"] == "Un cuerpo."

    email = db.get_prospect(pid)["email"]
    assert email["scheduled_at"] == out["scheduled_at"]
    assert email["scheduler_job_id"] == out["job_id"]


def test_scheduled_draft_shows_as_scheduled_in_the_pipeline(fake):
    pid = drafted()
    db.set_queued(pid, True)
    service.schedule_outreach(pid, soon(days=2))

    row = next(r for r in db.active_contacts() if r["id"] == pid)
    assert row["status"] == "scheduled"
    assert row["scheduled_at"]


def test_local_time_is_converted_to_utc(fake):
    pid = drafted()
    # 10:00 in Madrid (CEST) is 08:00 UTC — what the scheduler must store.
    out = service.schedule_outreach(pid, "2099-06-01T10:00:00+02:00")
    assert out["scheduled_at"] == "2099-06-01T08:00:00+00:00"


def test_a_naive_time_is_read_as_utc(fake):
    pid = drafted()
    out = service.schedule_outreach(pid, "2099-06-01T08:00:00")
    assert out["scheduled_at"] == "2099-06-01T08:00:00+00:00"


def test_a_time_in_the_past_is_refused(fake):
    pid = drafted()
    with pytest.raises(ValueError, match="already passed"):
        service.schedule_outreach(pid, soon(hours=-1))


def test_scheduling_needs_a_draft_and_an_address(fake):
    no_contact = drafted(contact=None)
    with pytest.raises(ValueError, match="contact address"):
        service.schedule_outreach(no_contact, soon(days=1))

    no_draft = db.insert_prospect(make_record("Globex"))
    with pytest.raises(ValueError, match="No drafted email"):
        service.schedule_outreach(no_draft, soon(days=1))

    with pytest.raises(LookupError):
        service.schedule_outreach(9999, soon(days=1))


def test_double_scheduling_is_refused(fake):
    pid = drafted()
    service.schedule_outreach(pid, soon(days=1))
    with pytest.raises(ValueError, match="Already scheduled"):
        service.schedule_outreach(pid, soon(days=2))


def test_an_unreachable_scheduler_leaves_nothing_scheduled(fake):
    pid = drafted()
    fake.unavailable = True
    with pytest.raises(scheduler.SchedulerUnavailable):
        service.schedule_outreach(pid, soon(days=1))
    assert db.get_prospect(pid)["email"]["scheduled_at"] is None


def test_the_idempotency_key_pins_prospect_and_moment(fake, monkeypatch):
    seen = {}
    monkeypatch.setattr(scheduler, "queue",
                        lambda **kw: seen.update(kw) or fake.queue(**kw))
    pid = drafted()
    out = service.schedule_outreach(pid, soon(days=1))
    assert seen["idempotency_key"] == f"prospect-{pid}-{out['scheduled_at']}"


# --- the draft is frozen once scheduled --------------------------------------

def test_a_scheduled_draft_cannot_be_edited(fake):
    pid = drafted()
    service.schedule_outreach(pid, soon(days=1))
    with pytest.raises(ValueError, match="cancel the schedule"):
        service.save_email_edits(pid, "nuevo asunto", "nuevo cuerpo")


def test_a_scheduled_draft_is_not_redrafted_or_sent_in_bulk(fake):
    pid = drafted()
    db.set_queued(pid, True)
    service.schedule_outreach(pid, soon(days=1))

    assert [w["id"] for w in db.queued_needing_draft()] == []
    assert [w["id"] for w in db.prospects_to_draft([pid])] == []
    assert [w["id"] for w in service.sendable()] == []


# --- cancelling ---------------------------------------------------------------

def test_cancel_frees_the_draft_again(fake):
    pid = drafted()
    out = service.schedule_outreach(pid, soon(days=1))

    service.unschedule_outreach(pid)

    assert fake.jobs[out["job_id"]]["status"] == "canceled"
    assert db.get_prospect(pid)["email"]["scheduled_at"] is None
    service.save_email_edits(pid, "otro asunto", "otro cuerpo")   # editable again


def test_cancelling_something_unscheduled_is_refused(fake):
    pid = drafted()
    with pytest.raises(ValueError, match="isn't scheduled"):
        service.unschedule_outreach(pid)


def test_a_job_that_already_went_out_is_not_reported_as_cancelled(fake):
    pid = drafted()
    out = service.schedule_outreach(pid, soon(days=1))
    fake.complete(out["job_id"])

    with pytest.raises(scheduler.SchedulerUnavailable, match="already sent"):
        service.unschedule_outreach(pid)
    # Still scheduled locally: the reconcile pass records the send, not a guess.
    assert db.get_prospect(pid)["email"]["scheduled_at"]


# --- reconciling --------------------------------------------------------------

def test_a_sent_job_becomes_a_normal_send(fake):
    pid = drafted()
    out = service.schedule_outreach(pid, soon(days=1))
    fake.complete(out["job_id"], message_id="msg-9", thread_id="thr-9",
                  at="2026-09-21T08:00:00+00:00")

    counts = service.reconcile_scheduled()

    assert counts["sent"] == 1
    rec = db.get_prospect(pid)
    assert rec["email"]["sent_at"] == "2026-09-21T08:00:00+00:00"
    assert rec["email"]["scheduled_at"] is None
    # The history snapshots what actually went out, with the real thread id —
    # which is what a later follow-up replies into.
    assert rec["sends"][0]["thread_id"] == "thr-9"
    assert rec["sends"][0]["subject"] == "Un asunto"
    assert rec["sends"][0]["contact_email"] == "hola@acme.es"


def test_the_send_is_dated_when_it_went_out_not_when_reconciled(fake):
    """The follow-up schedule counts from sent_at, so reconciling days later
    must not claim the email went out today."""
    pid = drafted()
    out = service.schedule_outreach(pid, soon(days=1))
    fake.complete(out["job_id"], at="2026-09-14T08:00:00+00:00")

    service.reconcile_scheduled()

    assert db.get_prospect(pid)["email"]["sent_at"] == "2026-09-14T08:00:00+00:00"


def test_a_pending_job_is_left_alone(fake):
    pid = drafted()
    service.schedule_outreach(pid, soon(days=1))

    counts = service.reconcile_scheduled()

    assert counts == {"checked": 1, "sent": 0, "skipped": 0, "failed": 0,
                      "pending": 1, "notes": []}
    assert db.get_prospect(pid)["email"]["scheduled_at"]


def test_a_skipped_job_frees_the_draft_and_explains_itself(fake):
    pid = drafted()
    out = service.schedule_outreach(pid, soon(days=1))
    fake.finish(out["job_id"], "skipped", "Not sent: they replied on the thread first.")

    counts = service.reconcile_scheduled()

    assert counts["skipped"] == 1
    assert "they replied" in counts["notes"][0]
    rec = db.get_prospect(pid)
    assert rec["email"]["scheduled_at"] is None
    assert rec["email"]["sent_at"] is None        # it never went out


def test_a_failed_job_frees_the_draft(fake):
    pid = drafted()
    out = service.schedule_outreach(pid, soon(days=1))
    fake.finish(out["job_id"], "failed", "Gmail said no")

    counts = service.reconcile_scheduled()

    assert counts["failed"] == 1 and "Gmail said no" in counts["notes"][0]
    assert db.get_prospect(pid)["email"]["scheduled_at"] is None


def test_an_unreachable_scheduler_keeps_the_schedule(fake):
    pid = drafted()
    service.schedule_outreach(pid, soon(days=1))
    fake.unavailable = True

    counts = service.reconcile_scheduled()

    assert counts["pending"] == 1
    # We don't know what happened — don't guess, and don't lose the job id.
    assert db.get_prospect(pid)["email"]["scheduler_job_id"]


def test_a_job_the_scheduler_lost_is_flagged_not_assumed(fake):
    pid = drafted()
    out = service.schedule_outreach(pid, soon(days=1))
    del fake.jobs[out["job_id"]]

    counts = service.reconcile_scheduled()

    assert counts["failed"] == 1
    assert "check Gmail" in counts["notes"][0]
    rec = db.get_prospect(pid)
    assert rec["email"]["scheduled_at"] is None
    assert rec["email"]["sent_at"] is None


def test_reconcile_with_nothing_scheduled(fake):
    assert service.reconcile_scheduled()["checked"] == 0


# --- HTTP surface -------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from prospector.server import app
    with TestClient(app) as c:
        yield c


def test_schedule_and_cancel_over_http(client, fake):
    pid = drafted()

    r = client.post(f"/api/prospects/{pid}/schedule", json={"send_at": soon(days=1)})
    assert r.status_code == 200 and r.json()["job_id"]

    # The drawer's autosave must refuse while it's frozen on the scheduler.
    edit = client.put(f"/api/prospects/{pid}/email",
                      json={"subject": "nuevo", "body": "nuevo"})
    assert edit.status_code == 400 and "cancel the schedule" in edit.json()["detail"]

    assert client.delete(f"/api/prospects/{pid}/schedule").status_code == 200
    assert client.put(f"/api/prospects/{pid}/email",
                      json={"subject": "nuevo", "body": "nuevo"}).status_code == 200


def test_scheduling_errors_map_to_useful_statuses(client, fake):
    assert client.post("/api/prospects/9999/schedule",
                       json={"send_at": soon(days=1)}).status_code == 404

    pid = drafted(contact=None)
    r = client.post(f"/api/prospects/{pid}/schedule", json={"send_at": soon(days=1)})
    assert r.status_code == 400 and "contact address" in r.json()["detail"]

    ok = drafted("Globex")
    fake.unavailable = True
    r = client.post(f"/api/prospects/{ok}/schedule", json={"send_at": soon(days=1)})
    assert r.status_code == 502


def test_reconcile_endpoint_reports_what_happened(client, fake):
    pid = drafted()
    job_id = client.post(f"/api/prospects/{pid}/schedule",
                         json={"send_at": soon(days=1)}).json()["job_id"]
    fake.complete(job_id)

    body = client.post("/api/scheduler/reconcile").json()
    assert body["sent"] == 1
    assert db.get_prospect(pid)["email"]["sent_at"]


def test_scheduler_status_says_not_configured(client, monkeypatch):
    monkeypatch.delenv("SCHEDULER_URL", raising=False)
    body = client.get("/api/scheduler/status").json()
    assert body["connected"] is False and "reason" in body
