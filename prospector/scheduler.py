"""
Client for the prospector-scheduler service — "send this email on Monday at 10:00".

The Gmail API has no scheduled send, so a draft that shouldn't go out now is
handed to a small always-on service (the prospector-scheduler repo) that holds it
until its moment and then sends it. That service owns nothing else: it takes a
frozen payload and a time, and hands back the Gmail ids once it's done.

This module is only the HTTP client. What the dashboard *does* with a finished
job — recording the send, moving the row to Sent — lives in service.py, so the
prospect database stays the dashboard's alone.

Configure with SCHEDULER_URL and SCHEDULER_TOKEN in .env; without them
config.scheduler_enabled() is False and the dashboard hides the controls.
"""

from __future__ import annotations

import httpx

from .config import get_scheduler_token, get_scheduler_url, scheduler_enabled

TIMEOUT = 15.0


class SchedulerUnavailable(RuntimeError):
    """The scheduler isn't configured or can't be reached.

    The message is written to be shown verbatim in the dashboard, because the
    usual cause is mundane: the tunnel is down, or the box is rebooting.
    """


def _request(method: str, path: str, **kw) -> httpx.Response:
    """One authenticated call. Raises SchedulerUnavailable for transport-level
    problems; the caller maps HTTP statuses itself."""
    if not scheduler_enabled():
        raise SchedulerUnavailable(
            "Scheduling isn't set up. Set SCHEDULER_URL and SCHEDULER_TOKEN in "
            ".env (see the prospector-scheduler repo).")
    url = f"{get_scheduler_url()}{path}"
    try:
        return httpx.request(
            method, url, timeout=TIMEOUT,
            headers={"Authorization": f"Bearer {get_scheduler_token()}"}, **kw)
    except httpx.HTTPError as e:
        raise SchedulerUnavailable(f"Scheduler unreachable at {url} — {e}") from e


def _json(r: httpx.Response) -> dict:
    """Body of a successful response, or a readable error for a failed one."""
    if r.status_code == 401:
        raise SchedulerUnavailable(
            "The scheduler rejected our token — SCHEDULER_TOKEN doesn't match "
            "the one on the server.")
    if r.status_code >= 400:
        detail = ""
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = (r.text or "")[:200]
        raise SchedulerUnavailable(f"Scheduler said {r.status_code}: {detail}")
    return r.json()


def status() -> dict:
    """{'connected', 'account', 'pending'} for the dashboard's indicator.

    Never raises: an unreachable scheduler is a state to display, not an error
    that should break the Outreach tab.
    """
    if not scheduler_enabled():
        return {"connected": False, "reason": "Not configured (SCHEDULER_URL)."}
    try:
        body = _json(_request("GET", "/status"))
    except SchedulerUnavailable as e:
        return {"connected": False, "reason": str(e)}
    return {"connected": True, "account": body.get("account"),
            "pending": body.get("pending", 0)}


def queue(to: str, subject: str, body: str, send_at: str,
          thread_id: str | None = None, skip_if_replied: bool = False,
          idempotency_key: str | None = None) -> dict:
    """Queue an email for `send_at` (UTC ISO). Returns the stored job.

    `idempotency_key` makes a retried call return the existing job instead of
    queueing a second copy — two identical cold emails is the expensive failure.
    """
    return _json(_request("POST", "/jobs", json={
        "to": to, "subject": subject, "body": body, "send_at": send_at,
        "thread_id": thread_id, "skip_if_replied": skip_if_replied,
        "idempotency_key": idempotency_key,
    }))


def get(job_id: int) -> dict | None:
    """One job, or None if the scheduler has no such id (it was pruned, or the
    service was rebuilt from an empty database)."""
    r = _request("GET", f"/jobs/{job_id}")
    if r.status_code == 404:
        return None
    return _json(r)


def cancel(job_id: int) -> dict:
    """Withdraw a queued send.

    Raises SchedulerUnavailable if it already went out (the service answers 409)
    — the caller must not report that as cancelled.
    """
    r = _request("DELETE", f"/jobs/{job_id}")
    if r.status_code == 404:
        raise SchedulerUnavailable("The scheduler has no record of that job.")
    return _json(r)
