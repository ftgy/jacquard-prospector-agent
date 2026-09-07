"""
Gmail integration — send outreach in-app and detect replies.

One Google account (yours), authorized once via scripts/gmail_auth.py, which
writes token.json next to .env. This module only *reads* that token, refreshing
it silently when it expires, and never runs the interactive consent flow itself —
so importing it (and the web server) never blocks on a browser.

Two capabilities, matching the two OAuth scopes we ask for:
  * gmail.send     — send a plain-text email as you.
  * gmail.readonly — look at a thread to see whether someone replied.

Secrets (both git-ignored, both live at the project root):
  * credentials.json — the OAuth *client* downloaded from Google Cloud.
  * token.json       — your *authorization*, minted by scripts/gmail_auth.py.

See docs/gmail-setup.md for the one-time Google Cloud setup.
"""

from __future__ import annotations

import base64
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path

# The project root (one level up from this package) — where .env already lives,
# so the Gmail secrets sit beside it rather than inside the package.
ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = ROOT / "credentials.json"
TOKEN_PATH = ROOT / "token.json"

# Least privilege: send mail, and read threads to spot replies. Nothing else.
# Changing this list invalidates an existing token.json — re-run gmail_auth.py.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]


class GmailNotConfigured(RuntimeError):
    """No usable authorization yet — the caller should point at the setup docs.

    Raised when token.json is missing, or present but no longer refreshable
    (revoked / scopes changed). The message is safe to show a user verbatim.
    """


def _load_credentials():
    """Load token.json, refreshing it if it has expired. Never prompts.

    Returns google.oauth2.credentials.Credentials. Raises GmailNotConfigured if
    there's no token yet or it can't be refreshed without human consent.
    """
    # Imported lazily so the package (and the web server) load even before the
    # Google libraries are installed or the token exists.
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    if not TOKEN_PATH.exists():
        raise GmailNotConfigured(
            "Gmail isn't connected yet. Run `python scripts/gmail_auth.py` "
            "once to authorize your account (see docs/gmail-setup.md)."
        )

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as e:
            raise GmailNotConfigured(
                "Gmail authorization expired and couldn't be refreshed "
                f"({e}). Re-run `python scripts/gmail_auth.py`."
            ) from e
        TOKEN_PATH.write_text(creds.to_json())
        return creds
    raise GmailNotConfigured(
        "Gmail authorization is invalid. Re-run `python scripts/gmail_auth.py`."
    )


def _service():
    """Build the Gmail API client from the stored credentials."""
    from googleapiclient.discovery import build

    # cache_discovery=False silences a noisy warning and avoids writing an
    # on-disk discovery cache we don't want.
    return build("gmail", "v1", credentials=_load_credentials(),
                 cache_discovery=False)


def authorized() -> bool:
    """True if we have usable credentials (no network call for a valid token)."""
    try:
        _load_credentials()
        return True
    except GmailNotConfigured:
        return False


def ensure_authorized() -> None:
    """Raise GmailNotConfigured (with an actionable message) if not connected."""
    _load_credentials()


def account_email() -> str | None:
    """The address we send as, or None if not connected / unreachable."""
    try:
        profile = _service().users().getProfile(userId="me").execute()
        return profile.get("emailAddress")
    except GmailNotConfigured:
        return None
    except Exception:
        # Connected but the profile call failed (offline, transient) — don't
        # crash a status check over it.
        return None


def send_email(to: str, subject: str, body: str) -> dict:
    """Send a plain-text email as the authorized account.

    Returns {"message_id", "thread_id"} — the ids let us follow the thread
    later to detect a reply. Raises GmailNotConfigured if not connected;
    other failures propagate as googleapiclient HttpError for the caller to map.
    """
    svc = _service()

    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    # Gmail fills In/From from the authorized account; setting From explicitly is
    # optional, so we leave it off and let Gmail use the account's send identity.

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    sent = svc.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()
    return {
        "message_id": sent["id"],
        "thread_id": sent["threadId"],
    }


def check_reply(thread_id: str, my_email: str | None = None) -> str | None:
    """Return the internal date (ISO-ish ms epoch as a string) of the first reply
    on a thread, or None if the only messages are ones we sent.

    A "reply" is any message in the thread whose From address isn't ours. We
    resolve our own address once (my_email) to compare against; if it can't be
    determined we fall back to the SENT label, treating any message *without* it
    as inbound.
    """
    from googleapiclient.errors import HttpError

    svc = _service()
    if my_email is None:
        my_email = account_email()
    me = (my_email or "").strip().lower()

    try:
        thread = svc.users().threads().get(
            userId="me", id=thread_id,
            format="metadata", metadataHeaders=["From", "Date"],
        ).execute()
    except HttpError as e:
        if e.resp.status == 404:      # thread deleted — treat as no reply
            return None
        raise

    for m in thread.get("messages", []):
        headers = {h["name"].lower(): h["value"]
                   for h in m.get("payload", {}).get("headers", [])}
        sender = parseaddr(headers.get("from", ""))[1].strip().lower()
        label_ids = m.get("labelIds", [])
        is_ours = (sender == me) if me else ("SENT" in label_ids)
        if not is_ours:
            # internalDate is ms since epoch as a string; hand it back as-is so
            # the caller can store/format it however it likes.
            return m.get("internalDate")
    return None
