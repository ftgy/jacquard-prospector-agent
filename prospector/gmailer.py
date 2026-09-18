"""
Gmail integration — send outreach in-app and detect replies.

One Google account (yours), authorized once via scripts/gmail_auth.py, which
writes secrets/token.json. This module only *reads* that token, refreshing it
silently when it expires, and never runs the interactive consent flow itself —
so importing it (and the web server) never blocks on a browser.

Two capabilities, matching the two OAuth scopes we ask for:
  * gmail.send     — send a plain-text email as you.
  * gmail.readonly — look at a thread to see whether someone replied.

Secrets (both git-ignored, both live in secrets/ at the project root):
  * credentials.json — the OAuth *client* downloaded from Google Cloud.
  * token.json       — your *authorization*, minted by scripts/gmail_auth.py.

See docs/gmail-setup.md for the one-time Google Cloud setup.
"""

from __future__ import annotations

import base64
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
from pathlib import Path

from .config import get_send_as

# Gmail secrets live together in a git-ignored secrets/ dir at the project root
# (one level up from this package), kept out of the code and out of git.
ROOT = Path(__file__).resolve().parent.parent
SECRETS_DIR = ROOT / "secrets"
CREDENTIALS_PATH = SECRETS_DIR / "credentials.json"  # OAuth client from Google Cloud
TOKEN_PATH = SECRETS_DIR / "token.json"              # your authorization, minted locally

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


def _send_as_identities(svc) -> list[dict]:
    """The account's "Send mail as" identities (primary address included)."""
    return svc.users().settings().sendAs().list(userId="me").execute().get("sendAs", [])


def sender_address() -> str | None:
    """The address outreach goes out from: GMAIL_SEND_AS if set, else the
    account's own address (None if not connected / unreachable)."""
    return get_send_as() or account_email()


def my_addresses() -> set[str]:
    """Every address mail from us can carry — the account plus its send-as
    aliases — so reply detection doesn't mistake our own sends for replies."""
    svc = _service()
    mine = {(s.get("sendAsEmail") or "").lower() for s in _send_as_identities(svc)}
    mine.add((account_email() or "").lower())
    mine.discard("")
    return mine


def send_email(to: str, subject: str, body: str,
               thread_id: str | None = None) -> dict:
    """Send a plain-text email as the authorized account, or from its
    GMAIL_SEND_AS alias when one is configured.

    With `thread_id` the email goes out as a reply in that thread (a follow-up):
    Gmail files it there, and In-Reply-To/References point at the thread's last
    message so the recipient's client threads it too. The subject must match
    the thread's ("Re: …") for Gmail to accept the threadId.

    Returns {"message_id", "thread_id"} — the ids let us follow the thread
    later to detect a reply. Raises GmailNotConfigured if not connected;
    other failures propagate as googleapiclient HttpError for the caller to map.
    """
    svc = _service()

    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    alias = get_send_as()
    if alias:
        # Gmail only honours a From that is a verified send-as identity; use the
        # display name configured there so it matches mail sent from the web UI.
        ident = next((s for s in _send_as_identities(svc)
                      if (s.get("sendAsEmail") or "").lower() == alias), None)
        if ident is None:
            raise GmailNotConfigured(
                f"GMAIL_SEND_AS={alias} isn't a \"Send mail as\" address on this "
                "Gmail account. Add it in Gmail settings or fix .env.")
        msg["From"] = formataddr((ident.get("displayName") or "", alias))
    # Without an alias Gmail fills From from the authorized account.
    if thread_id:
        parent = _last_message_id(svc, thread_id)
        if parent:
            msg["In-Reply-To"] = parent
            msg["References"] = parent

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    payload = {"raw": raw, **({"threadId": thread_id} if thread_id else {})}
    sent = svc.users().messages().send(userId="me", body=payload).execute()
    return {
        "message_id": sent["id"],
        "thread_id": sent["threadId"],
    }


def _last_message_id(svc, thread_id: str) -> str | None:
    """The RFC 822 Message-ID header of a thread's newest message (what a reply's
    In-Reply-To points at), or None if the thread has none we can read."""
    thread = svc.users().threads().get(
        userId="me", id=thread_id, format="metadata", metadataHeaders=["Message-ID"],
    ).execute()
    for m in reversed(thread.get("messages", [])):
        for h in m.get("payload", {}).get("headers", []):
            if h["name"].lower() == "message-id":
                return h["value"]
    return None


def check_reply(thread_id: str, mine: set[str] | None = None) -> str | None:
    """Return the internal date (ISO-ish ms epoch as a string) of the first reply
    on a thread, or None if the only messages are ones we sent.

    A "reply" is any message in the thread whose From address isn't ours. `mine`
    is our addresses (account + send-as aliases, see my_addresses), resolved once
    by the caller; if empty we fall back to the SENT label, treating any message
    *without* it as inbound.
    """
    from googleapiclient.errors import HttpError

    svc = _service()
    if mine is None:
        mine = my_addresses()

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
        is_ours = (sender in mine) if mine else ("SENT" in label_ids)
        if not is_ours:
            # internalDate is ms since epoch as a string; hand it back as-is so
            # the caller can store/format it however it likes.
            return m.get("internalDate")
    return None
