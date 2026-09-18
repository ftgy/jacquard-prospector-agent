"""Gmail wrapper: the send-as From header and reply detection, against a fake
Gmail API service (no network, no token)."""

import base64
from email import message_from_bytes

import pytest

from prospector import gmailer

SEND_AS = [
    {"sendAsEmail": "me@gmail.com", "displayName": "", "isPrimary": True},
    {"sendAsEmail": "hello@feina.dev", "displayName": "feina.dev"},
]


class _Call:
    def __init__(self, result): self.result = result
    def execute(self): return self.result


class FakeGmail:
    """Just the users().settings().sendAs(), messages() and threads() calls we use."""

    def __init__(self, thread=None):
        self.sent_raw = None
        self.thread = thread or {"messages": []}

    def users(self): return self
    def settings(self): return self
    def sendAs(self): return self
    def messages(self): return self
    def threads(self): return self

    def list(self, userId): return _Call({"sendAs": SEND_AS})

    def send(self, userId, body):
        self.sent_raw = body["raw"]
        return _Call({"id": "m1", "threadId": "t1"})

    def get(self, userId, id, format, metadataHeaders): return _Call(self.thread)


@pytest.fixture
def fake(monkeypatch):
    svc = FakeGmail()
    monkeypatch.setattr(gmailer, "_service", lambda: svc)
    monkeypatch.setattr(gmailer, "account_email", lambda: "me@gmail.com")
    return svc


def _sent_headers(svc):
    return message_from_bytes(base64.urlsafe_b64decode(svc.sent_raw))


def test_send_uses_alias_with_its_display_name(fake, monkeypatch):
    monkeypatch.setenv("GMAIL_SEND_AS", "hello@feina.dev")
    assert gmailer.send_email("a@b.es", "Hola", "Cuerpo") == {"message_id": "m1",
                                                              "thread_id": "t1"}
    assert _sent_headers(fake)["From"] == "\"feina.dev\" <hello@feina.dev>"


def test_send_without_alias_leaves_from_to_gmail(fake, monkeypatch):
    monkeypatch.delenv("GMAIL_SEND_AS", raising=False)
    gmailer.send_email("a@b.es", "Hola", "Cuerpo")
    assert _sent_headers(fake)["From"] is None


def test_send_rejects_unverified_alias(fake, monkeypatch):
    monkeypatch.setenv("GMAIL_SEND_AS", "nope@feina.dev")
    with pytest.raises(gmailer.GmailNotConfigured):
        gmailer.send_email("a@b.es", "Hola", "Cuerpo")
    assert fake.sent_raw is None


def _msg(sender, ms, labels=("SENT",)):
    return {"internalDate": ms, "labelIds": list(labels),
            "payload": {"headers": [{"name": "From", "value": sender}]}}


def test_check_reply_ignores_mail_sent_from_the_alias(fake):
    fake.thread = {"messages": [_msg("feina.dev <hello@feina.dev>", "1")]}
    assert gmailer.check_reply("t1") is None          # our own send, not a reply
    fake.thread["messages"].append(_msg("Ana <ana@cliente.es>", "2", labels=()))
    assert gmailer.check_reply("t1") == "2"


def test_my_addresses_covers_account_and_aliases(fake):
    assert gmailer.my_addresses() == {"me@gmail.com", "hello@feina.dev"}
