"""
Client + model configuration.

Works against either:
  * the Anthropic API directly (default), or
  * an Anthropic-compatible proxy such as LiteLLM (set ANTHROPIC_BASE_URL).

Both use the official Anthropic SDK — a proxy is just a base_url change, because
LiteLLM serves the same /v1/messages endpoint the SDK already speaks.
"""

import os
from pathlib import Path

import anthropic

# Project root (one level up from this package) — where .env and the CSVs live.
ROOT = Path(__file__).resolve().parent.parent

# Override per-environment via .env. LiteLLM instances name models however their
# config declares them, so the model is configurable rather than hard-coded.
DEFAULT_MODEL = "vertex_ai/claude-opus-4-8"

# Language the agent writes its research output in (discovery notes, research
# summaries, and qualification verdicts). Flip this one variable to make the whole
# pipeline produce Spanish. Override per-environment with OUTPUT_LANGUAGE in .env.
# Known values are the keys of OUTPUT_LANGUAGES below.
OUTPUT_LANGUAGE = "spanish"

OUTPUT_LANGUAGES = {"english": "English", "spanish": "Spanish"}


def load_env():
    """Minimal .env loader so you don't need python-dotenv."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def use_system_ca_bundle():
    """Trust the system CA store if this network runs a TLS-intercepting proxy.

    The SDK verifies against certifi's bundle, which omits a proxy's private root
    CA — so requests fail with CERTIFICATE_VERIFY_FAILED even though curl works.
    Pointing at the system bundle keeps verification ON, just against the trust
    store that already has the proxy's CA. Respects an existing SSL_CERT_FILE.
    (Irrelevant for a plain-http:// LiteLLM endpoint, but harmless.)
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    for bundle in ("/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu
                   "/etc/pki/tls/certs/ca-bundle.crt"):    # RHEL/Fedora
        if Path(bundle).exists():
            os.environ["SSL_CERT_FILE"] = bundle
            return


def get_model() -> str:
    return os.environ.get("PROSPECT_MODEL", DEFAULT_MODEL)


def get_email_model() -> str:
    """Model for the outreach-email draft stage.

    Drafting a cold email is a Spanish creative-writing task, distinct from the
    research pipeline, so it can run on its own model — a cheaper Claude, or a
    different "voice" to A/B-test against outreach reply rates. Falls back to the
    pipeline model. Override with PROSPECT_EMAIL_MODEL in .env.

    Only consulted for the Anthropic email provider; DeepSeek uses
    get_deepseek_model() instead (see get_email_provider).
    """
    return os.environ.get("PROSPECT_EMAIL_MODEL") or get_model()


def get_email_provider() -> str:
    """Which provider drafts the outreach email: 'anthropic' (default) or 'deepseek'.

    The email stage is the one stage that can run off-Anthropic — a cheap,
    different "voice" to A/B against reply rates. DeepSeek speaks the OpenAI chat
    shape, not /v1/messages, so it takes a separate client and call path (see
    agent._structure_openai). Set EMAIL_PROVIDER=deepseek in .env to switch.
    """
    p = os.environ.get("EMAIL_PROVIDER", "anthropic").strip().lower()
    return p if p in ("anthropic", "deepseek") else "anthropic"


def get_send_as() -> str | None:
    """The Gmail send-as alias outreach goes out from (e.g. hello@feina.dev), or
    None to send as the authorized account itself. It must already be a verified
    "Send mail as" address in that Gmail account. Set GMAIL_SEND_AS in .env.
    """
    return os.environ.get("GMAIL_SEND_AS", "").strip().lower() or None


def get_review_model() -> str:
    """Claude model that reviews each drafted email against the playbook.

    The review always runs on Anthropic, whichever provider drafted the email —
    it's the check that catches the drafter drifting from the rules. Falls back to
    the email model, then the pipeline model. Override with PROSPECT_REVIEW_MODEL.
    """
    return os.environ.get("PROSPECT_REVIEW_MODEL") or get_email_model()


def email_review_enabled() -> bool:
    """Whether drafts get the Claude review pass. On unless EMAIL_REVIEW=off.

    The free deterministic checks (email_lint) run either way.
    """
    return os.environ.get("EMAIL_REVIEW", "on").strip().lower() not in ("off", "0", "false")


def get_scheduler_url() -> str | None:
    """Base URL of the prospector-scheduler service, or None if not configured.

    The Gmail API can't schedule a send, so "send this on Monday at 10:00" is
    handed to that always-on service instead (see the prospector-scheduler repo).
    Without SCHEDULER_URL the dashboard just hides the scheduling controls and
    everything else works as before.
    """
    return os.environ.get("SCHEDULER_URL", "").strip().rstrip("/") or None


def get_scheduler_token() -> str:
    """Shared secret for the scheduler API — must match its SCHEDULER_TOKEN."""
    return os.environ.get("SCHEDULER_TOKEN", "").strip()


def scheduler_enabled() -> bool:
    """Whether scheduling is available: both the URL and the token are set.

    A URL without a token would 401 on every call, so treat that as off rather
    than showing controls that can't work.
    """
    return bool(get_scheduler_url() and get_scheduler_token())


def get_auto_queue_target() -> int:
    """How many pending emails the auto-queue loop keeps on the scheduler.
    AUTO_QUEUE_TARGET=10 tops the queue up to 10; unset, 0 or "off" (the
    default) leaves the loop off and queueing stays manual."""
    raw = os.environ.get("AUTO_QUEUE_TARGET", "").strip().lower()
    return int(raw) if raw.isdigit() else 0


def get_auto_draft_buffer() -> int:
    """Ready first-email drafts the auto-queue loop keeps in reserve beyond what
    the scheduler holds: AUTO_DRAFT_BUFFER=20. Unset/0: draft only to fill the
    scheduler."""
    raw = os.environ.get("AUTO_DRAFT_BUFFER", "").strip()
    return int(raw) if raw.isdigit() else 0


def get_auto_queue_interval() -> float:
    """Minutes between auto-queue passes. AUTO_QUEUE_INTERVAL, default 15."""
    try:
        return max(1.0, float(os.environ.get("AUTO_QUEUE_INTERVAL", "15")))
    except ValueError:
        return 15.0


def get_auto_mark_tiers() -> set[str]:
    """Tiers a freshly researched company is marked "to contact" in, with no
    review by hand: AUTO_MARK_TIERS=A,B. Unset (the default) marks nothing."""
    raw = os.environ.get("AUTO_MARK_TIERS", "")
    return {t.strip().upper() for t in raw.split(",") if t.strip()}


def get_followup_days() -> list[int]:
    """The follow-up schedule: how many days after each email the next follow-up
    is due, one entry per follow-up. FOLLOWUP_DAYS=4,7 (the default) means the
    first follow-up is due 4 days after the first email and the second 7 days
    after that; then no more. FOLLOWUP_DAYS=off (or empty) turns it off.
    """
    raw = os.environ.get("FOLLOWUP_DAYS", "4,7").strip().lower()
    if raw in ("", "off", "0", "false"):
        return []
    return [int(d) for d in raw.split(",") if d.strip().isdigit() and int(d) > 0]


def get_deepseek_model() -> str:
    """DeepSeek model on the personal-key backup route. Override with DEEPSEEK_MODEL."""
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def get_deepseek_proxy_model() -> str | None:
    """DeepSeek model to draft with through the LiteLLM proxy, or None to skip it.

    With EMAIL_PROVIDER=deepseek the email stage tries this first and only falls
    back to the personal DEEPSEEK_API_KEY if the proxy call fails. Override with
    DEEPSEEK_PROXY_MODEL in .env; set it empty to always use the personal key.
    Needs ANTHROPIC_BASE_URL (no proxy → personal key only).
    """
    if not get_base_url():
        return None
    return os.environ.get("DEEPSEEK_PROXY_MODEL", "deepseek/deepseek-v4-pro").strip() or None


def get_deepseek_base_url() -> str:
    """DeepSeek API base. Their key talks to this directly, not via the proxy."""
    return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")


def get_output_language() -> str:
    """The key ('english'/'spanish') of the language research output is written in.

    Reads OUTPUT_LANGUAGE from the environment, else the module default. Unknown
    values fall back to English so a typo never breaks a run.
    """
    lang = os.environ.get("OUTPUT_LANGUAGE", OUTPUT_LANGUAGE).strip().lower()
    return lang if lang in OUTPUT_LANGUAGES else "english"


def output_language_name() -> str:
    """Human-readable name of the output language, e.g. 'Spanish'."""
    return OUTPUT_LANGUAGES[get_output_language()]


def get_base_url() -> str | None:
    return os.environ.get("ANTHROPIC_BASE_URL") or None


def using_proxy() -> bool:
    return bool(get_base_url())


def get_web_search_tool() -> str:
    """Which web_search variant to send.

    Anthropic direct / Bedrock serve web_search_20260209; Vertex only serves the
    older web_search_20250305. Override with WEB_SEARCH_TOOL in .env.
    """
    return os.environ.get("WEB_SEARCH_TOOL", "web_search_20260209")


def use_native_structured_output() -> bool:
    """Whether to trust output_config.format for schema-enforced JSON.

    Not universal: proxies (and some Vertex routes) accept the parameter and
    silently ignore it, returning prose — which is worse than rejecting it.
    So default to prompted JSON behind a proxy, native on Anthropic direct.
    Force either way with STRUCTURED_OUTPUT=native|prompted.
    """
    mode = os.environ.get("STRUCTURED_OUTPUT", "auto").lower()
    if mode == "native":
        return True
    if mode == "prompted":
        return False
    return not using_proxy()


def make_client() -> anthropic.Anthropic:
    """Build the SDK client, pointed at a proxy if ANTHROPIC_BASE_URL is set."""
    load_env()
    use_system_ca_bundle()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Set ANTHROPIC_API_KEY (see .env.example).\n"
            "Using LiteLLM? That's your LiteLLM key, not an Anthropic one."
        )

    base_url = get_base_url()
    # A LiteLLM key is a virtual key; the SDK sends it as x-api-key either way.
    return anthropic.Anthropic(base_url=base_url) if base_url else anthropic.Anthropic()


def make_deepseek_client():
    """OpenAI-compatible client pointed straight at DeepSeek (EMAIL_PROVIDER=deepseek).

    Uses its own DEEPSEEK_API_KEY and base URL — independent of the Anthropic/
    LiteLLM path, so it works even when the proxy is down or over budget.
    """
    from openai import OpenAI

    load_env()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit(
            "EMAIL_PROVIDER=deepseek needs DEEPSEEK_API_KEY (see .env.example)."
        )
    return OpenAI(base_url=get_deepseek_base_url(), api_key=key)


def make_deepseek_proxy_client():
    """OpenAI-compatible client pointed at the LiteLLM proxy's /v1 chat endpoint.

    Same virtual key as the Anthropic client; LiteLLM routes DeepSeek models to
    their OpenAI-shape API, so _structure_openai works unchanged against it.
    """
    from openai import OpenAI

    load_env()
    return OpenAI(base_url=get_base_url().rstrip("/") + "/v1",
                  api_key=os.environ.get("ANTHROPIC_API_KEY"))


def describe_target() -> str:
    """One-line summary of where requests are going — printed on every run."""
    where = get_base_url() or "https://api.anthropic.com (direct)"
    if get_email_provider() == "deepseek":
        backup = f"{get_deepseek_model()} via {get_deepseek_base_url()}"
        proxy = get_deepseek_proxy_model()
        email_note = (f" (email: {proxy} via proxy, backup {backup})" if proxy
                      else f" (email: {backup})")
    else:
        email = get_email_model()
        email_note = "" if email == get_model() else f" (email: {email})"
    return f"model={get_model()}{email_note} via {where}"
