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
DEFAULT_MODEL = "claude-opus-4-8"


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


def describe_target() -> str:
    """One-line summary of where requests are going — printed on every run."""
    where = get_base_url() or "https://api.anthropic.com (direct)"
    return f"model={get_model()} via {where}"
