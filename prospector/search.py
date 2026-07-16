"""
Grounded web search.

Anthropic's `web_search` server tool is unavailable on some routes (e.g. Claude
served via Vertex behind a GCP org policy that blocks partner-model features).
This module provides search through a *separate* grounded model instead, so the
Claude stages keep doing what they're good at — reasoning and scoring — while a
search-capable model fetches facts.

Default backend is Gemini + Google Search grounding via an OpenAI-compatible
endpoint (LiteLLM). Gemini is first-party on Vertex, so partner-model policies
don't apply to it.

Configure in .env:
    SEARCH_BACKEND=gemini|anthropic   (default: gemini when behind a proxy)
    SEARCH_MODEL=gemini-3.5-flash
"""

import os

from openai import OpenAI

from .config import get_base_url


def get_search_backend() -> str:
    """'gemini' (separate grounded model) or 'anthropic' (native web_search tool)."""
    default = "gemini" if get_base_url() else "anthropic"
    return os.environ.get("SEARCH_BACKEND", default).lower()


def get_search_model() -> str:
    return os.environ.get("SEARCH_MODEL", "gemini-3.5-flash")


def _openai_client() -> OpenAI:
    """OpenAI-compatible client. Gemini speaks this shape, not /v1/messages."""
    base = get_base_url()
    if not base:
        raise RuntimeError(
            "SEARCH_BACKEND=gemini needs ANTHROPIC_BASE_URL (your LiteLLM URL)."
        )
    return OpenAI(base_url=base.rstrip("/") + "/v1",
                  api_key=os.environ["ANTHROPIC_API_KEY"])


def grounded_search(system: str, ask: str, max_tokens: int = 4000) -> dict:
    """Answer `ask` using live web search. Returns {'text': ..., 'sources': [...]}.

    Sources come from the model's citation annotations, so they reflect pages it
    actually consulted rather than anything it invented.
    """
    client = _openai_client()
    resp = client.chat.completions.create(
        model=get_search_model(),
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": ask}],
        max_tokens=max_tokens,
        web_search_options={},  # enables Google Search grounding
    )
    msg = resp.choices[0].message
    return {"text": (msg.content or "").strip(),
            "sources": _sources_from_annotations(msg)}


def _sources_from_annotations(msg) -> list:
    """Pull deduplicated {title, url} out of url_citation annotations.

    Titles are the real source domains; URLs are Vertex grounding redirects that
    still resolve to the original page.
    """
    seen, out = set(), []
    for a in (getattr(msg, "annotations", None) or []):
        uc = getattr(a, "url_citation", None) or (
            a.get("url_citation") if isinstance(a, dict) else None)
        if not uc:
            continue
        title = getattr(uc, "title", None) or (uc.get("title") if isinstance(uc, dict) else None)
        url = getattr(uc, "url", None) or (uc.get("url") if isinstance(uc, dict) else None)
        key = title or url
        if key and key not in seen:
            seen.add(key)
            out.append({"title": title or "", "url": url or ""})
    return out
