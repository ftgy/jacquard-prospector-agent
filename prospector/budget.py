"""
How much of the LiteLLM key's budget is left, for the dashboard header.

The virtual key may only call LLM routes, so /key/info is off-limits. But
LiteLLM stamps every chat response with x-litellm-key-max-budget and
x-litellm-key-spend — including a 403 for a model the key isn't allowed to use,
which costs nothing. So the probe asks for such a model first, and only falls
back to a real 1-token call on a cheap model if that stops returning the headers.
The budget is per key, shared by every model on it.
"""

import os
import time

import httpx

from .config import get_base_url, load_env, use_system_ca_bundle

# A model this key is denied (free 403 that still carries the budget headers).
DEFAULT_FREE_PROBE = "qwen/qwen3-coder-30b-a3b-instruct"
# Cheapest allowed model, for when the free probe stops carrying the headers.
DEFAULT_PAID_PROBE = "gemini-3.7-flash"

CACHE_SECONDS = 60
_cache: dict = {"at": 0.0, "value": None}


def _probe(base: str, key: str, model: str) -> tuple[float, float] | None:
    """(max_budget, spend) from one chat call's headers, or None if absent."""
    r = httpx.post(f"{base}/v1/chat/completions", timeout=20,
                   headers={"Authorization": f"Bearer {key}"},
                   json={"model": model, "max_tokens": 1,
                         "messages": [{"role": "user", "content": "hi"}]})
    spend = r.headers.get("x-litellm-key-spend")
    if spend is None:
        return None
    max_budget = r.headers.get("x-litellm-key-max-budget")
    return (float(max_budget) if max_budget not in (None, "", "None") else None,
            float(spend))


def llm_budget(force: bool = False) -> dict:
    """{available, max_budget, spend, remaining} for the LiteLLM key.

    available=False (with a reason) when there's no proxy or the headers never
    came back. max_budget/remaining are None for a key with no budget cap.
    Cached for CACHE_SECONDS so page loads don't hammer the proxy.
    """
    if not force and _cache["value"] and time.time() - _cache["at"] < CACHE_SECONDS:
        return _cache["value"]

    load_env()
    base = get_base_url()
    if not base:
        return {"available": False, "reason": "Not using a LiteLLM proxy."}
    use_system_ca_bundle()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    base = base.rstrip("/")

    got = None
    try:
        for model in (os.environ.get("BUDGET_PROBE_MODEL", DEFAULT_FREE_PROBE),
                      os.environ.get("BUDGET_PAID_PROBE_MODEL", DEFAULT_PAID_PROBE)):
            got = _probe(base, key, model)
            if got:
                break
    except httpx.HTTPError as e:
        return {"available": False, "reason": f"Proxy unreachable: {e}"}
    if not got:
        return {"available": False, "reason": "Proxy didn't report the key's budget."}

    max_budget, spend = got
    value = {"available": True, "max_budget": max_budget, "spend": round(spend, 4),
             "remaining": round(max_budget - spend, 4) if max_budget is not None else None}
    _cache.update(at=time.time(), value=value)
    return value
