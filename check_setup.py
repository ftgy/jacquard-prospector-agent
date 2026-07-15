#!/usr/bin/env python3
"""
Verify your endpoint (Anthropic direct or LiteLLM) supports what this agent needs.

Run this FIRST after setting your key:

    python check_setup.py

It probes each feature independently and reports what works. A proxy may serve
plain messages fine but silently drop server-side tools or structured outputs —
this tells you which, before you burn a full run finding out.
"""

import json
import sys

import anthropic

from config import describe_target, get_base_url, get_model, load_env, make_client, using_proxy

PASS, FAIL, WARN = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"


def line(label, status, detail=""):
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))


def short(e: Exception, n: int = 130) -> str:
    return str(e).replace("\n", " ")[:n]


def check_models(client):
    """List models the endpoint exposes. LiteLLM names them per its own config."""
    print("\n1. Model availability")
    try:
        models = [m.id for m in client.models.list()]
        line("models.list()", PASS, f"{len(models)} available")
        for m in models[:20]:
            print(f"        - {m}")
        if len(models) > 20:
            print(f"        ... and {len(models) - 20} more")
        if get_model() not in models:
            line(f"configured model '{get_model()}'", WARN,
                 "not in list — set PROSPECT_MODEL in .env to one above")
            return models, False
        line(f"configured model '{get_model()}'", PASS, "available")
        return models, True
    except Exception as e:
        line("models.list()", WARN, f"unsupported here: {short(e)}")
        return [], True  # not fatal; proxies often omit this endpoint


def check_basic(client):
    print("\n2. Basic message (required)")
    try:
        r = client.messages.create(
            model=get_model(), max_tokens=32,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        line("messages.create()", PASS, f"replied {text!r}")
        return True
    except anthropic.AuthenticationError as e:
        line("messages.create()", FAIL, short(e))
        if using_proxy():
            print(f"        -> {get_base_url()} rejected the key. It wants your LiteLLM")
            print("           VIRTUAL key (from the LiteLLM UI / /key/generate),")
            print("           not an Anthropic sk-ant-... key.")
        else:
            print("        -> Check ANTHROPIC_API_KEY at console.anthropic.com.")
        return False
    except Exception as e:
        line("messages.create()", FAIL, short(e))
        if isinstance(e, anthropic.NotFoundError) and using_proxy():
            print(f"        -> '{get_model()}' isn't served by this proxy. Set")
            print("           PROSPECT_MODEL in .env to a model listed in step 1.")
        return False


def check_thinking(client):
    print("\n3. Adaptive thinking (used by every stage)")
    try:
        client.messages.create(
            model=get_model(), max_tokens=1024,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": "What is 17 * 23?"}],
        )
        line("thinking={'type':'adaptive'}", PASS)
        return True
    except Exception as e:
        line("thinking={'type':'adaptive'}", FAIL, short(e))
        print("        -> Fix: drop the thinking param in agent.py (_search/_structure).")
        return False


def check_structured(client):
    """Native schema enforcement is optional — agent.py falls back to prompted JSON."""
    print("\n4. Structured outputs (qualify + discovery need JSON)")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    native = False
    try:
        r = client.messages.create(
            model=get_model(), max_tokens=256,
            messages=[{"role": "user", "content": "Capital of France? JSON."}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        json.loads(text)
        line("native output_config.format", PASS, "schema enforced")
        native = True
    except json.JSONDecodeError:
        # Worst failure mode: accepted, ignored, prose returned. No error raised.
        line("native output_config.format", WARN, "accepted but SILENTLY IGNORED")
    except Exception as e:
        line("native output_config.format", WARN, f"unsupported: {short(e, 80)}")

    # What actually matters: can the agent get reliable JSON either way?
    from agent import _structure  # imported late so this file works standalone
    try:
        out = _structure(client, "You answer geography questions.",
                         "Capital of France?", schema, 256)
        assert isinstance(out.get("answer"), str)
        mode = "native" if native else "prompted fallback"
        line(f"agent JSON pipeline ({mode})", PASS, f"got {out}")
        return True
    except Exception as e:
        line("agent JSON pipeline", FAIL, short(e))
        return False


def check_search(client):
    """Verify the configured search backend actually reaches the live web."""
    from search import get_search_backend, get_search_model
    backend = get_search_backend()
    print(f"\n5. Web search — backend: {backend} (research + discovery need this)")
    if backend == "gemini":
        return check_gemini_search()
    return check_anthropic_web_search(client)


def check_gemini_search():
    """A grounded model must answer something it cannot know from training data.

    The probe has to *force* a search: grounding is the model's choice, and a
    trivial question ("what's today's date?") gets answered from context with no
    citations — which looks identical to grounding being switched off.
    """
    from search import get_search_model, grounded_search
    try:
        out = grounded_search(
            "You are a web researcher. Cite your sources.",
            "Search the web and name two recruiting agencies headquartered in "
            "Barcelona, with their websites.", 1024)
    except Exception as e:
        line(f"grounded search via {get_search_model()}", FAIL, short(e))
        print("        -> Check SEARCH_MODEL in .env is served by your proxy.")
        return False
    if not out["sources"]:
        # Answered without citations => not actually grounded, just recalling.
        line(f"grounded search via {get_search_model()}", FAIL,
             "replied but cited no sources — grounding appears off")
        print("        -> The model answered from memory. Check that your proxy")
        print("           passes web_search_options through to Vertex.")
        return False
    line(f"grounded search via {get_search_model()}", PASS,
         f"{len(out['sources'])} sources, e.g. {out['sources'][0]['title']}")
    print(f"        reply: {out['text'][:70].strip()}")
    return True


def check_anthropic_web_search(client):
    """Try both variants: Vertex serves only the older one, 1P/Bedrock the newer."""
    errors = []
    for variant in ("web_search_20260209", "web_search_20250305"):
        try:
            r = client.messages.create(
                model=get_model(), max_tokens=2048,
                tools=[{"type": variant, "name": "web_search", "max_uses": 2}],
                messages=[{"role": "user",
                           "content": "Search the web: who founded Anthropic?"}],
            )
            searched = any(getattr(b, "type", "") in
                           ("web_search_tool_result", "server_tool_use") for b in r.content)
            if searched:
                line(variant, PASS, "server-side search executed")
                if variant != "web_search_20260209":
                    print(f"        -> Set WEB_SEARCH_TOOL={variant} in .env")
                return True
            line(variant, FAIL, "accepted but never searched (silently dropped)")
        except Exception as e:
            line(variant, FAIL, short(e, 90))
            errors.append(str(e))

    blob = " ".join(errors)
    if "allowedPartnerModelFeatures" in blob or "Organization Policy" in blob:
        print("        -> Your GCP org policy blocks web_search for Anthropic partner")
        print("           models. A GCP org admin must allow:")
        print("             publishers/anthropic/models/<model>:web_search")
        print("           constraint: constraints/vertexai.allowedPartnerModelFeatures")
    print("        -> Fix: set SEARCH_BACKEND=gemini in .env to search with a")
    print("           grounded Gemini model instead (first-party on Vertex, so")
    print("           partner-model policies don't apply).")
    return False


def main():
    load_env()
    print("=" * 72)
    print("  SETUP CHECK —", describe_target())
    print("=" * 72)
    if using_proxy():
        print(f"\nUsing proxy: {get_base_url()}")
        print("Note: your ANTHROPIC_API_KEY should be the LiteLLM virtual key.")

    try:
        client = make_client()
    except SystemExit as e:
        sys.exit(str(e))

    check_models(client)
    if not check_basic(client):
        print("\nBasic messaging failed — nothing else can work. Fix this first.\n")
        sys.exit(1)

    thinking_ok = check_thinking(client)
    json_ok = check_structured(client)
    search_ok = check_search(client)

    print("\n" + "=" * 72)
    if json_ok and search_ok:
        print("  Ready. Run:  python main.py --discover \"<niche>\" --discover-only")
    elif json_ok and not search_ok:
        print("  BLOCKED: no web search on this endpoint.")
        print("  Reasoning and JSON work — but discovery and research both need the")
        print("  web, so the agent can't run as written. Resolve web search first")
        print("  (see options above), then re-run this check.")
    else:
        print("  BLOCKED: this endpoint can't produce reliable JSON.")
        print("  See the fix notes above and the README's Troubleshooting section.")
    if not thinking_ok:
        print("  Also: adaptive thinking unsupported — remove it from agent.py.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
