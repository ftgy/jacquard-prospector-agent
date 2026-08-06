#!/usr/bin/env python3
"""
Check which models your LiteLLM (or Anthropic) endpoint can actually serve.

The proxy can list a model in /v1/models yet fail every call to it with
"no healthy deployments" (LiteLLM benches a deployment after upstream errors).
Because that state flaps, this probes each model SEVERAL times and reports a
success rate — a single OK isn't proof the model is usable.

Usage:
    python scripts/health_check.py                  # all listed models, 3 tries each
    python scripts/health_check.py -n 10            # 10 tries each (spot flakiness)
    python scripts/health_check.py -m vertex_ai/claude-opus-4-8,gemini-3.5-flash
    python scripts/health_check.py --base-url http://host:4000 --api-key sk-...

Endpoint + key are read from (in order): CLI flags, environment, then the
project .env (ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY). Stdlib only — runs under
any python3 without installing anything.

Exit code: 0 if every tested model answered every time; 1 if any model was
degraded or down (handy in scripts / cron).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> dict:
    """Minimal .env reader (KEY=VALUE lines); no export, no quotes handling frills."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def resolve(name: str, cli_value: str | None, env_file: dict) -> str | None:
    return cli_value or os.environ.get(name) or env_file.get(name)


def api_base(base_url: str) -> str:
    """Normalize to a URL ending in /v1 (the SDK appends it; raw curl needs it)."""
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def _post(url: str, key: str, payload: dict, timeout: float):
    """POST JSON. Returns (status_code, parsed_body_or_text). Never raises."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except Exception as e:  # timeout, connection refused, DNS, ...
        return None, f"{type(e).__name__}: {e}"


def list_models(base: str, key: str, timeout: float) -> list[str]:
    url = base + "/models"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except Exception as e:
        sys.exit(f"Could not list models from {url}: {e}")
    ids = [m.get("id") for m in body.get("data", []) if m.get("id")]
    return sorted(ids)


def error_message(status, body) -> str:
    """Pull a short, human error out of whatever the proxy returned."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if body.get("message"):
            return str(body["message"])
    text = body if isinstance(body, str) else json.dumps(body)
    return f"HTTP {status}: {text}"


def probe(base: str, key: str, model: str, tries: int, timeout: float,
          prompt: str) -> dict:
    """Hit one model `tries` times. Returns counts, latency, and a sample error."""
    payload = {
        "model": model,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": prompt}],
    }
    ok = 0
    sample_err = ""
    latencies = []
    for _ in range(tries):
        t0 = time.monotonic()
        status, body = _post(base + "/chat/completions", key, payload, timeout)
        latencies.append(time.monotonic() - t0)
        healthy = status == 200 and isinstance(body, dict) and "error" not in body
        if healthy:
            ok += 1
        elif not sample_err:
            sample_err = error_message(status, body)
    return {
        "model": model, "ok": ok, "tries": tries, "error": sample_err,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def classify(ok: int, tries: int) -> tuple[str, str]:
    """(mark, label) for a result."""
    if ok == tries:
        return "✓", "healthy"
    if ok == 0:
        return "✗", "DOWN"
    return "~", "flaky"


def shorten(msg: str, width: int = 88) -> str:
    msg = " ".join(msg.split())  # collapse newlines/whitespace
    return msg if len(msg) <= width else msg[: width - 1] + "…"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Health-check LiteLLM / Anthropic models via repeated probes.")
    ap.add_argument("-m", "--models",
                    help="Comma-separated model ids to test (default: all listed).")
    ap.add_argument("-n", "--repeat", type=int, default=3,
                    help="Probes per model (default: 3). Raise it to spot flapping.")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="Per-request timeout in seconds (default: 30).")
    ap.add_argument("--prompt", default="Reply with the single word: ok",
                    help="Prompt sent on each probe.")
    ap.add_argument("--base-url", help="Endpoint (else $ANTHROPIC_BASE_URL / .env).")
    ap.add_argument("--api-key", help="Key (else $ANTHROPIC_API_KEY / .env).")
    args = ap.parse_args()

    env_file = load_env_file(ROOT / ".env")
    base_url = resolve("ANTHROPIC_BASE_URL", args.base_url, env_file)
    key = resolve("ANTHROPIC_API_KEY", args.api_key, env_file)
    if not base_url:
        sys.exit("No endpoint. Set ANTHROPIC_BASE_URL (.env) or pass --base-url.")
    if not key:
        sys.exit("No API key. Set ANTHROPIC_API_KEY (.env) or pass --api-key.")

    base = api_base(base_url)
    print(f"Endpoint: {base}")

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = list_models(base, key, args.timeout)
        print(f"Discovered {len(models)} model(s) from /models.")
    print(f"Probing each {args.repeat}× (timeout {args.timeout:g}s)…\n")

    name_w = max((len(m) for m in models), default=5)
    print(f"     {'MODEL'.ljust(name_w)}   OK/N   RATE    LAT")
    print(f"     {'-' * name_w}   ----   ----    ---")

    results = []
    for model in models:
        r = probe(base, key, model, args.repeat, args.timeout, args.prompt)
        results.append(r)
        mark, _ = classify(r["ok"], r["tries"])
        rate = f"{100 * r['ok'] // r['tries']:>3}%"
        lat = f"{r['avg_latency']:.1f}s"
        print(f"  {mark}  {model.ljust(name_w)}   {r['ok']}/{r['tries']}   {rate}   {lat:>5}")
        if r["error"]:
            print(f"       ↳ {shorten(r['error'])}")

    healthy = [r for r in results if r["ok"] == r["tries"]]
    flaky = [r for r in results if 0 < r["ok"] < r["tries"]]
    down = [r for r in results if r["ok"] == 0]
    print(f"\nSummary: {len(healthy)} healthy, {len(flaky)} flaky, {len(down)} down "
          f"(of {len(results)} tested).")
    if flaky or down:
        names = ", ".join(r["model"] for r in flaky + down)
        print(f"Needs attention: {names}")
    return 0 if not (flaky or down) else 1


if __name__ == "__main__":
    sys.exit(main())
