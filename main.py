#!/usr/bin/env python3
"""
Prospect research & qualification agent.

Usage:
  python main.py                          # reads prospects.csv (or prospects.example.csv)
  python main.py --file my_leads.csv      # CSV with a 'company' column (optional 'hint')
  python main.py --companies "Acme Inc" "Globex"   # qualify companies passed inline
  python main.py --out results.json       # where to write full results (default: results.json)

Requires ANTHROPIC_API_KEY in the environment (or a .env file next to this script).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import anthropic

from agent import discover_candidates, run_prospect
from config import describe_target, load_env, make_client, using_proxy
from icp import ICP

HERE = Path(__file__).parent


def friendly_api_error(e: Exception) -> str:
    """Turn common API failures into something actionable instead of a traceback."""
    msg = str(e)
    if isinstance(e, anthropic.APIConnectionError):
        cause = repr(e.__cause__ or "")
        if "CERTIFICATE_VERIFY_FAILED" in cause:
            return ("TLS verification failed — this network likely runs an intercepting "
                    "proxy. Point Python at your system CA bundle:\n"
                    "  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt")
        return f"Could not reach the API (check your network): {msg}"
    if isinstance(e, anthropic.AuthenticationError):
        return "ANTHROPIC_API_KEY is invalid or revoked. Check .env / console.anthropic.com."
    if "credit balance is too low" in msg:
        return ("Your Anthropic account is out of credits. Add credits at\n"
                "  https://console.anthropic.com -> Plans & Billing\n"
                "(Or point ANTHROPIC_BASE_URL at your LiteLLM instance instead.)")
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited. Wait a moment and retry, or lower --count."
    if isinstance(e, anthropic.NotFoundError) and using_proxy():
        return (f"Endpoint or model not found on the proxy: {msg}\n"
                "Run `python check_setup.py` to list models it actually serves, "
                "then set PROSPECT_MODEL in .env.")
    return msg


def read_prospects(path: Path) -> list:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            company = (row.get("company") or "").strip()
            if company:
                rows.append({"company": company, "hint": (row.get("hint") or "").strip()})
    return rows


def default_csv() -> Path:
    for name in ("prospects.csv", "prospects.example.csv"):
        p = HERE / name
        if p.exists():
            return p
    sys.exit("No prospects file found. Create prospects.csv or pass --companies.")


def print_report(results: list):
    ranked = sorted(results, key=lambda r: r.get("fit_score", -1), reverse=True)
    print("\n" + "=" * 70)
    print("  PROSPECT QUALIFICATION — ranked by fit")
    print("=" * 70)
    for r in ranked:
        if "error" in r:
            print(f"\n[!] {r['company']}: ERROR — {r['error']}")
            continue
        print(f"\n[{r['tier']}] {r['company']}  —  fit {r['fit_score']}/100 "
              f"(confidence: {r['confidence']})")
        print(f"    {r['one_line']}")
        if r.get("pain_points"):
            print("    Pain points an agent could solve:")
            for p in r["pain_points"][:3]:
                print(f"      • {p['pain']} → {p['agent_solution']}")
        if r.get("buying_signals"):
            print(f"    Buying signals: {'; '.join(r['buying_signals'][:3])}")
        if r.get("red_flags"):
            print(f"    Red flags: {'; '.join(r['red_flags'][:3])}")
        print(f"    Outreach angle: {r['outreach_angle']}")
    print("\n" + "=" * 70 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Find, research & qualify B2B prospects.")
    ap.add_argument("--file", type=Path, help="CSV with a 'company' column.")
    ap.add_argument("--companies", nargs="+", help="Company names to qualify inline.")
    ap.add_argument("--discover", metavar="NICHE",
                    help='Find companies in a niche, e.g. "recruiting agencies in Barcelona".')
    ap.add_argument("--count", type=int, default=10,
                    help="How many companies to discover (default: 10).")
    ap.add_argument("--discover-only", action="store_true",
                    help="List discovered companies without qualifying them (cheap preview).")
    ap.add_argument("--out", type=Path, default=HERE / "results.json",
                    help="Where to write full JSON results.")
    args = ap.parse_args()

    load_env()
    client = make_client()
    print(f"Using {describe_target()}")

    if args.discover:
        print(f"Discovering ~{args.count} companies for: {args.discover}...", flush=True)
        try:
            candidates = discover_candidates(client, args.discover, ICP, args.count)
        except Exception as e:
            sys.exit(f"\nDiscovery failed: {friendly_api_error(e)}")
        if not candidates:
            sys.exit("Discovery found no companies. Try a broader or more specific niche.")
        print(f"Found {len(candidates)}:")
        for c in candidates:
            print(f"  • {c['company']} ({c['website']}) — {c['why_candidate']}")
        if args.discover_only:
            args.out.write_text(json.dumps(candidates, indent=2))
            print(f"\nCandidates written to {args.out}")
            print("Re-run without --discover-only to research and qualify them.")
            return
        prospects = [{"company": c["company"], "hint": c["hint"]} for c in candidates]
        print()
    elif args.companies:
        prospects = [{"company": c, "hint": ""} for c in args.companies]
    else:
        prospects = read_prospects(args.file or default_csv())

    if not prospects:
        sys.exit("No prospects to process.")

    results = []
    for i, p in enumerate(prospects, 1):
        print(f"[{i}/{len(prospects)}] Researching {p['company']}...", flush=True)
        try:
            results.append(run_prospect(client, p["company"], ICP, p["hint"]))
        except Exception as e:  # one bad company shouldn't kill the batch
            reason = friendly_api_error(e)
            print(f"    failed: {reason}", flush=True)
            results.append({"company": p["company"], "error": reason})

    args.out.write_text(json.dumps(results, indent=2))
    print_report(results)
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
