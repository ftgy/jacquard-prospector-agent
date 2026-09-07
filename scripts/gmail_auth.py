#!/usr/bin/env python3
"""
One-time Gmail authorization for in-app outreach.

Run this once (and again only if you revoke access or change scopes). It opens a
browser, asks you to grant this app permission to send mail and read threads as
your account, and writes the resulting token to token.json at the project root.
The web server then uses that token silently — this interactive flow never runs
inside the server.

Prerequisite: credentials.json (a Desktop OAuth client) downloaded from Google
Cloud into the project root. See docs/gmail-setup.md for how to create it.

Usage:
    python scripts/gmail_auth.py            # authorize / re-authorize
    python scripts/gmail_auth.py --status   # show who's connected, don't change it
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # find prospector/

from prospector import gmailer


def _authorize() -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not gmailer.CREDENTIALS_PATH.exists():
        sys.exit(
            "Missing secrets/credentials.json. Download a Desktop OAuth client "
            "from Google Cloud and save it there first — see docs/gmail-setup.md."
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(gmailer.CREDENTIALS_PATH), gmailer.SCOPES
    )
    # Opens the browser and spins up a throwaway localhost server to catch the
    # redirect. port=0 lets the OS pick a free port.
    creds = flow.run_local_server(port=0, prompt="consent")
    gmailer.SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    gmailer.TOKEN_PATH.write_text(creds.to_json())

    who = gmailer.account_email() or "your account"
    print(f"\n✓ Connected {who}. Token saved to {gmailer.TOKEN_PATH.name}.")


def _status() -> None:
    if not gmailer.authorized():
        print("Not connected. Run `python scripts/gmail_auth.py` to authorize.")
        return
    print(f"Connected as {gmailer.account_email() or '(unknown)'}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true",
                    help="Show connection status without changing it.")
    args = ap.parse_args()
    _status() if args.status else _authorize()
