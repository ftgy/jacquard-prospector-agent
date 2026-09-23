#!/usr/bin/env python3
"""
Launch the dashboard web server.

  python run_server.py                 # http://127.0.0.1:8082
  python run_server.py --no-reload     # as the prospector-web daemon runs it

Thin entrypoint so the FastAPI app (a package module using relative imports) can
be started from the project root. Equivalent to `python -m prospector.server`.

Autoreload is on by default: edits to the prospector/ package restart the server
automatically. uvicorn needs the app as an import string for reload to work, so
we pass "prospector.server:app" rather than the imported object. (The dashboard
HTML is served no-cache, so its edits just need a browser refresh, not a
restart.)

The daemon (deploy/prospector-web.service) passes --no-reload: a reload restarts
the process, which would cut a draft run or the auto-queue loop off halfway.
"""

import sys

import uvicorn

if __name__ == "__main__":
    reload = "--no-reload" not in sys.argv[1:]
    uvicorn.run(
        "prospector.server:app",
        host="127.0.0.1",
        port=8082,
        reload=reload,
        reload_dirs=["prospector"] if reload else None,
    )
