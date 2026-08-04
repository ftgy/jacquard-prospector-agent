#!/usr/bin/env python3
"""
Launch the dashboard web server.

  python run_server.py                 # http://127.0.0.1:8000

Thin entrypoint so the FastAPI app (a package module using relative imports) can
be started from the project root. Equivalent to `python -m prospector.server`.

Autoreload is on: edits to the prospector/ package restart the server
automatically. uvicorn needs the app as an import string for reload to work, so
we pass "prospector.server:app" rather than the imported object. (The dashboard
HTML is served no-cache, so its edits just need a browser refresh, not a
restart.)
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "prospector.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["prospector"],
    )
