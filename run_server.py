#!/usr/bin/env python3
"""
Launch the dashboard web server.

  python run_server.py                 # http://127.0.0.1:8000

Thin entrypoint so the FastAPI app (a package module using relative imports) can
be started from the project root. Equivalent to `python -m prospector.server`.
"""

import uvicorn

from prospector.server import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
