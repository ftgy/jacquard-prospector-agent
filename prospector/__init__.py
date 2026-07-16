"""
Prospector — a prospect discovery, research & qualification agent.

Package layout:
  config, icp, search   — configuration, ICP definition, grounded web search
  agent                 — discovery + research + qualification pipeline (the engine)
  db, service           — SQLite persistence and the agent<->db service layer
  server                — FastAPI backend + dashboard (static/index.html)

Entry points live at the project root: main.py (CLI) and run_server.py (web);
one-off scripts live in scripts/.
"""
