# Prospect Research & Qualification Agent

An AI agent that **finds** companies in a niche, researches them on the web, and
scores them as potential clients against **your** Ideal Customer Profile — so you
can find and prioritize businesses to pitch AI-automation services to.

Built on the Claude API. Search and reasoning are **separate, swappable backends**,
so it runs against whatever your endpoint actually supports:

- **Search** — a grounded model fetches live facts with citations
  (Gemini + Google Search grounding, or Claude's native `web_search` tool).
- **Reasoning + scoring** — Claude does the judgement work.

**Stage 0 — Discover** *(optional)*: give it a niche ("recruiting agencies in
Barcelona") and it finds real companies that plausibly fit your ICP. Cheap wide net.

Then, per prospect:

1. **Research** — Claude searches the web for what the company does, its size,
   tech maturity, hiring/growth signals, and manual workflows worth automating.
2. **Qualify** — Claude scores that research against your ICP and returns a
   structured verdict: fit score, tier (A/B/C/disqualified), pain points +
   how an agent solves each, buying signals, red flags, and an outreach angle.

## Setup

```bash
cd leads_research_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then edit .env — see below
```

You can run against **either** endpoint. Both use the official Anthropic SDK;
a proxy is just a `base_url` change, since LiteLLM serves the same `/v1/messages`
endpoint the SDK speaks.

**Option A — LiteLLM** (no Anthropic credits needed). In `.env`:

```bash
ANTHROPIC_BASE_URL=http://192.168.132.144:4000
ANTHROPIC_API_KEY=<your LiteLLM virtual key>   # NOT an sk-ant-... key
PROSPECT_MODEL=vertex_ai/claude-opus-4-8       # LiteLLM's name for the model
SEARCH_BACKEND=gemini                          # see "How search works" below
SEARCH_MODEL=gemini-3.5-flash
```

**Option B — Anthropic direct.** Leave `ANTHROPIC_BASE_URL` unset and use a real
`sk-ant-...` key (the account needs credits).

### Then verify your endpoint

```bash
python scripts/check_setup.py
```

**Run this before anything else.** It lists the models your endpoint serves and
probes the three features this agent needs — adaptive thinking, structured
outputs, and the `web_search` server tool. A proxy can serve plain messages
perfectly while silently dropping server-side tools, so this tells you what
works *before* you spend a run finding out.

## Run

```bash
# Discover companies in a niche, then research + qualify each:
python main.py --discover "recruiting agencies in Barcelona" --count 10

# Preview who it finds WITHOUT paying for deep research (cheap, fast):
python main.py --discover "e-commerce brands in Madrid" --count 15 --discover-only

# Qualify companies you already have:
python main.py --companies "Acme Inc" "Globex Corp"

# Or from a CSV (a 'company' column, optional 'hint' column):
python main.py --file prospects.csv
```

It prints a ranked report to the terminal, writes full details (including
research summaries and source URLs) to `results.json`, **and stores every
prospect in a SQLite database** (`prospector.db`) so runs accumulate over time.
Pass `--no-db` to skip the database for a run.

**Tip:** run `--discover-only` first to sanity-check the niche and eyeball the
company list, then re-run without it to qualify. Discovery is ~2 API calls total;
qualification is ~2 calls *per company*, so previewing first saves real money.

## Dashboard

A browser panel for the stored prospects — browse, filter, and launch new
research runs with live progress.

```bash
python run_server.py          # then open http://127.0.0.1:8000
```

To backfill the dashboard from an existing `results.json`:

```bash
python scripts/import_results.py
```

The panel shows ranked prospects with fit-signal meters and a tier breakdown,
a detail drawer (pain points → agent solutions, buying signals, sources), and a
run panel that discovers a niche or qualifies named companies, streaming results
into the table as each company finishes. Both light and dark themes.

## How search works (and why it's swappable)

`web_search` is an **Anthropic server-side tool**, not a model capability — so it
only works if your endpoint routes to Anthropic and permits the feature. Two
backends, set via `SEARCH_BACKEND`:

| `SEARCH_BACKEND` | What it does | Use when |
|---|---|---|
| `gemini` | Gemini + Google Search grounding, over the OpenAI-compatible endpoint. Returns cited sources. | Anthropic's `web_search` is unavailable (default behind a proxy). |
| `anthropic` | Claude's native `web_search` server tool. | You're on the Anthropic API directly, or a proxy that passes it through. |

Either way Claude still does all reasoning and scoring — only fact-gathering moves.

**Why the default is `gemini` here:** this LiteLLM instance serves Claude via
Vertex AI, where a GCP org policy
(`constraints/vertexai.allowedPartnerModelFeatures`) blocks `web_search` for
Anthropic *partner* models. Gemini is *first-party* on Vertex, so that policy
doesn't apply and its grounding works. `check_setup.py` verifies which you have.

## Tune your targeting

Edit **`prospector/icp.py`** — that one file describes who you are and what a
good-fit client looks like. The qualifier reads it verbatim; the sharper it is,
the better the scores. No other code needs to change.

## Project layout

The code lives in a `prospector/` package; entry points sit at the project root.

| Path | What it does |
|------|--------------|
| `prospector/icp.py`     | Your Ideal Customer Profile — **edit this to change targeting**. |
| `prospector/config.py`  | Client + model setup; switches between Anthropic and LiteLLM. |
| `prospector/search.py`  | Grounded web search; swappable backend (Gemini / Anthropic). |
| `prospector/agent.py`   | Core: `discover_candidates()`, `research_company()`, `qualify_company()`, `run_prospect()`. |
| `prospector/db.py`      | SQLite persistence (`prospector.db`): prospects + runs. |
| `prospector/service.py` | Bridges the agent and the database; shared by the CLI and the web server. |
| `prospector/server.py`  | FastAPI backend + dashboard (`prospector/static/index.html`). |
| `main.py`               | CLI entry: discovers/reads prospects, runs the pipeline, prints + saves. |
| `run_server.py`         | Web entry: launches the dashboard server. |
| `scripts/check_setup.py`   | Diagnostic: what your endpoint serves and supports. **Run first.** |
| `scripts/import_results.py`| Backfill the database from an existing `results.json`. |
| `prospects.example.csv` | Sample input; copy to `prospects.csv` and fill in your leads. |

## Troubleshooting

**"Your credit balance is too low"** — the API key works, but the account needs
credits. Add them at console.anthropic.com → Plans & Billing, or point
`ANTHROPIC_BASE_URL` at LiteLLM instead.

**LiteLLM: "Invalid proxy server token passed"** — you're sending an Anthropic
key. LiteLLM wants its own *virtual key* (LiteLLM UI, or `POST /key/generate`).

**LiteLLM: `NotFoundError` / unknown model** — LiteLLM names models per its own
config, which may not match Anthropic's public IDs. Run `check_setup.py` to list
what it serves, then set `PROSPECT_MODEL` in `.env`.

**`web_search` rejected / "Organization Policy constraint ... allowedPartnerModelFeatures"**
— your GCP org blocks the feature for Anthropic partner models on Vertex. Either
have a GCP admin allow `publishers/anthropic/models/<model>:web_search`, or just
set `SEARCH_BACKEND=gemini` (what this project does by default).

**Search returns no sources / seems to answer from memory** — grounding is the
model's own choice; it may skip searching for questions it thinks it knows. The
research prompts force a search. If `check_setup.py` reports "cited no sources",
confirm your proxy passes `web_search_options` through to Vertex.

**Structured outputs "silently ignored"** — expected on this route: LiteLLM/Vertex
accepts `output_config.format` and returns prose anyway. `agent.py` detects this
via `config.use_native_structured_output()` and falls back to prompted JSON with
defensive parsing. Force either mode with `STRUCTURED_OUTPUT=native|prompted`.

**TLS / `CERTIFICATE_VERIFY_FAILED`** — some networks (corporate/VPN) intercept
TLS. `curl` works because it trusts the system CA store; Python doesn't, because
the SDK verifies against `certifi`'s bundle. `main.py` auto-points at the system
bundle, but if you call `agent.py` directly from your own script:

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

Never "fix" this by disabling verification — this keeps verification on, just
against the trust store that already has the proxy's CA.

## Cost

Each prospect is ~2 Claude calls (research + qualify) plus web searches. Opus 4.8
is $5/$25 per million tokens; expect a few cents per prospect. To cut cost, swap
`MODEL` in `agent.py` to `claude-sonnet-5` ($3/$15) — quality stays strong for
this task.

## Ideas to extend

- **CRM push**: write tier-A results to a Google Sheet, Airtable, or HubSpot.
- **Draft outreach**: add a stage that turns the outreach angle into a full email.
- **Dedup / caching**: skip companies already in `results.json`.
```
