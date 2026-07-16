"""
Core agent: find companies, research them, and qualify them against your ICP.

Discovery (optional, given a niche):
  0. discover_candidates() -> real companies matching a niche + your ICP

Then per prospect:
  1. research_company()  -> web research + sources (uses the web_search server tool)
  2. qualify_company()   -> structured scoring against your ICP (structured output)

Every stage uses adaptive thinking and the model from config.get_model(). Stages
that need the web search first, then a second call structures the result — mixing
the web_search server tool with structured output in one call is unreliable.
"""

import json
import re

import anthropic

from .config import get_model, get_web_search_tool, use_native_structured_output
from .search import get_search_backend, grounded_search


def _web_search_tool() -> dict:
    return {"type": get_web_search_tool(), "name": "web_search", "max_uses": 6}


def _search(client: anthropic.Anthropic, system: str, ask: str,
            max_tokens: int = 4000) -> dict:
    """Answer `ask` using live web search. Returns {'text', 'sources'}.

    Two backends (see search.py): a separate grounded model (Gemini), or Claude's
    own web_search server tool where that's actually available.
    """
    if get_search_backend() == "gemini":
        return grounded_search(system, ask, max_tokens)
    return _anthropic_search(client, system, ask, max_tokens)


def _anthropic_search(client: anthropic.Anthropic, system: str, ask: str,
                      max_tokens: int) -> dict:
    """Claude + the web_search server tool, resuming if the search loop pauses."""
    messages = [{"role": "user", "content": ask}]
    for _ in range(6):
        resp = client.messages.create(
            model=get_model(),
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            tools=[_web_search_tool()],
            messages=messages,
        )
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        break
    text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
    return {"text": text, "sources": _extract_sources(resp)}


def _extract_json(text: str) -> dict:
    """Parse JSON from a model reply that may be fenced or have prose around it."""
    text = text.strip()
    if text.startswith("```"):  # strip ```json fences
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")  # first..last brace
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"No JSON object found in reply: {text[:200]!r}")


def _structure(client: anthropic.Anthropic, system: str, ask: str, schema: dict,
               max_tokens: int = 4000) -> dict:
    """One structured turn. Returns parsed JSON.

    Uses native schema enforcement where it's actually honored; otherwise asks
    for JSON in the prompt and parses defensively. Proxies commonly accept
    output_config and ignore it, so prompted mode is the safe default there
    (see config.use_native_structured_output). One retry on unparseable output.
    """
    native = use_native_structured_output()
    kwargs = {}
    if native:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    else:
        system += (
            "\n\nReply with ONE JSON object and nothing else — no prose, no "
            "markdown fences. It must match this JSON Schema exactly:\n"
            + json.dumps(schema, indent=2)
        )

    messages = [{"role": "user", "content": ask}]
    for attempt in range(2):
        resp = client.messages.create(
            model=get_model(),
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            messages=messages,
            **kwargs,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return _extract_json(text)
        except (json.JSONDecodeError, ValueError):
            if attempt == 1:
                raise
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "That was not valid JSON. Reply with "
                                            "ONLY the JSON object, no other text."},
            ]
    raise AssertionError("unreachable")


# --- Stage 0: discovery ------------------------------------------------------

DISCOVERY_SYSTEM = """You find real, verifiable companies that a consultant could \
pitch. Rules:
- Only REAL companies you found evidence of on the web. Never invent names.
- Every company needs a real website domain you actually saw in the results.
- Skip household-name mega-corps unless the niche explicitly calls for them —
  the target is businesses reachable by an independent contractor.
- Skip companies that are themselves AI-automation consultancies (competitors).
- Prefer companies with some sign of the manual back-office work the ICP targets.
If you cannot find enough real companies, return fewer. Never pad the list."""

DISCOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "website": {"type": "string", "description": "Domain or URL."},
                    "hint": {
                        "type": "string",
                        "description": "One line of context to seed deeper research.",
                    },
                    "why_candidate": {
                        "type": "string",
                        "description": "Why this plausibly fits the ICP, per the search.",
                    },
                },
                "required": ["company", "website", "hint", "why_candidate"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def discover_candidates(client: anthropic.Anthropic, niche: str, icp: str,
                        count: int = 10) -> list:
    """Find real companies matching a niche + ICP. Returns candidate dicts.

    Does NOT qualify them — that's run_prospect()'s job. This is a cheap wide net;
    qualification is the expensive deep pass.
    """
    search = _search(
        client,
        DISCOVERY_SYSTEM,
        f"Find about {count} real companies matching this niche: {niche}.\n\n"
        f"They should plausibly fit this ideal customer profile:\n{icp}\n\n"
        "Search the web. For each company list its name, website, what it does, "
        "and why it might fit. Only companies you actually found.",
        max_tokens=6000,
    )
    search_text = search["text"]

    result = _structure(
        client,
        "You extract structured company lists from research notes. Include only "
        "companies explicitly named in the notes with a real website. Never invent "
        "or pad entries.",
        f"Extract up to {count} companies from these research notes.\n\n"
        f"=== NOTES ===\n{search_text}",
        DISCOVERY_SCHEMA,
        max_tokens=6000,
    )
    return _dedupe(result.get("candidates", []))


def _dedupe(candidates: list) -> list:
    """Drop repeats by normalized company name."""
    seen, out = set(), []
    for c in candidates:
        key = (c.get("company") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


# --- Stage 0b: niche suggestion ----------------------------------------------

NICHE_SCHEMA = {
    "type": "object",
    "properties": {
        "niches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "niche": {
                        "type": "string",
                        "description": "Ready-to-search niche phrase including the "
                                       "location, e.g. 'independent recruiting "
                                       "agencies in Barcelona'.",
                    },
                    "why": {
                        "type": "string",
                        "description": "Why this segment fits the ICP — the "
                                       "automatable back-office pain and any buying signals.",
                    },
                    "local_angle": {
                        "type": "string",
                        "description": "What makes this niche notable in this "
                                       "specific city or region.",
                    },
                },
                "required": ["niche", "why", "local_angle"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["niches"],
    "additionalProperties": False,
}


def _niche_system(icp: str) -> str:
    return f"""You suggest promising B2B niches to prospect in a given city or \
region. A niche is a concrete, searchable *segment* of businesses — an industry + \
business-type — not a single company.

Good niches are full of small-to-mid businesses with visible, automatable \
back-office work, and fit the consultant's ICP below. For each niche:
- Make `niche` a ready-to-search phrase that INCLUDES the location, so it can be
  fed straight into company discovery (e.g. "boutique law firms in Valencia").
- Grade honestly. Prefer segments with repetitive manual workflows and real budget
  over glamorous-but-poor-fit ones.
- Ground it in what the city is actually known for economically where you can.
- Avoid sectors dominated by mega-corps or that are mostly large enterprises, and
  avoid niches that are themselves AI-automation consultancies (competitors).

Here is the ICP:
{icp}"""


def suggest_niches(client: anthropic.Anthropic, location: str, icp: str,
                   count: int = 8) -> list:
    """Propose searchable B2B niches for a location that fit the ICP.

    This is a reasoning pass with NO web search: niche *ideation* draws on the
    model's knowledge of a city's economy, unlike discovery which must find real,
    verifiable companies on the web. Cheap and fast (one call) — a starting point.
    Returns a list of {niche, why, local_angle}; each `niche` feeds
    discover_candidates() directly.
    """
    result = _structure(
        client,
        _niche_system(icp),
        f"Suggest about {count} B2B niches worth prospecting in: {location}.\n"
        "Each niche must be concrete, searchable, and include the location.",
        NICHE_SCHEMA,
        max_tokens=4000,
    )
    return result.get("niches", [])


# --- Stage 1: research -------------------------------------------------------

RESEARCH_SYSTEM = """You are a sharp B2B prospect researcher. Given a company, \
use web search to find concrete, current facts. Prioritize:
- What the company does and who it sells to
- Rough size (employees / revenue band) and industry
- Signals of manual/repetitive back-office work (ops, support, data entry, research)
- Technical maturity (SaaS tools, CRM, APIs, engineering presence)
- Growth and hiring signals (recent hires, "we're scaling", funding, job posts)
- Anything suggesting budget to hire an outside contractor

Be factual and concise. Never invent facts — if something isn't found, say so. \
Prefer recent sources. End with a short bulleted evidence list."""


def research_company(client: anthropic.Anthropic, company: str, hint: str = "") -> dict:
    """Research one company. Returns {'text': summary, 'sources': [{title,url}]}."""
    ask = f"Research this company as a potential client: {company}."
    if hint:
        ask += f" Extra context: {hint}."
    ask += " Search the web and summarize what you find."
    return _search(client, RESEARCH_SYSTEM, ask)


def _extract_sources(resp) -> list:
    """Pull deduplicated {title, url} from web_search_tool_result blocks."""
    seen, sources = set(), []
    for block in resp.content:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        content = getattr(block, "content", None)
        if not isinstance(content, list):  # error object, not a result list
            continue
        for r in content:
            url = getattr(r, "url", None)
            if url and url not in seen:
                seen.add(url)
                sources.append({"title": getattr(r, "title", "") or url, "url": url})
    return sources


# --- Stage 2: qualify --------------------------------------------------------

QUALIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "fit_score": {"type": "integer", "description": "0-100; higher = better fit"},
        "tier": {"type": "string", "enum": ["A", "B", "C", "disqualified"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "one_line": {"type": "string", "description": "One-sentence verdict."},
        "pain_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pain": {"type": "string"},
                    "evidence": {"type": "string"},
                    "agent_solution": {
                        "type": "string",
                        "description": "How an AI agent could remove this pain.",
                    },
                },
                "required": ["pain", "evidence", "agent_solution"],
                "additionalProperties": False,
            },
        },
        "buying_signals": {"type": "array", "items": {"type": "string"}},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "outreach_angle": {
            "type": "string",
            "description": "A specific opener to start a conversation with this company.",
        },
    },
    "required": [
        "company", "fit_score", "tier", "confidence", "one_line",
        "pain_points", "buying_signals", "red_flags", "outreach_angle",
    ],
    "additionalProperties": False,
}


def _qualify_system(icp: str) -> str:
    return f"""You qualify B2B prospects against a specific consultant's Ideal \
Customer Profile. Score honestly — a low score on a poor fit is more useful than \
false optimism.

Scoring guide for fit_score (0-100):
- 80-100 (tier A): strong fit, clear automatable pain, buying signals present
- 60-79  (tier B): decent fit, some pain or some signals, worth a look
- 40-59  (tier C): weak fit, unclear pain or no signals
- 0-39   (disqualified): poor fit or a red flag (competitor, too big, no budget)

Base everything ONLY on the research provided. If the research is thin, lower \
confidence rather than inventing pain points. Ground every pain point and signal \
in something the research actually says.

Here is the ICP:
{icp}"""


def qualify_company(client: anthropic.Anthropic, company: str, research_text: str,
                    icp: str) -> dict:
    """Score researched company against the ICP. Returns the parsed JSON verdict."""
    return _structure(
        client,
        _qualify_system(icp),
        f"Company: {company}\n\n=== RESEARCH ===\n{research_text}\n\n"
        "Qualify this prospect against the ICP.",
        QUALIFY_SCHEMA,
    )


def run_prospect(client: anthropic.Anthropic, company: str, icp: str,
                 hint: str = "") -> dict:
    """Full pipeline for one company: research -> qualify -> combined record."""
    research = research_company(client, company, hint)
    verdict = qualify_company(client, company, research["text"], icp)
    verdict["research_summary"] = research["text"]
    verdict["sources"] = research["sources"]
    return verdict
