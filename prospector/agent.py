"""
Core agent: find companies, research them, and qualify them against your ICP.

Discovery (optional, given a niche):
  0. discover_candidates() -> real companies matching a niche + your ICP

Then per prospect:
  1. research_company()  -> web research + sources (uses the web_search server tool)
  2. qualify_company()   -> structured scoring against your ICP (structured output)

Every stage uses adaptive thinking and the model from config.get_model(), except
the outreach-email draft, which uses config.get_email_model() so that creative
Spanish copywriting can run on its own model. Stages that need the web search
first, then a second call structures the result — mixing the web_search server
tool with structured output in one call is unreliable.
"""

import json
import logging
import os
import re

import anthropic
import openai

from .config import (
    get_deepseek_model,
    get_deepseek_proxy_model,
    get_email_model,
    get_email_provider,
    get_model,
    get_output_language,
    get_review_model,
    get_web_search_tool,
    make_deepseek_client,
    make_deepseek_proxy_client,
    output_language_name,
    use_native_structured_output,
)
from .docs import load_prompt
from .search import get_search_backend, grounded_search

log = logging.getLogger(__name__)


def _output_language_note() -> str:
    """Instruction appended to research/qualify/discovery prompts to set the
    language of the generated output. Empty for English (the model's default), so
    English runs are byte-for-byte unchanged; set via config.OUTPUT_LANGUAGE."""
    lang = output_language_name()
    if lang == "English":
        return ""
    return (f"\n\nIMPORTANT: Write ALL of your output — every summary, verdict, "
            f"pain point, buying signal, and outreach angle — in {lang}, using "
            f"natural, idiomatic business {lang}. Keep company names, URLs, and "
            f"other proper nouns exactly as they appear; translate everything else.")


# Research depth. The default ("normal") is the historical setting; "thorough" is
# opt-in per run from the Research-companies tab — more web searches and a longer
# summary buy deeper coverage at higher cost. See research_company().
DEFAULT_SEARCH_USES = 6
DEFAULT_SEARCH_TOKENS = 4000
THOROUGH_SEARCH_USES = 12
THOROUGH_SEARCH_TOKENS = 8000


def _web_search_tool(max_uses: int = DEFAULT_SEARCH_USES) -> dict:
    return {"type": get_web_search_tool(), "name": "web_search", "max_uses": max_uses}


def _search(client: anthropic.Anthropic, system: str, ask: str,
            max_tokens: int = DEFAULT_SEARCH_TOKENS,
            max_uses: int = DEFAULT_SEARCH_USES) -> dict:
    """Answer `ask` using live web search. Returns {'text', 'sources'}.

    Two backends (see search.py): a separate grounded model (Gemini), or Claude's
    own web_search server tool where that's actually available.
    """
    if get_search_backend() == "gemini":
        return grounded_search(system, ask, max_tokens)
    return _anthropic_search(client, system, ask, max_tokens, max_uses)


def _anthropic_search(client: anthropic.Anthropic, system: str, ask: str,
                      max_tokens: int, max_uses: int = DEFAULT_SEARCH_USES) -> dict:
    """Claude + the web_search server tool, resuming if the search loop pauses."""
    messages = [{"role": "user", "content": ask}]
    for _ in range(max_uses + 2):  # a few extra passes to absorb pause_turn resumes
        resp = client.messages.create(
            model=get_model(),
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            tools=[_web_search_tool(max_uses)],
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
               max_tokens: int = 4000, model: str | None = None) -> dict:
    """One structured turn. Returns parsed JSON.

    `model` overrides the pipeline model for this one call (see the email stage,
    which can run on its own model); None uses config.get_model().

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
            model=model or get_model(),
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


def _structure_openai(client, system: str, ask: str, schema: dict,
                      max_tokens: int = 4000, model: str = "") -> dict:
    """One structured turn against an OpenAI-compatible model (DeepSeek).

    The Anthropic path (_structure) can't be reused: DeepSeek speaks
    chat.completions with a `system`/`user` message list and JSON mode, not
    /v1/messages with output_config or the `thinking` param. So this mirrors the
    *prompted-JSON* branch of _structure — schema pasted into the prompt, parsed
    defensively with one retry — over the OpenAI SDK. JSON mode also requires the
    word "JSON" in the prompt, which the schema instruction already provides.
    """
    system += (
        "\n\nReply with ONE JSON object and nothing else — no prose, no markdown "
        "fences. It must match this JSON Schema exactly:\n" + json.dumps(schema, indent=2)
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": ask}]
    for attempt in range(2):
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
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


def _structure_deepseek(system: str, ask: str, schema: dict,
                        max_tokens: int = 4000) -> dict:
    """One structured DeepSeek turn: via the LiteLLM proxy, personal key as backup.

    The proxy model (config.get_deepseek_proxy_model) goes first so drafting runs
    on the shared gateway; any API error there — proxy down, model benched, out
    of budget — retries once on the personal DEEPSEEK_API_KEY. Without a personal
    key the proxy error is raised as-is.
    """
    proxy_model = get_deepseek_proxy_model()
    if proxy_model:
        try:
            return _structure_openai(make_deepseek_proxy_client(), system, ask,
                                     schema, max_tokens=max_tokens, model=proxy_model)
        except openai.APIError as e:
            if not os.environ.get("DEEPSEEK_API_KEY"):
                raise
            log.warning("DeepSeek via proxy (%s) failed, using personal key: %s",
                        proxy_model, e)
    return _structure_openai(make_deepseek_client(), system, ask, schema,
                             max_tokens=max_tokens, model=get_deepseek_model())


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
                        count: int = 10, exclude: list[str] | None = None) -> list:
    """Find real companies matching a niche + ICP. Returns candidate dicts.

    Does NOT qualify them — that's run_prospect()'s job. This is a cheap wide net;
    qualification is the expensive deep pass. Pass `exclude` (company names we
    already have) to steer the search toward DIFFERENT companies — this is what
    lets a repeated search surface fresh results instead of the same names.
    """
    ask = (
        f"Find about {count} real companies matching this niche: {niche}.\n\n"
        f"They should plausibly fit this ideal customer profile:\n{icp}\n\n"
        "Search the web. For each company list its name, website, what it does, "
        "and why it might fit. Only companies you actually found."
    )
    if exclude:
        ask += ("\n\nWe ALREADY have the companies below — do not return any of "
                "them. Find DIFFERENT ones:\n"
                + "\n".join(f"- {name}" for name in exclude))
    search = _search(client, DISCOVERY_SYSTEM + _output_language_note(), ask,
                     max_tokens=6000)
    search_text = search["text"]

    result = _structure(
        client,
        "You extract structured company lists from research notes. Include only "
        "companies explicitly named in the notes with a real website. Never invent "
        "or pad entries." + _output_language_note(),
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
        _niche_system(icp) + _output_language_note(),
        f"Suggest about {count} B2B niches worth prospecting in: {location}.\n"
        "Each niche must be concrete, searchable, and include the location.",
        NICHE_SCHEMA,
        max_tokens=4000,
    )
    return result.get("niches", [])


# --- Niche categorization ----------------------------------------------------

CATEGORIZE_SYSTEM = """You file a prospecting search under a niche CATEGORY — the \
kind of business it targets, with the location stripped out. The goal is that \
searches for the same business type in different places land in one category: \
"real estate agencies in Barcelona" and "estate agents in Marbella" are both \
"Real estate agencies".

Rules:
- The category names the business type/industry only. Never include a city, \
region, or country.
- If one of the existing categories already fits (the same or clearly the same \
kind of business), return it VERBATIM — copy the exact string. Small wording or \
plural differences don't matter; reuse the existing one.
- Only invent a new category when none of the existing ones genuinely fit.
- Keep new names short, plural, Title Case, no location (2-4 words), e.g. \
"Boutique law firms", "Independent recruiting agencies", "Dental clinics"."""

CATEGORIZE_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}},
    "required": ["category"],
    "additionalProperties": False,
}


def categorize_niche(client: anthropic.Anthropic, query: str,
                     existing: list[str]) -> str:
    """Map a discovery query to a location-agnostic niche category name.

    Returns one of `existing` verbatim when the niche fits it, otherwise a fresh
    short name. A reasoning pass with no web search — cheap and fast, like
    suggest_niches. `existing` is the current category names, to bias reuse so
    the same business type across cities collapses into one category.
    """
    listing = ("\n".join(f"- {c}" for c in existing)
               if existing else "(none yet — this is the first category)")
    result = _structure(
        client,
        CATEGORIZE_SYSTEM,
        f"Existing categories:\n{listing}\n\n"
        f"Search to file: {query!r}\n\n"
        "Return the category it belongs to.",
        CATEGORIZE_SCHEMA,
        max_tokens=1500,
    )
    return (result.get("category") or "").strip()


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

# Appended to the research prompt for a "thorough" run: pushes the model to search
# harder and go deeper, to match the higher search/token budget it's given.
THOROUGH_RESEARCH_NOTE = """

This is a THOROUGH research pass — spend the larger search budget. Don't stop at \
the first summary: dig into the company's own site (about, careers, blog, product \
pages), recent news and funding, LinkedIn/Crunchbase-style profiles, job postings, \
and reviews. Corroborate key facts across more than one source. Go deeper on the \
signals of manual/repetitive work and on budget/hiring evidence — these drive \
qualification. Produce a fuller, well-organized summary while staying strictly \
factual."""


def research_company(client: anthropic.Anthropic, company: str, hint: str = "",
                     thorough: bool = False) -> dict:
    """Research one company. Returns {'text': summary, 'sources': [{title,url}]}.

    `thorough` opts into a deeper pass: more web searches and a longer summary
    (see the *_SEARCH_USES / *_SEARCH_TOKENS constants), plus a prompt that pushes
    the model to corroborate across sources. Off by default to keep normal runs
    fast and cheap.
    """
    ask = f"Research this company as a potential client: {company}."
    if hint:
        ask += f" Extra context: {hint}."
    ask += " Search the web and summarize what you find."
    system = RESEARCH_SYSTEM + (THOROUGH_RESEARCH_NOTE if thorough else "")
    return _search(
        client, system + _output_language_note(), ask,
        max_tokens=THOROUGH_SEARCH_TOKENS if thorough else DEFAULT_SEARCH_TOKENS,
        max_uses=THOROUGH_SEARCH_USES if thorough else DEFAULT_SEARCH_USES,
    )


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
        _qualify_system(icp) + _output_language_note(),
        f"Company: {company}\n\n=== RESEARCH ===\n{research_text}\n\n"
        "Qualify this prospect against the ICP.",
        QUALIFY_SCHEMA,
    )


def run_prospect(client: anthropic.Anthropic, company: str, icp: str,
                 hint: str = "", thorough: bool = False) -> dict:
    """Full pipeline for one company: research -> qualify -> combined record.

    `thorough` deepens the research stage only (see research_company); qualifying
    always scores against whatever research produced.
    """
    research = research_company(client, company, hint, thorough=thorough)
    verdict = qualify_company(client, company, research["text"], icp)
    verdict["research_summary"] = research["text"]
    verdict["sources"] = research["sources"]
    return verdict


# --- Stage 3: outreach email -------------------------------------------------

EMAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {
            "type": "string",
            "description": "A short, specific, result-oriented subject line — a "
                           "pointed question or a felt outcome tied to the one task "
                           "you identified (e.g. \"¿Sigue alguien asignando leads a "
                           "mano?\"), never a flat description of an internal task "
                           "and never \"AI\" or clickbait.",
        },
        "body": {
            "type": "string",
            "description": "Plain-text email body, greeting through sign-off. End "
                           "with the sender's name, then the website + LinkedIn on "
                           "the line below, as a plain signature.",
        },
    },
    "required": ["subject", "body"],
    "additionalProperties": False,
}


_LANGUAGES = {"english": "English", "spanish": "Spanish"}

# Who the outreach is signed by, and the website referenced as the "proof asset"
# (the prospect-research agent that likely found the reader is itself the portfolio).
SENDER_NAME = "Francisco Narduzzi"
WEBSITE = "feina.dev"
LINKEDIN = "linkedin.com/in/francisco-narduzzi"
# The signature's contact line: website and LinkedIn, joined with a middot.
SIGNATURE_LINKS = f"{WEBSITE} · {LINKEDIN}"

# The self-introduction (beat 3 of the playbook, opening the offer) is the one
# line with no per-prospect content — it says nothing about the reader — so it is
# fixed here rather than regenerated on every draft: tune the positioning in one
# place and every email introduces the sender the same way, with no wording
# drift. Injected verbatim into the prompt (like SIGNATURE_LINKS); the model
# writes the observation and the felt question first, then opens the offer
# paragraph with this line. Keyed by output language.
SELF_INTRO = {
    "spanish": (
        "Soy ingeniero de software y me dedico a automatizar justo ese tipo de "
        "tareas con agentes de IA."
    ),
    "english": (
        "I'm a software engineer and I automate exactly this kind of task with AI "
        "agents."
    ),
}

# The studio background and the email's shape + rules are NOT inlined here — they
# live as editable Markdown under prospector/prompts/, read at draft time so the
# outreach voice can be tuned without touching this module (see prospector/docs.py):
#   - studio brief  -> prospector/prompts/studio-brief.md
#   - email playbook -> prospector/prompts/email-playbook.md
#   - follow-up playbook -> prospector/prompts/followup-playbook.md


def about_feina() -> str:
    """Studio background the email draws AT MOST ONE line from (prompt fragment)."""
    return load_prompt("studio-brief")


def email_playbook() -> str:
    """The email's shape and hard rules (prompt fragment)."""
    return load_prompt("email-playbook")


def followup_playbook() -> str:
    """A follow-up's shape and hard rules (prompt fragment)."""
    return load_prompt("followup-playbook")


def _language_note(language: str) -> tuple[str, str]:
    """(language name, the Spanish register note) for the email briefs."""
    es_note = (' (in Spanish, address the reader as a team with the Spain '
               'second-person plural "vosotros"/"vuestro" and the -áis/-éis verb '
               'endings — the natural, peer-to-peer register; never the formal '
               '"usted", which reads like a bank, and never the singular "tú")'
               if language == "spanish" else "")
    return _LANGUAGES.get(language, "English"), es_note


def _email_system(icp: str, language: str) -> str:
    lang, es_note = _language_note(language)
    intro = SELF_INTRO.get(language, SELF_INTRO["english"])
    return f"""You write ONE cold outreach email on behalf of \
{SENDER_NAME}, who runs {WEBSITE} — a small Barcelona engineering studio. It is a \
personal note from one person to another, signed by {SENDER_NAME}: the sender's own \
observations of the reader are first-person singular ("I noticed…"), but the studio \
and its work are always "we" — the company voice, never a headcount claim. The goal \
is to earn a short reply, not to close a sale.

Write the ENTIRE email — subject and body — in {lang}, in natural, idiomatic, \
plain-spoken business {lang}{es_note}. The subject is short, specific, and \
result-oriented: it makes the reader feel the outcome, not read an internal task \
description. Prefer a pointed, concrete question that names the pain (e.g. "¿Sigue \
alguien asignando leads a mano?") or a felt result, grounded in the one task you \
identified. Never lead with a bare "Automatizar el…" task label, never "AI", never \
clickbait.

Voice: plain, specific, unhyped, and HUMBLE. Concrete over abstract — name the task \
and the hours it eats, not "cutting-edge AI". Say the honest thing, even when it \
costs the sale. You are an outsider guessing at how they work from the outside, so \
anything you did not directly observe is a hypothesis: hedge it ("I imagine…", \
"I'd guess…"), rather than asserting how their \
business runs as if you knew it better than they do. Open with a short greeting; if \
you don't know the reader's name, keep it neutral rather than inventing one. Then go \
straight into the personalized beats — no introduction up front. The email is about \
THEM — the studio gets one line at most. The offer paragraph (after the felt \
question) OPENS with this self-introduction, reproduced VERBATIM — do not \
paraphrase, translate, reorder, or add to it:

{intro}

{email_playbook()}

Background on the studio — draw AT MOST ONE line from this, and do NOT summarize it:
{about_feina()}

Who I look for and how I qualify a fit (my ICP — use it to judge which pain to lead \
with; never restate it to the reader):
{icp}

End with a short valediction ("Un saludo," in Spanish) on its own line, then the \
sender's name "{SENDER_NAME}" on the next line, then "{SIGNATURE_LINKS}" on the line \
right below it, verbatim, as a plain signature."""


def _followup_system(icp: str, language: str) -> str:
    lang, es_note = _language_note(language)
    return f"""You write ONE short follow-up email on behalf of {SENDER_NAME}, who \
runs {WEBSITE} — a small Barcelona engineering studio. He already sent this company \
a cold outreach email (shown below the prospect facts) and got no reply. The \
follow-up goes out as a reply in the same thread, so the reader sees the earlier \
email(s) right below it. The goal is still to earn a short reply, not to close a sale.

Write the body in {lang}, in natural, idiomatic, plain-spoken business \
{lang}{es_note}. The subject is fixed (the thread's "Re: …"), so write only the body.

{followup_playbook()}

Background on the studio — use it only if it helps the one new thing, and never \
summarize it:
{about_feina()}

Who I look for (my ICP — use it to judge which angle matters; never restate it):
{icp}

End with a short valediction ("Un saludo," in Spanish) on its own line, then the \
sender's name "{SENDER_NAME}" on the next line, then "{SIGNATURE_LINKS}" on the line \
right below it, verbatim, as a plain signature."""


def _sends_context(sends: list[dict]) -> str:
    """The emails already sent to a prospect, oldest first (sends come newest first)."""
    parts = []
    for i, sd in enumerate(reversed(sends), 1):
        parts.append(f"--- Email {i}, sent {(sd.get('sent_at') or '')[:10]} ---\n"
                     f"Subject: {sd.get('subject') or ''}\n\n{sd.get('body') or ''}")
    return "\n\n".join(parts)


def followup_subject(sends: list[dict]) -> str:
    """A follow-up's subject: "Re: " + the thread's first subject, so Gmail and the
    reader's client keep it in the same thread."""
    first = (sends[-1].get("subject") or "").strip() if sends else ""
    while first.lower().startswith("re:"):
        first = first[3:].strip()
    return f"Re: {first}" if first else "Re:"


def draft_followup_email(client: anthropic.Anthropic, record: dict, sends: list[dict],
                         icp: str, language: str | None = None,
                         current: str = "") -> dict:
    """Draft a follow-up to the emails already sent to a prospect (`sends`,
    newest first, as db.list_sends returns them). Same provider as
    draft_outreach_email; `current` is a follow-up body being replaced, passed so
    the model offers a different take. Returns {'subject', 'body'} — the subject
    is the thread's fixed "Re: …", only the body is written."""
    system = _followup_system(icp, language or get_output_language())
    n = len(sends)
    ask = (f"Write follow-up number {n} to this company (the earlier "
           f"{'email' if n == 1 else f'{n} emails'} below got no reply), grounded "
           "only in the facts below.")
    if current.strip():
        ask += " Give a clearly different take from the current follow-up (below)."
    ask += ("\n\n=== PROSPECT ===\n" + _email_context(record)
            + "\n\n=== ALREADY SENT (oldest first) ===\n" + _sends_context(sends))
    if current.strip():
        ask += "\n\n=== CURRENT FOLLOW-UP (replace it) ===\n" + current
    if get_email_provider() == "deepseek":
        out = _structure_deepseek(system, ask, BODY_SCHEMA, max_tokens=1500)
    else:
        out = _structure(client, system, ask, BODY_SCHEMA,
                         max_tokens=1500, model=get_email_model())
    return {"subject": followup_subject(sends), "body": out["body"].strip()}


def _email_context(record: dict) -> str:
    """Compact the qualified record into the facts the email should draw on."""
    lines = [f"Company: {record.get('company', '')}"]
    if record.get("one_line"):
        lines.append(f"Verdict: {record['one_line']}")
    if record.get("outreach_angle"):
        lines.append(f"Outreach angle: {record['outreach_angle']}")
    for p in (record.get("pain_points") or []):
        lines.append(f"- Pain: {p.get('pain','')} | Evidence: {p.get('evidence','')} "
                     f"| An agent could: {p.get('agent_solution','')}")
    if record.get("buying_signals"):
        lines.append("Buying signals: " + "; ".join(record["buying_signals"]))
    if record.get("research_summary"):
        lines.append(f"\nResearch notes:\n{record['research_summary']}")
    return "\n".join(lines)


def draft_outreach_email(client: anthropic.Anthropic, record: dict, icp: str,
                         language: str | None = None) -> dict:
    """Draft a cold outreach email from a qualified prospect record.

    A single reasoning pass over research we already have (no web search), like
    niche suggestion. `language` is 'english' or 'spanish'; None follows the
    global config.OUTPUT_LANGUAGE. Returns {'subject', 'body'}.

    Provider is EMAIL_PROVIDER: 'anthropic' uses the passed client; 'deepseek'
    builds its own OpenAI-shape client and ignores `client` (so the rest of the
    pipeline — including contact lookup — still runs on Anthropic).
    """
    system = _email_system(icp, language or get_output_language())
    ask = ("Write a cold outreach email to this company, grounded only in the facts "
           "below.\n\n=== PROSPECT ===\n" + _email_context(record))
    if get_email_provider() == "deepseek":
        return _structure_deepseek(system, ask, EMAIL_SCHEMA, max_tokens=2000)
    return _structure(
        client, system, ask, EMAIL_SCHEMA,
        max_tokens=2000, model=get_email_model(),
    )


SUBJECT_SCHEMA = {
    "type": "object",
    "properties": {"subject": EMAIL_SCHEMA["properties"]["subject"]},
    "required": ["subject"],
    "additionalProperties": False,
}


def draft_email_subject(client: anthropic.Anthropic, record: dict, body: str,
                        icp: str, language: str | None = None,
                        current: str = "") -> str:
    """Write a fresh subject line for an existing email body, leaving the body alone.

    Same brief and provider (EMAIL_PROVIDER) as draft_outreach_email, so the
    subject rules match; `current` is the subject being replaced, passed so the
    model offers a genuinely different option.
    """
    system = _email_system(icp, language or get_output_language())
    ask = ("The email body below is final. Write ONLY a new subject line for it, "
           "following the brief's subject rules, grounded in the same one task the "
           "body leads with.")
    if current.strip():
        ask += f" Give a clearly different option from the current subject: {current!r}."
    ask += ("\n\n=== PROSPECT ===\n" + _email_context(record)
            + "\n\n=== EMAIL BODY ===\n" + body)
    if get_email_provider() == "deepseek":
        out = _structure_deepseek(system, ask, SUBJECT_SCHEMA, max_tokens=300)
    else:
        out = _structure(client, system, ask, SUBJECT_SCHEMA,
                         max_tokens=300, model=get_email_model())
    return out["subject"].strip()


BODY_SCHEMA = {
    "type": "object",
    "properties": {"body": EMAIL_SCHEMA["properties"]["body"]},
    "required": ["body"],
    "additionalProperties": False,
}


def draft_email_body(client: anthropic.Anthropic, record: dict, subject: str,
                     icp: str, language: str | None = None,
                     current: str = "") -> str:
    """Write a fresh email body under an existing subject, leaving the subject alone.

    The counterpart of draft_email_subject: same brief and provider
    (EMAIL_PROVIDER); `current` is the body being replaced, passed so the model
    offers a genuinely different take rather than a light rewording.
    """
    system = _email_system(icp, language or get_output_language())
    ask = ("The subject line below is final. Write ONLY a new email body for it, "
           "following the brief, leading with the same one task the subject points at.")
    if current.strip():
        ask += " Give a clearly different take from the current body (below)."
    ask += ("\n\n=== PROSPECT ===\n" + _email_context(record)
            + "\n\n=== SUBJECT ===\n" + subject)
    if current.strip():
        ask += "\n\n=== CURRENT BODY (replace it) ===\n" + current
    if get_email_provider() == "deepseek":
        out = _structure_deepseek(system, ask, BODY_SCHEMA, max_tokens=2000)
    else:
        out = _structure(client, system, ask, BODY_SCHEMA,
                         max_tokens=2000, model=get_email_model())
    return out["body"].strip()


# --- Stage 3b: review the draft against the playbook -------------------------

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "description": "Every playbook rule the draft broke, one entry each. "
                           "Empty if the draft already follows every rule.",
            "items": {
                "type": "object",
                "properties": {
                    "rule": {"type": "string",
                             "description": "Short name of the rule broken."},
                    "detail": {"type": "string",
                               "description": "What was wrong and how you fixed it."},
                },
                "required": ["rule", "detail"],
                "additionalProperties": False,
            },
        },
        "subject": {"type": "string",
                    "description": "The corrected subject (unchanged if it was fine)."},
        "body": {"type": "string",
                 "description": "The corrected body (unchanged if it was fine)."},
    },
    "required": ["issues", "subject", "body"],
    "additionalProperties": False,
}

_REVIEW_EDITOR = """You are the editor who checks a cold outreach email before it \
is sent. Another model wrote it following the brief below. Your job is to find \
every place the draft breaks the brief and fix it with the SMALLEST possible edit.

- Keep the writer's voice, wording, and choices wherever they follow the rules. Do \
not polish, restyle, or rewrite sentences that are already fine — a draft that \
breaks nothing comes back byte-for-byte unchanged, with an empty issues list.
- Check it against the research: any fact not supported by the prospect notes is \
invented — remove it or soften it into a hedged guess.
- Never add new facts, names, or claims of your own.
- Fixed lines (self-introduction, ask opener, signature) \
must match the brief exactly; restore them verbatim if they drifted. The offer \
sentence after the self-introduction is NOT fixed: if it says concretely what the \
system would do for this company, keep it — never swap it for the generic "ese \
flujo concreto" wording. But hold it to the brief: cut any detail the research \
doesn't support (systems, tools, how they work) and trim it to one short flow.
- Automatic checks may have flagged problems already; fix each one (they are \
reliable), then look for what they can't catch: people named, inferences stated \
as fact, the agent-found-you story, generic flattery, more than one task, a lead-in \
before the ask, a flat or AI-sounding subject.

=== THE WRITER'S BRIEF ===
"""


def review_outreach_email(client: anthropic.Anthropic, record: dict, draft: dict,
                          icp: str, language: str,
                          lint_issues: list[dict] | None = None,
                          sends: list[dict] | None = None) -> dict:
    """Check a drafted email against the playbook and fix what it breaks.

    Always runs on Anthropic (config.get_review_model), whichever provider wrote
    the draft. `lint_issues` are the deterministic findings (email_lint) passed in
    as a head start. `sends` (the emails already sent) marks the draft as a
    follow-up: it's checked against the follow-up brief, with those emails as
    context, and its subject is kept as is. Returns {'issues', 'subject', 'body'};
    the edits are minimal, so a clean draft comes back unchanged.
    """
    if sends:
        system = (_REVIEW_EDITOR.replace("a cold outreach email", "a follow-up email")
                  + _followup_system(icp, language))
    else:
        system = _REVIEW_EDITOR + _email_system(icp, language)
    flagged = "\n".join(f"- [{i['rule']}] {i['detail']}" for i in lint_issues or [])
    ask = ("=== PROSPECT RESEARCH ===\n" + _email_context(record)
           + ("\n\n=== ALREADY SENT (oldest first) ===\n" + _sends_context(sends)
              if sends else "")
           + "\n\n=== AUTOMATIC CHECKS FLAGGED ===\n" + (flagged or "(nothing)")
           + "\n\n=== DRAFT ===\nSubject: " + draft.get("subject", "")
           + "\n\n" + draft.get("body", ""))
    # Generous ceiling: adaptive thinking on a long brief routinely spends ~3.5k
    # tokens before the answer, and running out mid-thought returns empty text.
    out = _structure(client, system, ask, REVIEW_SCHEMA,
                     max_tokens=16000, model=get_review_model())
    return {
        "issues": [i for i in out.get("issues") or [] if isinstance(i, dict)],
        "subject": (draft.get("subject", "") if sends
                    else out.get("subject") or draft.get("subject", "")),
        "body": out.get("body") or draft.get("body", ""),
    }


# --- Stage 4: find where to send it ------------------------------------------

CONTACT_SYSTEM = """You find the best public contact details for a specific \
company so a consultant can send it a cold outreach email.

- Search the company's OWN official website first — check its contact, about, or
  legal-notice page (in Spain, "contacto" / "aviso legal"). Reputable business
  directories are a fallback.
- Return only a real email address you actually saw on a page. NEVER guess one or
  build it from a pattern like info@theirdomain. If you can't find a real address,
  say so plainly and don't provide one.
- Prefer a general or commercial inbox (info@, contact@, contacto@, comercial@,
  hola@) over a named person's address.
- Also report the company's phone number and official website if you find them,
  and the exact URL where you saw the email address."""

CONTACT_SCHEMA = {
    "type": "object",
    "properties": {
        "email": {
            "type": "string",
            "description": "The contact email address, verbatim. Empty string if "
                           "no real address was found.",
        },
        "phone": {"type": "string", "description": "Phone number, or empty string."},
        "website": {"type": "string", "description": "Official website, or empty string."},
        "source_url": {
            "type": "string",
            "description": "URL where the email was found. Empty string if none.",
        },
    },
    "required": ["email", "phone", "website", "source_url"],
    "additionalProperties": False,
}


def _contact_context(record: dict) -> str:
    """A few facts to pin the search to the RIGHT company (name can be ambiguous)."""
    lines = [f"Company: {record.get('company', '')}"]
    if record.get("one_line"):
        lines.append(f"What they do: {record['one_line']}")
    summary = record.get("research_summary") or ""
    if summary:
        lines.append("Context from earlier research:\n" + summary[:800])
    return "\n".join(lines)


def find_contact(client: anthropic.Anthropic, record: dict) -> dict | None:
    """Find where to send outreach: search the web, then structure the result.

    Returns {'email', 'phone', 'website', 'source_url'} with '' for anything not
    found, or None if no real email address turned up.
    """
    context = _contact_context(record)
    search = _search(
        client,
        CONTACT_SYSTEM,
        "Find the direct business contact email for this company, plus its phone "
        "and website. Use its own official site's contact/legal pages first.\n\n"
        "=== COMPANY ===\n" + context,
        max_tokens=3000,
    )
    contact = _structure(
        client,
        "You extract contact details from research notes into JSON. Copy the email "
        "exactly as written in the notes; if the notes report no real email was "
        "found, return an empty string for it. Never invent an address.",
        "=== RESEARCH NOTES ===\n" + search["text"],
        CONTACT_SCHEMA,
        max_tokens=800,
    )
    contact = {k: (str(v).strip() if v else "") for k, v in contact.items()}
    return contact if contact.get("email") else None
