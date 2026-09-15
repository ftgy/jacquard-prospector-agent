"""
Deterministic checks on a drafted outreach email — the playbook rules code can
verify exactly, with no model call.

The drafting model (DeepSeek in particular) sometimes paraphrases a fixed line,
slips into "usted", or runs long. Those are mechanical, so they're caught here for
free and reliably; the judgment rules (guessing vs. diagnosing, invented facts,
naming people) are left to the Claude review pass (agent.review_outreach_email),
which receives these findings as a head start. The same checks run again on the
reviewed text, and an empty result there is the gate for sending unattended.

lint_email() returns a list of {'rule', 'detail'} dicts — empty means clean.
"""

import re

from .agent import INVITE_CORRECTION, SELF_INTRO, SENDER_NAME, SIGNATURE_LINKS

# The fixed opener of the ask (playbook beat 5). Spanish only — the playbook
# defines no English equivalent.
ASK_OPENER = {"spanish": "¿Os cuadra agendar una llamada de 20 minutos para"}

# Playbook: "Aim for 150–180 words — a ceiling, not a target." Counted on the body
# without the signature block.
MAX_WORDS = 180
# Playbook asks for ~20 words per sentence; flag only the clearly runaway ones.
MAX_SENTENCE_WORDS = 35

_BANNED = {
    "spanish": [
        (r"\busted(es)?\b", "Formal “usted” — address the team as vosotros."),
        (r"pequeñ[oa]\s+agente|agente\s+pequeñ[oa]", "Calls the agent “pequeño”."),
        (r"tendría sentido", "“¿Tendría sentido…” — use the fixed ask opener."),
        (r"espero que (este|el) (correo|email|mensaje)", "Filler opener."),
        (r"espero que (estéis|os encontréis) bien", "Filler opener."),
        (r"\bagéntic[oa]s?\b", "AI mysticism (“agéntico”)."),
        (r"\d\s*€|€\s*\d", "Mentions a price."),
        (r"[¡!]", "Exclamation mark."),
    ],
    "english": [
        (r"small agent|little agent", "Diminutive for the agent."),
        (r"quick question|circling back|reaching out|finds you well",
         "Banned filler phrase."),
        (r"\bagentic\b", "AI mysticism (“agentic”)."),
        (r"[$€]\s*\d", "Mentions a price."),
        (r"!", "Exclamation mark."),
    ],
}
# Deliberately NOT here: words like "presupuesto"/"precio" (usually the prospect's
# own business — quotes, listing prices) and "tú" (often inside a quote from their
# website). Those false positives would block sending; the Claude review judges them.

_URL = re.compile(r"https?://|www\.|\b[a-z0-9-]+\.(dev|com|es|io)\b", re.I)


def _squash(text: str) -> str:
    """Collapse all whitespace so a fixed line survives being re-wrapped."""
    return " ".join((text or "").split())


def _split_signature(body: str) -> tuple[str, str]:
    """(prose, signature). The signature starts at the sender's name line; if it's
    missing, the whole body counts as prose."""
    idx = body.rfind(SENDER_NAME)
    return (body[:idx], body[idx:]) if idx != -1 else (body, "")


def lint_email(subject: str, body: str, language: str = "spanish") -> list[dict]:
    """Check one draft against the mechanical playbook rules."""
    issues: list[dict] = []

    def flag(rule: str, detail: str) -> None:
        issues.append({"rule": rule, "detail": detail})

    flat = _squash(body)
    for rule, fixed in (("self-intro", SELF_INTRO.get(language)),
                        ("invite-correction", INVITE_CORRECTION.get(language)),
                        ("ask-opener", ASK_OPENER.get(language))):
        if fixed and _squash(fixed) not in flat:
            flag(rule, f"Missing the fixed line, verbatim: “{fixed}”")

    prose, signature = _split_signature(body)
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    if lines[-2:] != [SENDER_NAME, SIGNATURE_LINKS]:
        flag("signature", f"Must end with “{SENDER_NAME}” then “{SIGNATURE_LINKS}” "
                          "on the line below.")

    lowered = f"{subject}\n{prose}"
    for pattern, detail in _BANNED.get(language, []):
        m = re.search(pattern, lowered, re.I)
        if m:
            flag("banned", f"{detail} (“{m.group(0)}”)")

    if _URL.search(prose):
        flag("url-in-body", "Links belong only on the signature line.")

    words = len(prose.split())
    if words > MAX_WORDS:
        flag("length", f"{words} words before the signature (ceiling {MAX_WORDS}).")

    for sentence in re.split(r"(?<=[.?!])\s+", _squash(prose)):
        n = len(sentence.split())
        if n > MAX_SENTENCE_WORDS:
            flag("long-sentence", f"{n} words: “{sentence[:80]}…”")

    if re.search(r"\b(AI|IA)\b", subject):
        flag("subject", "Subject mentions AI/IA.")
    if re.match(r"\s*automatiza", subject, re.I):
        flag("subject", "Subject is a bare “Automatizar…” task label.")

    return issues
