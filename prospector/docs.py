"""
Load the outreach voice + rules for the prospect-email prompt from local prompt
fragments, so the prompt is defined in one place and versioned alongside the code
that uses it (it used to be read from a sibling `website` repo, which meant the
agent couldn't draft an email unless that repo happened to be checked out beside
this one).

The text lives as plain Markdown under prospector/prompts/. Editing those files
changes the agent's outreach voice directly — there is no second copy.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def prompts_dir() -> Path:
    """Directory holding the outreach prompt fragments."""
    return _PROMPTS_DIR


def load_prompt(name: str) -> str:
    """Return the text of prospector/prompts/NAME.md, whitespace-stripped.

    Fails loudly on a missing file: a silently-empty voice/ruleset would let the
    agent draft an email with no guidance, which is exactly what we don't want.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Outreach prompt fragment not found: {path}\n"
            f"Expected it under {_PROMPTS_DIR}."
        )
    return path.read_text(encoding="utf-8").strip()
