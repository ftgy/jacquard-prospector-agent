"""Pure helpers in agent.py: JSON extraction and candidate dedup (no network)."""

import pytest

from prospector import agent
from prospector.agent import _dedupe, _extract_json, suggest_niches


def test_extract_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_prose_around_it():
    text = 'Here is the result:\n{"tier": "A", "fit_score": 80}\nHope that helps!'
    assert _extract_json(text) == {"tier": "A", "fit_score": 80}


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError):
        _extract_json("there is no object here")


def test_dedupe_by_normalized_name():
    candidates = [
        {"company": "Acme"},
        {"company": "acme"},       # case-insensitive duplicate
        {"company": "  Acme  "},   # whitespace duplicate
        {"company": "Globex"},
    ]
    assert [c["company"] for c in _dedupe(candidates)] == ["Acme", "Globex"]


def test_dedupe_drops_blank_names():
    assert _dedupe([{"company": ""}, {"company": "  "}, {"company": "Real"}]) == \
        [{"company": "Real"}]


def test_suggest_niches_returns_list_and_passes_count(monkeypatch):
    seen = {}

    def fake_structure(client, system, ask, schema, max_tokens=4000):
        seen["ask"] = ask
        seen["schema"] = schema
        return {"niches": [
            {"niche": "boutique law firms in Valencia", "why": "manual doc work",
             "local_angle": "legal cluster"},
        ]}

    monkeypatch.setattr(agent, "_structure", fake_structure)
    niches = suggest_niches(object(), "Valencia", "my ICP", count=5)

    assert niches[0]["niche"] == "boutique law firms in Valencia"
    assert "Valencia" in seen["ask"] and "5" in seen["ask"]
    assert seen["schema"] is agent.NICHE_SCHEMA


def test_suggest_niches_missing_key_returns_empty(monkeypatch):
    monkeypatch.setattr(agent, "_structure",
                        lambda *a, **k: {})  # model returned no 'niches'
    assert suggest_niches(object(), "Nowhere", "icp") == []
