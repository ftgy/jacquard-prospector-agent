"""Pure helpers in agent.py: JSON extraction and candidate dedup (no network)."""

import pytest

from prospector.agent import _dedupe, _extract_json


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
