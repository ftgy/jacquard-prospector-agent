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


def _fake_deepseek(monkeypatch, proxy_fails: bool, personal_key: str | None):
    """Wire _structure_deepseek to fake clients; returns the models called, in order."""
    calls = []
    proxy, personal = object(), object()
    monkeypatch.setattr(agent, "get_deepseek_proxy_model", lambda: "deepseek/deepseek-v4-pro")
    monkeypatch.setattr(agent, "get_deepseek_model", lambda: "deepseek-chat")
    monkeypatch.setattr(agent, "make_deepseek_proxy_client", lambda: proxy)
    monkeypatch.setattr(agent, "make_deepseek_client", lambda: personal)
    if personal_key:
        monkeypatch.setenv("DEEPSEEK_API_KEY", personal_key)
    else:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def fake_structure_openai(client, system, ask, schema, max_tokens=4000, model=""):
        calls.append(model)
        if client is proxy and proxy_fails:
            raise agent.openai.APIConnectionError(request=None)
        return {"subject": model}

    monkeypatch.setattr(agent, "_structure_openai", fake_structure_openai)
    return calls


def test_deepseek_uses_proxy_first(monkeypatch):
    calls = _fake_deepseek(monkeypatch, proxy_fails=False, personal_key="sk-x")
    out = agent._structure_deepseek("sys", "ask", {})
    assert out == {"subject": "deepseek/deepseek-v4-pro"}
    assert calls == ["deepseek/deepseek-v4-pro"]


def test_deepseek_falls_back_to_personal_key(monkeypatch):
    calls = _fake_deepseek(monkeypatch, proxy_fails=True, personal_key="sk-x")
    out = agent._structure_deepseek("sys", "ask", {})
    assert out == {"subject": "deepseek-chat"}
    assert calls == ["deepseek/deepseek-v4-pro", "deepseek-chat"]


def test_deepseek_proxy_error_raised_without_personal_key(monkeypatch):
    _fake_deepseek(monkeypatch, proxy_fails=True, personal_key=None)
    with pytest.raises(agent.openai.APIConnectionError):
        agent._structure_deepseek("sys", "ask", {})
