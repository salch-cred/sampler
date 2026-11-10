"""Putting settings into requests: each API mode, and the models that must be left alone."""

import sys
import types

import pytest
from sampler import request as req
from sampler.rules import Setting

BASE = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
never = lambda *a: False


def test_chat_completions_gets_temperature_and_top_p():
    changed, reason = req.apply(BASE, Setting(0.2, 0.9, "default"), api_mode="chat_completions",
                                model="m", omits=never)
    assert changed["temperature"] == 0.2 and changed["top_p"] == 0.9
    assert reason == "temperature 0.2, top_p 0.9 (default)"


def test_original_request_is_not_mutated():
    original = {"model": "m", "messages": []}
    req.apply(original, Setting(0.2), api_mode="chat_completions", model="m", omits=never)
    assert "temperature" not in original


def test_anthropic_gets_temperature_but_not_both():
    changed, _ = req.apply(BASE, Setting(0.2, 0.9), api_mode="anthropic_messages", model="claude",
                           omits=never)
    assert changed["temperature"] == 0.2 and "top_p" not in changed


def test_anthropic_top_p_alone_is_allowed():
    changed, _ = req.apply(BASE, Setting(None, 0.9), api_mode="anthropic_messages", model="claude",
                           omits=never)
    assert changed["top_p"] == 0.9 and "temperature" not in changed


def test_bedrock_uses_inference_config():
    request = dict(BASE, inferenceConfig={"maxTokens": 100})
    changed, _ = req.apply(request, Setting(0.4, 0.8), api_mode="bedrock_converse", model="m",
                           omits=never)
    assert changed["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.4, "topP": 0.8}


@pytest.mark.parametrize("mode", ["codex_responses", "codex_app_server", "", "something_new"])
def test_other_api_modes_are_left_alone(mode):
    changed, reason = req.apply(BASE, Setting(0.2), api_mode=mode, model="m", omits=never)
    assert changed is None and "provider default" in reason


def test_models_hermes_omits_are_never_changed():
    changed, reason = req.apply(BASE, Setting(0.2), api_mode="chat_completions", model="o3",
                                omits=lambda *a: True)
    assert changed is None and "leaves temperature out" in reason


def test_nothing_configured_changes_nothing():
    assert req.apply(BASE, Setting(), api_mode="chat_completions", model="m", omits=never)[0] is None


def test_uses_hermes_own_check_when_available(monkeypatch):
    sentinel = object()
    fake = types.ModuleType("agent.auxiliary_client")
    fake.OMIT_TEMPERATURE = sentinel
    fake._fixed_temperature_for_model = lambda model, base_url=None, provider=None: (
        sentinel if model == "special" else None)
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake)
    assert req.hermes_omits_temperature("special") is True
    assert req.hermes_omits_temperature("o3-mini") is False       # Hermes' answer wins


@pytest.mark.parametrize("model,omitted", [
    ("o3-mini", True), ("openai/o4-mini", True), ("gpt-5.5", True), ("moonshotai/kimi-k3", True),
    ("gemini-3.6-pro", True), ("z-ai/glm-5.3-flash", False), ("claude-sonnet-5-5", False),
    ("deepseek/deepseek-v4.1-flash", False),
])
def test_fallback_list_when_hermes_check_is_unavailable(monkeypatch, model, omitted):
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", None)    # import now fails
    assert req.hermes_omits_temperature(model) is omitted


def test_a_failing_hermes_check_falls_back(monkeypatch):
    fake = types.ModuleType("agent.auxiliary_client")
    fake.OMIT_TEMPERATURE = object()

    def broken(*a, **k):
        raise RuntimeError("boom")
    fake._fixed_temperature_for_model = broken
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake)
    assert req.hermes_omits_temperature("gpt-5.5") is True
    assert req.hermes_omits_temperature("glm-5") is False
