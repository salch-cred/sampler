"""The plugin as Hermes sees it: middleware, the /sampler command and the settings."""

import pytest
import sampler


class FakeCtx:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self.middleware = {}
        self.commands = {}

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_middleware(self, kind, callback):
        self.middleware.setdefault(kind, []).append(callback)

    def register_command(self, name, handler, **_):
        self.commands[name] = handler


REQUEST = {"model": "glm-test", "messages": [{"role": "user", "content": "hi"}]}


def call(ctx, **context):
    base = dict(request=dict(REQUEST), original_request=dict(REQUEST), session_id="S1",
                platform="cli", model="glm-test", provider="nvidia", base_url="https://x",
                api_mode="chat_completions", _payload_key="request",
                middleware_schema_version="hermes.middleware.v1")
    base.update(context)
    return ctx.middleware["llm_request"][0](**base)


@pytest.fixture
def ctx():
    fake = FakeCtx({"temperature": "0.3", "rules": "platform:telegram=0.8, model:*coder*=0.1"})
    sampler.register(fake)
    return fake


def test_registers_middleware_and_command(ctx):
    assert "llm_request" in ctx.middleware and "sampler" in ctx.commands


def test_default_applies(ctx):
    out = call(ctx)
    assert out["request"]["temperature"] == 0.3
    assert out["source"] == "sampler" and "default" in out["reason"]


def test_platform_rule(ctx):
    assert call(ctx, platform="telegram")["request"]["temperature"] == 0.8


def test_model_rule(ctx):
    assert call(ctx, model="qwen3-coder")["request"]["temperature"] == 0.1


def test_session_override_and_reset(ctx):
    cmd = ctx.commands["sampler"]
    assert "0.05" in cmd("0.05", session_id="S1")
    assert call(ctx)["request"]["temperature"] == 0.05
    assert call(ctx, session_id="S2")["request"]["temperature"] == 0.3      # other sessions untouched
    cmd("reset", session_id="S1")
    assert call(ctx)["request"]["temperature"] == 0.3


def test_session_top_p(ctx):
    ctx.commands["sampler"]("top_p 0.9", session_id="S1")
    out = call(ctx)
    assert out["request"]["top_p"] == 0.9 and out["request"]["temperature"] == 0.3


def test_invalid_command_values(ctx):
    cmd = ctx.commands["sampler"]
    assert "Usage" in cmd("hot", session_id="S1")
    assert "Usage" in cmd("5", session_id="S1")
    assert "between 0 and 1" in cmd("top_p 3", session_id="S1")


def test_unsupported_api_mode_passes_through(ctx):
    assert call(ctx, api_mode="codex_responses") is None


def test_nothing_configured_passes_through():
    fake = FakeCtx({})
    sampler.register(fake)
    assert call(fake) is None


def test_status_shows_settings_counts_and_bad_rules():
    fake = FakeCtx({"temperature": "0.3", "rules": "platform:cli=0.2, oops"})
    sampler.register(fake)
    call(fake)
    call(fake, api_mode="codex_responses")
    out = fake.commands["sampler"]("status", session_id="S1")
    assert "default temperature: 0.3" in out and "platform:cli=0.2" in out
    assert "changed: 1" in out and "left as they were: 1" in out and "oops" in out


def test_non_dict_request_is_ignored(ctx):
    assert ctx.middleware["llm_request"][0](request=None, api_mode="chat_completions") is None
