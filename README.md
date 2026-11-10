# sampler

**Set temperature and top_p for your Hermes agent: by default, per platform, model or provider, or per session.**

Hermes doesn't let you set the sampling temperature for your main agent. It uses the provider's default, or a fixed value for a few models. Yet the right temperature depends on the work: low for code and data, higher for writing and brainstorming. Sampler adds that control ([hermes-agent#17565](https://github.com/NousResearch/hermes-agent/issues/17565)).

## Use it

Set under `plugins.entries.sampler.settings` in `config.yaml`:

```yaml
plugins:
  entries:
    sampler:
      settings:
        temperature: "0.3"                                  # default for every request
        top_p: ""                                           # empty: the provider default
        rules: "platform:telegram=0.8, model:*coder*=0.1"   # first match wins
```

In a session:

```text
/sampler 0.2          temperature 0.2 for this session only
/sampler top_p 0.9    top_p 0.9 for this session only
/sampler reset        back to the configured settings
/sampler status       what applies, how many requests were changed, any rules it couldn't read
```

**Which setting wins:** the session override, then the first matching rule, then the default. Anything left unset stays at the provider's default.

**Rules** match by `platform` (cli, telegram, discord…), `model` or `provider`, with `*` wildcards, case-insensitively. A rule it can't read is ignored and listed by `/sampler status`.

## Safety

- **Models that must not get a temperature are never changed.** Hermes already leaves temperature out for reasoning models (OpenAI's o-series and GPT-5 family), Kimi, newer Gemini, and any route that has rejected it before. Sampler asks Hermes's own check (`agent.auxiliary_client._fixed_temperature_for_model`) before every change, so it follows the same list as Hermes. If that check is unavailable, Sampler uses a built-in list of the same model families.
- **Only request types it knows are touched:** OpenAI-compatible chat and Anthropic requests (top-level `temperature` and `top_p`) and Bedrock Converse (`inferenceConfig`). Codex/Responses requests, which go to reasoning models, and any other type are passed through unchanged.
- **Anthropic gets temperature or top_p, not both,** because newer Claude models refuse the pair. When both are set, temperature is sent.
- **Every change is labelled** in Hermes's middleware trace as `source: sampler`, with the reason, such as `temperature 0.2 (rule platform:cli)`.

## Limits

- **Main agent requests only.** Hermes's auxiliary calls, such as titling and compression, choose their own settings.
- **Session overrides live in memory.** They reset when Hermes restarts.
- **Hermes's own check is an internal function**, so the import is guarded. If a future Hermes renames it, Sampler falls back to its built-in list, which may lag behind Hermes's.

## What it touches

- **Changes model request arguments** (temperature, top_p) through Hermes's `llm_request` middleware, and nothing else.
- **No model calls, no network, no files written, no telemetry, no dependencies.**

## Development

```bash
pytest tests
```

48 tests cover rule parsing and precedence, each request type, the omit check and its fallback, the `/sampler` command, and the middleware wiring.

## License

MIT
