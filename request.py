"""Putting sampling settings into a model request, only where the model accepts them."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Optional

from .rules import Setting

# Where each API mode keeps its sampling settings.
FLAT = {"chat_completions", "anthropic_messages"}
BEDROCK = {"bedrock_converse"}

# Used only when Hermes' own check is unavailable: families that reject a temperature.
_FALLBACK_OMIT = re.compile(
    r"kimi|moonshot|(^|[/:])o[134](-|$)|(^|[/:])gpt-5|gemini-3\.[6-9]|gemini-[4-9]", re.I)


def hermes_omits_temperature(model: str, base_url: str = "", provider: str = "") -> bool:
    """True when Hermes itself leaves temperature out for this model or route.

    Hermes keeps that list (reasoning models, Kimi, newer Gemini, routes that already rejected a
    temperature), so Sampler asks it instead of guessing. The fallback list covers older builds.
    """
    try:
        from agent.auxiliary_client import OMIT_TEMPERATURE, _fixed_temperature_for_model
    except Exception:
        return bool(_FALLBACK_OMIT.search(model or ""))
    try:
        return _fixed_temperature_for_model(model, base_url, provider) is OMIT_TEMPERATURE
    except Exception:
        return bool(_FALLBACK_OMIT.search(model or ""))


def apply(request: dict, setting: Setting, *, api_mode: str, model: str, base_url: str = "",
          provider: str = "", omits=hermes_omits_temperature) -> tuple[Optional[dict], str]:
    """Return (changed request, reason) or (None, why it was left alone)."""
    if setting.empty:
        return None, "nothing configured"
    if api_mode not in FLAT | BEDROCK:
        return None, f"api mode {api_mode or 'unknown'} is left to the provider default"
    if omits(model, base_url, provider):
        return None, f"Hermes leaves temperature out for {model}"
    changed = deepcopy(request)
    parts = []
    if api_mode in FLAT:
        if setting.temperature is not None:
            changed["temperature"] = setting.temperature
            parts.append(f"temperature {setting.temperature:g}")
        # Newer Anthropic models refuse temperature and top_p together; temperature wins.
        if setting.top_p is not None and not (api_mode == "anthropic_messages"
                                             and setting.temperature is not None):
            changed["top_p"] = setting.top_p
            parts.append(f"top_p {setting.top_p:g}")
    else:
        config = dict(changed.get("inferenceConfig") or {})
        if setting.temperature is not None:
            config["temperature"] = setting.temperature
            parts.append(f"temperature {setting.temperature:g}")
        if setting.top_p is not None:
            config["topP"] = setting.top_p
            parts.append(f"top_p {setting.top_p:g}")
        changed["inferenceConfig"] = config
    if not parts:
        return None, "nothing to set"
    return changed, ", ".join(parts) + (f" ({setting.source})" if setting.source else "")
