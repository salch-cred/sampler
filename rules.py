"""Which sampling settings apply to a model request.

Order of precedence: a per-session override set with /sampler, then the first matching rule from
config.yaml, then the configured default. Anything not set is left to the provider's own default.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Optional

TEMPERATURE_RANGE = (0.0, 2.0)
TOP_P_RANGE = (0.0, 1.0)
_RULE = re.compile(r"(platform|model|provider)\s*:\s*([^=]+?)\s*=\s*([0-9]*\.?[0-9]+)")


@dataclass
class Setting:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    source: str = ""

    @property
    def empty(self) -> bool:
        return self.temperature is None and self.top_p is None


def number(raw: object, bounds: tuple[float, float]) -> Optional[float]:
    """A number inside bounds, or None for an empty or invalid value."""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    low, high = bounds
    return value if low <= value <= high else None


def parse_rules(text: str) -> tuple[list[tuple[str, str, float]], list[str]]:
    """'platform:telegram=0.8, model:*coder*=0.1' becomes [(kind, pattern, temperature)].

    Returns the rules and the parts that could not be read, so the user can be told about them.
    """
    rules: list[tuple[str, str, float]] = []
    rejected: list[str] = []
    for part in re.split(r"[,;\n]+", text or ""):
        part = part.strip()
        if not part:
            continue
        match = _RULE.fullmatch(part)
        value = number(match.group(3), TEMPERATURE_RANGE) if match else None
        if not match or value is None:
            rejected.append(part)
            continue
        rules.append((match.group(1), match.group(2).strip().lower(), value))
    return rules, rejected


def resolve(*, default: Setting, rules: list[tuple[str, str, float]], session: Setting,
            platform: str = "", model: str = "", provider: str = "") -> Setting:
    facts = {"platform": (platform or "").lower(), "model": (model or "").lower(),
             "provider": (provider or "").lower()}
    result = Setting(default.temperature, default.top_p, "default" if not default.empty else "")
    for kind, pattern, value in rules:
        if fnmatch.fnmatch(facts[kind], pattern):
            result = Setting(value, result.top_p, f"rule {kind}:{pattern}")
            break
    if session.temperature is not None:
        result = Setting(session.temperature, result.top_p, "this session")
    if session.top_p is not None:
        result = Setting(result.temperature, session.top_p, "this session")
    return result
