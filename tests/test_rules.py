"""Rules: reading them, rejecting bad ones, and which setting wins."""

import pytest
from sampler.rules import TEMPERATURE_RANGE, Setting, number, parse_rules, resolve


@pytest.mark.parametrize("raw,value", [("0.2", 0.2), (" 1 ", 1.0), ("0", 0.0), ("2", 2.0),
                                       ("", None), ("abc", None), ("2.5", None), ("-0.1", None),
                                       (None, None)])
def test_number(raw, value):
    assert number(raw, TEMPERATURE_RANGE) == value


def test_parse_rules_reads_good_rules_and_reports_bad_ones():
    rules, rejected = parse_rules("platform:telegram=0.8, model:*coder*=0.1; provider: nvidia = 0.4,"
                                  " nonsense, model:x=9")
    assert rules == [("platform", "telegram", 0.8), ("model", "*coder*", 0.1), ("provider", "nvidia", 0.4)]
    assert rejected == ["nonsense", "model:x=9"]


def test_empty_rules():
    assert parse_rules("") == ([], [])


def test_default_applies_when_nothing_else_matches():
    s = resolve(default=Setting(0.3), rules=[], session=Setting(), platform="cli", model="m")
    assert (s.temperature, s.source) == (0.3, "default")


def test_first_matching_rule_wins():
    rules, _ = parse_rules("model:*coder*=0.1, platform:cli=0.9")
    s = resolve(default=Setting(0.3), rules=rules, session=Setting(), platform="cli",
                model="qwen/qwen3-coder")
    assert (s.temperature, s.source) == (0.1, "rule model:*coder*")


def test_rules_match_case_insensitively():
    rules, _ = parse_rules("platform:TELEGRAM=0.8")
    assert resolve(default=Setting(), rules=rules, session=Setting(),
                   platform="Telegram").temperature == 0.8


def test_session_override_beats_rules_and_default():
    rules, _ = parse_rules("platform:cli=0.9")
    s = resolve(default=Setting(0.3, 0.95), rules=rules, session=Setting(0.0), platform="cli")
    assert (s.temperature, s.top_p, s.source) == (0.0, 0.95, "this session")


def test_nothing_configured_is_empty():
    assert resolve(default=Setting(), rules=[], session=Setting()).empty
