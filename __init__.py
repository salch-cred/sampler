"""sampler: set temperature and top_p for your agent, per platform, model, provider or session.

  temperature: "0.3"                        default for every request
  top_p: ""                                 left to the provider unless set
  rules: "platform:telegram=0.8, model:*coder*=0.1"
  /sampler 0.2 | /sampler top_p 0.9 | /sampler reset | /sampler status

Models Hermes itself sends without a temperature (reasoning models and similar) are never changed.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional

from . import request as req
from .rules import TEMPERATURE_RANGE, TOP_P_RANGE, Setting, number, parse_rules, resolve

if TYPE_CHECKING:
    from hermes_cli.plugins import PluginContext

_CTX: Any = None
_LOCK = threading.Lock()
_SESSIONS: dict[str, Setting] = {}
_COUNTS = {"set": 0, "left": 0}


def _setting(key: str) -> str:
    try:
        return str(_CTX.get_config(key, "") or "") if _CTX is not None else ""
    except Exception:
        return ""


def _default() -> Setting:
    return Setting(number(_setting("temperature"), TEMPERATURE_RANGE),
                   number(_setting("top_p"), TOP_P_RANGE))


def _on_request(request: Optional[dict] = None, *, session_id: str = "", platform: str = "",
                model: str = "", provider: str = "", base_url: str = "", api_mode: str = "",
                **_: Any) -> Optional[dict]:
    if not isinstance(request, dict):
        return None
    rules, _rejected = parse_rules(_setting("rules"))
    with _LOCK:
        session = _SESSIONS.get(session_id or "", Setting())
    setting = resolve(default=_default(), rules=rules, session=session, platform=platform,
                      model=model, provider=provider)
    changed, reason = req.apply(request, setting, api_mode=api_mode, model=model,
                                base_url=base_url, provider=provider)
    with _LOCK:
        _COUNTS["set" if changed is not None else "left"] += 1
    if changed is None:
        return None
    return {"request": changed, "source": "sampler", "reason": reason}


def _status(session_id: str) -> str:
    rules, rejected = parse_rules(_setting("rules"))
    default = _default()
    with _LOCK:
        session = _SESSIONS.get(session_id or "", Setting())
        counts = dict(_COUNTS)
    show = lambda v: "provider default" if v is None else f"{v:g}"
    lines = [f"**Sampler** default temperature: {show(default.temperature)}, top_p: {show(default.top_p)}.",
             "Rules: " + (", ".join(f"{k}:{p}={v:g}" for k, p, v in rules) or "none") + ".",
             f"This session: temperature {show(session.temperature)}, top_p {show(session.top_p)}.",
             f"Requests changed: {counts['set']}, left as they were: {counts['left']}."]
    if rejected:
        lines.append("Ignored rules (not readable): " + ", ".join(rejected))
    return "\n".join(lines)


def sampler_command(raw_args: str = "", session_id: str = "", **_: Any) -> str:
    words = (raw_args or "").split()
    action = words[0].lower() if words else "status"
    sid = session_id or ""
    if action == "status":
        return _status(sid)
    if action == "reset":
        with _LOCK:
            _SESSIONS.pop(sid, None)
        return "Sampler: this session is back to the configured settings."
    if action == "top_p":
        value = number(words[1] if len(words) > 1 else "", TOP_P_RANGE)
        if value is None:
            return "Give a top_p between 0 and 1, for example `/sampler top_p 0.9`."
        with _LOCK:
            current = _SESSIONS.get(sid, Setting())
            _SESSIONS[sid] = Setting(current.temperature, value, "this session")
        return f"Sampler: top_p {value:g} for this session."
    value = number(action, TEMPERATURE_RANGE)
    if value is None:
        return ("Usage: `/sampler 0.2` (temperature 0 to 2 for this session), `/sampler top_p 0.9`, "
                "`/sampler reset`, `/sampler status`.")
    with _LOCK:
        current = _SESSIONS.get(sid, Setting())
        _SESSIONS[sid] = Setting(value, current.top_p, "this session")
    return f"Sampler: temperature {value:g} for this session."


def register(ctx: "PluginContext") -> None:
    global _CTX
    _CTX = ctx
    ctx.register_middleware("llm_request", _on_request)
    ctx.register_command("sampler", sampler_command,
                         description="Set temperature and top_p for this session or show the settings",
                         args_hint="[0.2 | top_p 0.9 | reset | status]")
