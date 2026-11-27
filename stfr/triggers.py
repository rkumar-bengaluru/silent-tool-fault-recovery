"""Recovery triggers. Each takes (Step, TriggerCtx) and returns True to fire."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .contracts import check_contract
from .faults import FaultyService
from .svc import TOOLS, ToolError


@dataclass
class Step:
    index: int
    tool: str
    args: dict
    result: dict | None
    error: str | None
    confidence: float
    before: dict | None = None  # harness snapshot before a write; used only to compute the expected effect


@dataclass
class TriggerCtx:
    svc: FaultyService
    threshold: float
    counters: Counter


def _contract(s, c):
    return s.error is None and s.result is not None and check_contract(s.tool, s.args, s.result) is not None


def _readback(s, c):
    if s.error is not None or TOOLS[s.tool].kind != "write":
        return False
    if s.before is None:
        raise ValueError("write step needs before snapshot")
    exp = TOOLS[s.tool].effect(s.args, s.result, s.before)
    recs = {k[:2] for k in exp}
    c.counters["readback_calls"] += len(recs)
    got = {}
    for coll, i in recs:
        try:
            got[(coll, i)] = c.svc.independent_get(coll, i)["record"]
        except ToolError:
            got[(coll, i)] = None
    for (coll, i, f), v in exp.items():
        rec = got[(coll, i)]
        if f == "__exists__":
            if (rec is not None) != v:
                return True
        elif rec is None or rec.get(f) != v:
            return True
    return False


TRIGGERS: dict[str, Callable[[Step, TriggerCtx], bool]] = {
    "never": lambda s, c: False,
    "always": lambda s, c: True,
    "on_error": lambda s, c: s.error is not None,
    "on_uncertainty": lambda s, c: s.confidence < c.threshold,
    "on_contract": _contract,
    "on_readback": _readback,
    "readback_contract": lambda s, c: _contract(s, c) or _readback(s, c),
}
