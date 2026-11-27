"""The one fixed agent loop shared by every arm; arms differ only in trigger and response."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

from .faults import Fault, FaultyService
from .svc import SCHEMAS, TOOLS, Service, ToolError
from .tasks import Task
from .triggers import TRIGGERS, Step, TriggerCtx

_TOOL_LINES = "\n".join(f"- {t.name}({', '.join(t.params)})" for t in TOOLS.values())
_COLLS = "\n".join(f"- {c}: {', '.join(f'{f}:{t.__name__}' for f, t in s.items())}" for c, s in SCHEMAS.items())
SYSTEM_PROMPT = f"""You operate a small record service by calling tools to complete the user's task.

Tools (args are a JSON object with exactly these names):
{_TOOL_LINES}

Collections and their fields (ids look like o1, i2, a3):
{_COLLS}

In list_records, "where" is an object of field: value equality filters ({{}} matches all). In create_record, "fields" must hold every field of the collection.

Write tools echo their effect: update_field returns the record after the write; transfer returns from_balance and to_balance after the transfer; create_record returns the new id and the stored record; delete_record returns only ok.

Reply with one JSON object only: {{"thought": str, "tool": str, "args": object, "confidence": number 0-1}} to call a tool, or {{"done": true, "summary": str}} when the task is complete.
Each tool result comes back as "RESULT step <k>: <json>" or "ERROR step <k>: <msg>"."""
NOTICE = ("NOTICE: a check flagged a possible problem with step {k} ({tool}). Its result may not match "
          "the real state. Decide how to continue: you may re-read state, re-issue the call, or change your plan.")


@dataclass
class Episode:
    task_id: str
    fault: str
    trigger: str
    response: str
    seed: int
    status: str  # done | abstained | budget_exhausted | parse_failure
    steps: list[dict]
    flags: list[int]
    final_state: dict
    tokens_in: int
    tokens_out: int
    cost_usd: float
    readback_calls: int
    fault_applied: bool
    fault_applied_call: int | None
    fault_read_kind: str | None = None


def _parse(text: str) -> dict:
    a, b = text.find("{"), text.rfind("}")  # tolerate prose / code fences around the object
    obj = json.loads(text[a:b + 1])
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    if obj.get("done") is True:
        return obj
    if not isinstance(obj.get("tool"), str) or obj["tool"] not in TOOLS:
        raise ValueError(f"unknown tool {obj.get('tool')!r}")
    if not isinstance(obj.get("args", {}), dict):
        raise ValueError("args must be an object")
    return obj


def run_episode(task: Task, chat, *, fault: Fault, slot: int, trigger: str, response: str,
                threshold: float, seed: int, max_steps: int = 15) -> Episode:
    base = Service(task.initial)
    svc = FaultyService(base, fault, slot, f"{task.id}|{fault.value}|{seed}")
    ctx = TriggerCtx(svc, threshold, Counter())
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task.prompt}]
    steps, flags, status, bad = [], [], "budget_exhausted", 0
    tin = tout = 0
    cost = 0.0
    for k in range(1, max_steps + 1):
        r = chat.chat(msgs, seed=seed)
        tin, tout, cost = tin + r.input_tokens, tout + r.output_tokens, cost + (0 if r.cached else r.cost_usd)
        msgs.append({"role": "assistant", "content": r.text})
        row = {"k": k, "action": r.text, "tool": None, "args": None, "result": None, "error": None,
               "confidence": None, "flagged": False}
        steps.append(row)
        try:
            act = _parse(r.text)
        except ValueError as e:  # includes json.JSONDecodeError
            bad += 1
            row["error"] = f"invalid action: {e}"
            msgs.append({"role": "user", "content": f"ERROR step {k}: {row['error']}"})
            if bad == 3:
                status = "parse_failure"
                break
            continue
        bad = 0
        row["action"] = act
        if act.get("done") is True:
            status = "done"
            break
        tool, args = act["tool"], act.get("args", {})
        conf = act.get("confidence")  # missing / non-numeric / NaN / inf -> 1.0; finite values clamped to [0, 1]
        conf = 1.0 if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) \
            else min(1.0, max(0.0, float(conf)))
        before = base.snapshot() if TOOLS[tool].kind == "write" else None
        res = err = None
        try:
            res = svc.call(tool, args)
            msgs.append({"role": "user", "content": f"RESULT step {k}: {json.dumps(res)}"})
        except ToolError as e:
            err = str(e)
            msgs.append({"role": "user", "content": f"ERROR step {k}: {err}"})
        row.update(tool=tool, args=args, result=res, error=err, confidence=conf)
        if TRIGGERS[trigger](Step(k, tool, args, res, err, conf, before), ctx):
            row["flagged"] = True
            flags.append(k)
            if response == "abstain":
                status = "abstained"
                break
            msgs.append({"role": "user", "content": NOTICE.format(k=k, tool=tool)})
    return Episode(task.id, fault.value, trigger, response, seed, status, steps, flags, base.snapshot(),
                   tin, tout, cost, ctx.counters["readback_calls"], svc.applied, svc.applied_call, svc.read_kind)
