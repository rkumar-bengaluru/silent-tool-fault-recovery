"""Episode outcome from final service state (no model judge)."""
from __future__ import annotations

from collections import Counter

from .agent import Episode
from .tasks import Task

EX = "__exists__"


def _changes(initial: dict, final: dict) -> set:
    """(coll, id, field) diffs; record creation/deletion is (coll, id, "__exists__")."""
    out = set()
    for c in initial.keys() | final.keys():
        a, b = initial.get(c, {}), final.get(c, {})
        for i in a.keys() | b.keys():
            if (i in a) != (i in b):
                out.add((c, i, EX))
            for f in (a.get(i) or {}).keys() | (b.get(i) or {}).keys():
                if (a.get(i) or {}).get(f) != (b.get(i) or {}).get(f):
                    out.add((c, i, f))
    return out


def grade(task: Task, ep: Episode) -> dict:
    final = ep.final_state
    ch = _changes(task.initial, final)
    goal = {tuple(k.split("/")): v for k, v in task.goal.items()}
    goal_recs = {k[:2] for k in goal}
    allowed = {tuple(a) for a in task.allowed}
    unintended = [x for x in ch if x not in goal and not (x[2] == EX and x[:2] in goal_recs)]
    held = all((final.get(c, {}).get(i) or {}).get(f, object()) == v for (c, i, f), v in goal.items())
    outside = any(x[:2] not in allowed for x in ch)
    success = ep.status == "done" and held and not outside and not unintended
    created = Counter(c for c, i, f in ch if f == EX and i in final.get(c, {}))
    ref = Counter(s["args"].get("collection") for s in task.ref_steps if s["tool"] == "create_record")
    # The faulted step is the k of the ep.fault_applied_call-th (0-based) step that reached the service
    # (agent.py: tool is set for every service call, erroring ones included; parse failures never reach it).
    called = [s["k"] for s in ep.steps if s["tool"] is not None]
    faulted = called[ep.fault_applied_call] if ep.fault_applied and ep.fault_applied_call is not None \
        and ep.fault_applied_call < len(called) else None
    false_flags = len(ep.flags) if ep.fault in ("clean", "delayed") else sum(k != faulted for k in ep.flags)
    return {"success": success, "wrong_commit": ep.status == "done" and not success,
            "abstained": ep.status == "abstained", "unintended_changes": len(unintended),
            "duplicates": sum(max(0, n - ref[c]) for c, n in created.items()), "false_flags": false_flags,
            "extra_calls": len(called) - len(task.ref_steps), "readback_calls": ep.readback_calls,
            "fault_applied": ep.fault_applied, "fault_read_kind": ep.fault_read_kind}
