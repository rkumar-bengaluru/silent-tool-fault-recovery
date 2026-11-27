"""Fault injection. A fault ARMS at the slot-th successful call of its kind (0-based; write calls,
or read calls for SILENT_READ) and fires once on the first ELIGIBLE call at or after that.
Calls that raise ToolError never consume the fault nor advance the slot counter."""
from __future__ import annotations

import copy
import random
from enum import Enum

from .svc import TOOLS, Service, ToolError


class Fault(Enum):
    CLEAN = "clean"
    LOUD = "loud"
    SILENT_DROP = "silent_drop"
    SILENT_WRONG = "silent_wrong"
    SILENT_PARTIAL = "silent_partial"
    SILENT_READ = "silent_read"
    DELAYED = "delayed"


def _eligible(fault, tool, res):
    if fault == Fault.SILENT_WRONG:
        return tool == "update_field"
    if fault == Fault.SILENT_PARTIAL:
        return tool in ("transfer", "create_record")
    if fault == Fault.SILENT_READ:
        return tool == "get_record" or (tool == "list_records" and bool(res["records"]))
    return TOOLS[tool].kind == "write"  # LOUD, SILENT_DROP, DELAYED


class FaultyService:
    def __init__(self, svc: Service, fault: Fault, slot: int, rng_seed: int | str):
        self.svc, self.fault, self.slot = svc, fault, slot
        self.rng = random.Random(rng_seed)
        self.counts = {"read": 0, "write": 0}  # successful agent calls; independent_get never counts
        self.total = 0  # all agent calls, including erroring ones
        self.applied = False
        self.read_kind: str | None = None  # SILENT_READ only: "plausible" | "implausible"
        self.applied_call: int | None = None
        self.stale: dict[tuple, list] = {}  # (coll, id) -> [pre-write record or None, reads left]

    def snapshot(self) -> dict:
        return self.svc.snapshot()

    def independent_get(self, collection: str, id: str) -> dict:
        return self._read({"collection": collection, "id": id}, "get_record")

    def _stale_get(self, args):
        st = self.stale.get((args["collection"], args["id"]))
        if st and st[1] > 0:
            self.svc.check("get_record", args)
            if st[0] is None:
                raise ToolError(f"{args['collection']}/{args['id']} not found")
            return {"ok": True, "record": copy.deepcopy(st[0])}
        return None

    def _read(self, args, tool):
        if tool == "get_record":
            res = self._stale_get(args)
            return res if res is not None else self.svc.call(tool, args)
        res = self.svc.call(tool, args)
        c, where = args["collection"], args["where"]
        recs = {r["id"]: r for r in res["records"]}
        for (sc, i), st in self.stale.items():
            if sc != c or st[1] <= 0:
                continue
            pre = st[0]
            if pre is not None and all({"id": i, **pre}.get(k) == v for k, v in where.items()):
                recs[i] = {"id": i, **copy.deepcopy(pre)}
            elif i in recs:  # created within the delay, or stale pre-state doesn't match
                del recs[i]
        return {"ok": True, "records": list(recs.values())}

    def call(self, tool: str, args: dict) -> dict:
        live = list(self.stale.values())  # staleness ages by completed agent calls, never by reads
        try:
            return self._call(tool, args)
        finally:
            for st in live:
                st[1] = max(0, st[1] - 1)

    def _call(self, tool: str, args: dict) -> dict:
        n, self.total = self.total, self.total + 1
        saved, before = self.svc._save(), self.svc.snapshot()
        res = self._read(args, tool) if tool in ("get_record", "list_records") and self._valid(tool, args) \
            else self.svc.call(tool, args)  # ToolError propagates without consuming anything
        kind = TOOLS[tool].kind
        idx, self.counts[kind] = self.counts[kind], self.counts[kind] + 1
        armed = (self.fault != Fault.CLEAN and not self.applied and idx >= self.slot
                 and (kind == "read") == (self.fault == Fault.SILENT_READ))
        if not (armed and _eligible(self.fault, tool, res)):
            return res
        self.applied, self.applied_call = True, n
        f = self.fault
        if f == Fault.LOUD:
            self.svc._load(saved)
            raise ToolError("503 service unavailable")
        if f == Fault.SILENT_READ:
            return self._wrong_read(res, args["collection"])
        if f == Fault.DELAYED:
            for c, i in {k[:2] for k in TOOLS[tool].effect(args, res, before)}:
                self.stale[(c, i)] = [before.get(c, {}).get(i), 2]
        elif f == Fault.SILENT_PARTIAL and tool == "create_record":
            rec = self.svc.state[args["collection"]][res["id"]]
            keys = sorted(rec)
            for k in keys[max(1, len(keys) // 2):]:
                del rec[k]
        elif f == Fault.SILENT_PARTIAL:  # transfer: debit only
            self.svc.state["accounts"][args["to_id"]]["balance"] = before["accounts"][args["to_id"]]["balance"]
        else:  # SILENT_DROP, SILENT_WRONG
            self.svc._load(saved)
            if f == Fault.SILENT_WRONG:
                rec = self.svc.state[args["collection"]][args["id"]]
                rec[self.rng.choice([k for k in rec if k != args["field"]])] = args["value"]
        return res

    def _valid(self, tool, args):
        self.svc.check(tool, args)
        return True

    def _wrong_read(self, res, coll):
        rec = res["record"] if "record" in res else res["records"][0]
        ints = [f for f in rec if f != "id" and type(rec[f]) is int]
        field = self.rng.choice(ints or [f for f in rec if f != "id"])
        v = rec[field]
        bad = self.rng.random() < 0.5  # implausible: trips a contract domain check
        if isinstance(v, int):
            rec[field] = -(v + 1) if bad else (lambda n: n if n != v else v + 1)(int(v * self.rng.choice([0.5, 1.5])))
        elif bad and coll == "orders" and field == "status":
            rec[field] = "status_invalid"
        else:
            bad = False  # other strings have no checkable domain: plausible fallback
            others = sorted({r[field] for r in self.svc.state[coll].values() if field in r} - {v})
            rec[field] = self.rng.choice(others) if others else v + "_x"
        self.read_kind = "implausible" if bad else "plausible"
        return res
