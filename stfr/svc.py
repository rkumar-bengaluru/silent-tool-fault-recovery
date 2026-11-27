"""Deterministic mock service: three collections, six tools, effect postconditions."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

# Fixed field schema per collection (a contract checker verifies get_record results against it).
SCHEMAS: dict[str, dict[str, type]] = {
    "orders": {"customer": str, "item": str, "qty": int, "status": str},
    "inventory": {"sku": str, "stock": int, "price": int},
    "accounts": {"owner": str, "balance": int, "tier": str},
}
ORDER_STATUSES = {"pending", "ready", "shipped", "cancelled"}


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    kind: str  # "read" | "write"
    params: list
    # effect(args, result, before=None) -> {(coll, id, field): value}. `before` is the state
    # snapshot taken before the call; only `transfer` needs it. Deletion is
    # {(coll, id, "__exists__"): False}; creation also sets {(coll, id, "__exists__"): True}.
    effect: Callable | None = None


def _eff_update(args, result, before=None):
    return {(args["collection"], args["id"], args["field"]): args["value"]}


def _eff_create(args, result, before=None):
    c, i = args["collection"], result["id"]
    return {**{(c, i, f): v for f, v in args["fields"].items()}, (c, i, "__exists__"): True}


def _eff_delete(args, result, before=None):
    return {(args["collection"], args["id"], "__exists__"): False}


def _eff_transfer(args, result, before):
    a, f, t = args["amount"], args["from_id"], args["to_id"]
    acc = before["accounts"]
    return {("accounts", f, "balance"): acc[f]["balance"] - a,
            ("accounts", t, "balance"): acc[t]["balance"] + a}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("get_record", "read", ["collection", "id"], None),
    Tool("list_records", "read", ["collection", "where"], None),
    Tool("create_record", "write", ["collection", "fields"], _eff_create),
    Tool("update_field", "write", ["collection", "id", "field", "value"], _eff_update),
    Tool("delete_record", "write", ["collection", "id"], _eff_delete),
    Tool("transfer", "write", ["from_id", "to_id", "amount"], _eff_transfer),
]}


class Service:
    def __init__(self, state: dict[str, dict[str, dict]]):
        self.state = copy.deepcopy(state)
        # next id per collection: prefix = first letter, sequential, never reused
        self._n = {c: max([int(i[1:]) for i in r if i[1:].isdigit()] + [0]) for c, r in self.state.items()}

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state)

    def _save(self):
        return self.snapshot(), dict(self._n)

    def _load(self, saved):
        self.state, self._n = copy.deepcopy(saved[0]), dict(saved[1])

    def _coll(self, c):
        if c not in self.state:
            raise ToolError(f"unknown collection {c!r}")
        return self.state[c]

    def _rec(self, c, i):
        r = self._coll(c).get(i)
        if r is None:
            raise ToolError(f"{c}/{i} not found")
        return r

    def check(self, tool: str, args: dict) -> None:
        """Validate tool name, required params and types; raises ToolError."""
        if tool not in TOOLS:
            raise ToolError(f"unknown tool {tool!r}")
        if not isinstance(args, dict):
            raise ToolError("args must be an object")
        missing = [p for p in TOOLS[tool].params if p not in args]
        if missing:
            raise ToolError(f"missing args: {missing}")
        a = args
        for k in ("collection", "id", "field", "from_id", "to_id"):
            if k in a and not isinstance(a[k], str):
                raise ToolError(f"{k} must be a string")
        if "collection" in a:
            self._coll(a["collection"])
        c = a.get("collection")
        ok = lambda f, v: type(v) is SCHEMAS[c][f]  # noqa: E731  (bool/str rejected for int fields)
        if tool == "list_records":
            if not isinstance(a["where"], dict):
                raise ToolError("where must be an object")
            for f, v in a["where"].items():
                if f == "id":
                    if not isinstance(v, str):
                        raise ToolError("bad where clause on 'id'")
                elif f not in SCHEMAS[c] or not ok(f, v):
                    raise ToolError(f"bad where clause on {f!r}")
        elif tool == "create_record":
            if not isinstance(a["fields"], dict) or set(a["fields"]) != set(SCHEMAS[c]):
                raise ToolError(f"fields must be an object with exactly {sorted(SCHEMAS[c])}")
            for f, v in a["fields"].items():
                if not ok(f, v):
                    raise ToolError(f"field {f!r} must be {SCHEMAS[c][f].__name__}")
        elif tool == "update_field":
            if a["field"] not in SCHEMAS[c]:
                raise ToolError(f"unknown field {a['field']!r}")
            if not ok(a["field"], a["value"]):
                raise ToolError(f"field {a['field']!r} must be {SCHEMAS[c][a['field']].__name__}")
        elif tool == "transfer":
            if type(a["amount"]) is not int or a["amount"] <= 0:
                raise ToolError("amount must be a positive integer")

    def call(self, tool: str, args: dict) -> dict:
        self.check(tool, args)
        return getattr(self, "_" + tool)(**{p: args[p] for p in TOOLS[tool].params})

    def _get_record(self, collection, id):
        return {"ok": True, "record": copy.deepcopy(self._rec(collection, id))}

    def _list_records(self, collection, where):
        return {"ok": True, "records": [{"id": i, **copy.deepcopy(r)} for i, r in self._coll(collection).items()
                                        if all({"id": i, **r}.get(k) == v for k, v in where.items())]}

    def _create_record(self, collection, fields):
        rows = self._coll(collection)
        if set(fields) != set(SCHEMAS[collection]):
            raise ToolError(f"fields must be exactly {sorted(SCHEMAS[collection])}")
        self._n[collection] += 1
        rid = f"{collection[0]}{self._n[collection]}"
        rows[rid] = copy.deepcopy(fields)
        return {"ok": True, "id": rid, "record": copy.deepcopy(fields)}

    def _update_field(self, collection, id, field, value):
        rec = self._rec(collection, id)
        if field not in SCHEMAS[collection]:
            raise ToolError(f"unknown field {field!r}")
        if collection == "orders" and field == "status" and value not in ORDER_STATUSES:
            raise ToolError(f"invalid status {value!r}")
        rec[field] = value
        return {"ok": True, "record": copy.deepcopy(rec)}

    def _delete_record(self, collection, id):
        self._rec(collection, id)
        del self.state[collection][id]
        return {"ok": True}

    def _transfer(self, from_id, to_id, amount):
        src, dst = self._rec("accounts", from_id), self._rec("accounts", to_id)
        if amount <= 0 or src["balance"] < amount:
            raise ToolError("invalid amount or insufficient funds")
        src["balance"] -= amount
        dst["balance"] += amount
        return {"ok": True, "from_balance": src["balance"], "to_balance": dst["balance"]}
