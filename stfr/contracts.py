"""Return-value contract: sees only (tool, args, result), never service state."""
from __future__ import annotations

from .svc import ORDER_STATUSES, SCHEMAS


def _record_checks(coll, rec):
    if set(rec) != set(SCHEMAS[coll]):
        return "schema_fields"
    if any(type(v) is not SCHEMAS[coll][f] for f, v in rec.items()):
        return "field_type"
    if any(type(v) is int and v < 0 for v in rec.values()):
        return "non_negative"
    if coll == "orders" and rec["status"] not in ORDER_STATUSES:
        return "status_domain"
    return None


def check_contract(tool: str, args: dict, result: dict) -> str | None:
    if not isinstance(result, dict) or result.get("ok") is not True:
        return "has_ok"
    if tool == "get_record":
        if "record" not in result:
            return "has_record"
        return _record_checks(args["collection"], result["record"])
    if tool == "list_records":
        for r in result.get("records", []):
            bad = _record_checks(args["collection"], {k: v for k, v in r.items() if k != "id"})
            if bad:
                return bad
            if any(r.get(k) != v for k, v in args["where"].items()):
                return "echo_where"
    if tool == "update_field":
        rec = result.get("record")
        if not isinstance(rec, dict) or rec.get(args["field"], object()) != args["value"]:
            return "echo_effect"
        return _record_checks(args["collection"], rec)
    if tool == "transfer":
        if any(type(result.get(k)) is not int or result[k] < 0 for k in ("from_balance", "to_balance")):
            return "echo_effect"
    if tool == "create_record":
        rid = result.get("id")
        if not isinstance(rid, str) or rid[:1] != args["collection"][0]:
            return "echo_id"
        if result.get("record") != args["fields"]:
            return "echo_effect"
    return None
