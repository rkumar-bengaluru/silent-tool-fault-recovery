from collections import Counter

import pytest

from stfr.contracts import check_contract
from stfr.faults import Fault, FaultyService
from stfr.svc import TOOLS, Service, ToolError
from stfr.triggers import TRIGGERS, Step, TriggerCtx
from tests.test_svc import UPD, XFER, mk

CREATE = {"collection": "orders", "fields": {"customer": "x", "item": "y", "qty": 1, "status": "pending"}}
DELETE = {"collection": "orders", "id": "o1"}
GET = {"collection": "orders", "id": "o1"}


def run(fault, tool, args, slot=0, threshold=0.5):
    """Make one faulty write; return (step, ctx)."""
    s = mk()
    f = FaultyService(s, fault, slot, 1)
    before = f.snapshot()
    err = res = None
    try:
        res = f.call(tool, args)
    except ToolError as e:
        err = str(e)
    step = Step(0, tool, args, res, err, 1.0, before if TOOLS[tool].kind == "write" else None)
    return step, TriggerCtx(f, threshold, Counter())


def test_contract_flags_negative_balance_and_bad_status():
    acc = {"collection": "accounts", "id": "a1"}
    assert check_contract("get_record", acc, {"ok": True, "record": {"owner": "a", "balance": -5, "tier": "g"}}) == "non_negative"
    bad = {"customer": "a", "item": "b", "qty": 1, "status": "lost"}
    assert check_contract("get_record", GET, {"ok": True, "record": bad}) == "status_domain"
    assert check_contract("update_field", UPD, {}) == "has_ok"
    assert check_contract("get_record", GET, {"ok": True, "record": {"customer": "a"}}) == "schema_fields"
    assert check_contract("create_record", CREATE, {"ok": True, "id": "a3"}) == "echo_id"


def test_contract_passes_clean_results_for_all_tools():
    s = mk()
    for tool, args in [("get_record", GET), ("list_records", {"collection": "orders", "where": {"status": "ready"}}),
                       ("create_record", CREATE), ("update_field", UPD), ("transfer", XFER),
                       ("delete_record", {"collection": "orders", "id": "o2"})]:
        assert check_contract(tool, args, s.call(tool, args)) is None, tool


@pytest.mark.parametrize("fault,tool,args", [
    (Fault.SILENT_DROP, "update_field", UPD), (Fault.SILENT_DROP, "delete_record", DELETE),
    (Fault.SILENT_DROP, "transfer", XFER), (Fault.SILENT_WRONG, "update_field", UPD),
    (Fault.SILENT_PARTIAL, "transfer", XFER), (Fault.SILENT_PARTIAL, "create_record", CREATE)])
def test_readback_fires_on_silent_drop_and_wrong_and_partial(fault, tool, args):
    step, ctx = run(fault, tool, args)
    assert TRIGGERS["on_readback"](step, ctx)


@pytest.mark.parametrize("tool,args", [("update_field", UPD), ("delete_record", DELETE), ("transfer", XFER),
                                       ("create_record", CREATE)])
def test_readback_quiet_on_clean_write(tool, args):
    step, ctx = run(Fault.CLEAN, tool, args)
    assert not TRIGGERS["on_readback"](step, ctx)
    assert not TRIGGERS["readback_contract"](step, ctx)


def test_readback_false_alarm_on_delayed_write():
    step, ctx = run(Fault.DELAYED, "update_field", UPD)
    assert TRIGGERS["on_readback"](step, ctx)


def test_contract_cannot_see_silent_drop():
    step, ctx = run(Fault.SILENT_DROP, "update_field", UPD)
    assert check_contract(step.tool, step.args, step.result) is None
    assert not TRIGGERS["on_contract"](step, ctx)


def test_on_error_only_on_loud():
    loud, ctx = run(Fault.LOUD, "update_field", UPD)
    clean, ctx2 = run(Fault.CLEAN, "update_field", UPD)
    assert TRIGGERS["on_error"](loud, ctx) and not TRIGGERS["on_error"](clean, ctx2)
    assert not TRIGGERS["on_readback"](loud, ctx)  # no read-back on errored writes
    assert TRIGGERS["always"](clean, ctx2) and not TRIGGERS["never"](loud, ctx)


def test_uncertainty_threshold():
    step, ctx = run(Fault.CLEAN, "update_field", UPD, threshold=0.5)
    for conf, want in [(0.49, True), (0.5, False), (0.9, False)]:
        step.confidence = conf
        assert TRIGGERS["on_uncertainty"](step, ctx) is want


def test_readback_counts_extra_calls():
    step, ctx = run(Fault.CLEAN, "transfer", XFER)
    TRIGGERS["on_readback"](step, ctx)
    assert ctx.counters["readback_calls"] == 2  # two accounts
    assert (ctx.svc.counts, ctx.svc.total) == ({"read": 0, "write": 1}, 1)  # agent counters untouched
    r, ctx2 = run(Fault.CLEAN, "get_record", GET)
    assert not TRIGGERS["on_readback"](r, ctx2) and ctx2.counters["readback_calls"] == 0


def test_readback_contract_short_circuits_on_contract():
    step, ctx = run(Fault.CLEAN, "create_record", CREATE)
    step.result = {"ok": True, "id": "a3"}  # violates echo_id
    assert TRIGGERS["readback_contract"](step, ctx)
    assert ctx.counters["readback_calls"] == 0


def test_readback_write_step_needs_before():
    step, ctx = run(Fault.CLEAN, "transfer", XFER)
    step.before = None
    with pytest.raises(ValueError, match="needs before snapshot"):
        TRIGGERS["on_readback"](step, ctx)
