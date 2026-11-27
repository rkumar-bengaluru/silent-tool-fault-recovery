import pytest

from stfr.faults import Fault, FaultyService
from stfr.svc import SCHEMAS, TOOLS, Service, ToolError


def mk():
    return Service({
        "orders": {"o1": {"customer": "ann", "item": "pen", "qty": 2, "status": "pending"},
                   "o2": {"customer": "bob", "item": "ink", "qty": 5, "status": "ready"}},
        "inventory": {"i1": {"sku": "pen", "stock": 10, "price": 2},
                      "i2": {"sku": "ink", "stock": 4, "price": 7}},
        "accounts": {"a1": {"owner": "ann", "balance": 100, "tier": "gold"},
                     "a2": {"owner": "bob", "balance": 50, "tier": "basic"}},
    })


def fs(fault, slot=0, seed=1):
    s = mk()
    return s, FaultyService(s, fault, slot, seed)


UPD = {"collection": "orders", "id": "o1", "field": "status", "value": "shipped"}
XFER = {"from_id": "a1", "to_id": "a2", "amount": 30}


def test_update_and_get_roundtrip():
    s = mk()
    assert s.call("update_field", UPD)["ok"]
    rec = s.call("get_record", {"collection": "orders", "id": "o1"})["record"]
    assert rec["status"] == "shipped"
    assert set(rec) == set(SCHEMAS["orders"])


def test_create_ids_are_sequential():
    s = mk()
    f = {"customer": "x", "item": "y", "qty": 1, "status": "pending"}
    a = s.call("create_record", {"collection": "orders", "fields": f})["id"]
    b = s.call("create_record", {"collection": "orders", "fields": f})["id"]
    assert (a, b) == ("o3", "o4")
    assert s.snapshot()["orders"][b] == f


def test_transfer_moves_balance_and_rejects_overdraft():
    s = mk()
    s.call("transfer", XFER)
    acc = s.snapshot()["accounts"]
    assert (acc["a1"]["balance"], acc["a2"]["balance"]) == (70, 80)
    with pytest.raises(ToolError):
        s.call("transfer", {**XFER, "amount": 1000})
    assert s.snapshot()["accounts"] == acc


def test_silent_drop_returns_success_without_effect():
    s, f = fs(Fault.SILENT_DROP)
    before = s.snapshot()
    assert f.call("update_field", UPD)["ok"] is True
    assert s.snapshot() == before


def test_silent_wrong_writes_other_field():
    s, f = fs(Fault.SILENT_WRONG)
    assert f.call("update_field", UPD)["ok"]
    rec = s.snapshot()["orders"]["o1"]
    assert rec["status"] == "pending"
    assert [k for k, v in rec.items() if v == "shipped"] not in ([], ["status"])


def test_silent_partial_transfer_debits_only():
    s, f = fs(Fault.SILENT_PARTIAL)
    assert f.call("transfer", XFER)["ok"]
    acc = s.snapshot()["accounts"]
    assert (acc["a1"]["balance"], acc["a2"]["balance"]) == (70, 50)


def test_loud_raises_and_leaves_state():
    s, f = fs(Fault.LOUD)
    before = s.snapshot()
    with pytest.raises(ToolError, match="503 service unavailable"):
        f.call("update_field", UPD)
    assert s.snapshot() == before


def test_silent_read_returns_plausible_wrong_value():
    for seed in range(8):
        s, f = fs(Fault.SILENT_READ, seed=seed)
        before = s.snapshot()
        rec = f.call("get_record", {"collection": "accounts", "id": "a1"})["record"]
        truth = before["accounts"]["a1"]
        diff = [k for k in truth if rec[k] != truth[k]]
        assert len(diff) == 1 and type(rec[diff[0]]) is type(truth[diff[0]])
        assert s.snapshot() == before


def test_delayed_write_visible_after_two_further_calls():
    s, f = fs(Fault.DELAYED)
    f.call("update_field", UPD)
    assert s.snapshot()["orders"]["o1"]["status"] == "shipped"
    g = {"collection": "orders", "id": "o1"}
    assert f.call("get_record", g)["record"]["status"] == "pending"
    assert f.independent_get("orders", "o1")["record"]["status"] == "pending"
    assert f.call("get_record", g)["record"]["status"] == "pending"
    assert f.call("get_record", g)["record"]["status"] == "shipped"


def test_delayed_counts_any_agent_call_even_erroring():
    s, f = fs(Fault.DELAYED)
    f.call("update_field", UPD)
    with pytest.raises(ToolError):
        f.call("get_record", {"collection": "orders", "id": "zz"})
    f.call("transfer", XFER)
    assert f.call("get_record", {"collection": "orders", "id": "o1"})["record"]["status"] == "shipped"


def test_independent_get_does_not_consume_staleness():
    s, f = fs(Fault.DELAYED)
    f.call("update_field", UPD)
    for _ in range(2):
        assert f.independent_get("orders", "o1")["record"]["status"] == "pending"
    assert f.call("get_record", {"collection": "orders", "id": "o1"})["record"]["status"] == "pending"


def test_fault_fires_once_then_reissue_works():
    s, f = fs(Fault.LOUD, slot=1)
    f.call("update_field", {**UPD, "value": "ready"})  # write 0: clean
    with pytest.raises(ToolError):
        f.call("update_field", UPD)  # write 1: fault
    f.call("update_field", UPD)  # reissue: normal
    assert s.snapshot()["orders"]["o1"]["status"] == "shipped"


def test_effect_postcondition_for_each_write_tool():
    s = mk()
    cases = [
        ("update_field", UPD),
        ("create_record", {"collection": "inventory", "fields": {"sku": "cap", "stock": 1, "price": 3}}),
        ("transfer", XFER),
        ("delete_record", {"collection": "orders", "id": "o2"}),
    ]
    for name, args in cases:
        before = s.snapshot()
        res = s.call(name, args)
        eff = TOOLS[name].effect(args, res, before)
        after = s.snapshot()
        assert eff
        for (c, i, fld), v in eff.items():
            if fld == "__exists__":
                assert (i in after[c]) is v
            else:
                assert after[c][i][fld] == v


G1 = {"collection": "orders", "id": "o1"}
DEL1 = {"collection": "orders", "id": "o1"}
NEWO = {"collection": "orders", "fields": {"customer": "x", "item": "y", "qty": 1, "status": "pending"}}
LALL = {"collection": "orders", "where": {}}


def test_delayed_delete_keeps_old_record_for_two_calls():
    s, f = fs(Fault.DELAYED)
    f.call("delete_record", DEL1)
    assert "o1" not in s.snapshot()["orders"]
    assert f.call("get_record", G1)["record"]["status"] == "pending"
    assert f.independent_get("orders", "o1")["record"]["status"] == "pending"
    assert f.call("get_record", G1)["record"]["status"] == "pending"
    with pytest.raises(ToolError):
        f.call("get_record", G1)


def test_delayed_list_records_stale_view():
    s, f = fs(Fault.DELAYED)
    f.call("update_field", UPD)
    rows = {r["id"]: r for r in f.call("list_records", LALL)["records"]}
    assert rows["o1"]["status"] == "pending"
    assert {r["id"]: r for r in f.call("list_records", LALL)["records"]}["o1"]["status"] == "pending"
    assert {r["id"]: r for r in f.call("list_records", LALL)["records"]}["o1"]["status"] == "shipped"
    s, f = fs(Fault.DELAYED)  # created within the delay: hidden
    f.call("create_record", NEWO)
    assert [r["id"] for r in f.call("list_records", LALL)["records"]] == ["o1", "o2"]
    s, f = fs(Fault.DELAYED)  # deleted within the delay: shown
    f.call("delete_record", DEL1)
    assert sorted(r["id"] for r in f.call("list_records", LALL)["records"]) == ["o1", "o2"]


def test_erroring_write_does_not_consume_fault():
    s, f = fs(Fault.SILENT_DROP, slot=0)
    with pytest.raises(ToolError):
        f.call("update_field", {**UPD, "value": "bogus"})
    assert not f.applied
    f.call("update_field", UPD)
    assert f.applied and f.applied_call == 1
    assert s.snapshot()["orders"]["o1"]["status"] == "pending"


def test_silent_wrong_waits_for_update_field():
    s, f = fs(Fault.SILENT_WRONG, slot=0)
    f.call("create_record", NEWO)
    assert not f.applied and "o3" in s.snapshot()["orders"]
    f.call("update_field", UPD)
    assert f.applied and f.applied_call == 1
    assert s.snapshot()["orders"]["o1"]["status"] == "pending"


def test_silent_partial_create_stores_some_fields():
    s, f = fs(Fault.SILENT_PARTIAL)
    rid = f.call("create_record", NEWO)["id"]
    rec = s.snapshot()["orders"][rid]
    assert 0 < len(rec) < 4 and f.applied


def test_applied_false_without_eligible_call():
    s, f = fs(Fault.SILENT_WRONG)
    f.call("get_record", G1)
    f.call("delete_record", DEL1)
    assert f.applied is False and f.applied_call is None


def test_silent_fault_fires_once():
    s, f = fs(Fault.SILENT_DROP)
    f.call("update_field", UPD)
    f.call("update_field", UPD)
    assert s.snapshot()["orders"]["o1"]["status"] == "shipped"


@pytest.mark.parametrize("coll,i", [("orders", "o1"), ("inventory", "i1")])
def test_silent_read_other_collections_and_list(coll, i):
    for seed in range(6):
        s, f = fs(Fault.SILENT_READ, seed=seed)
        truth = s.snapshot()[coll][i]
        rec = f.call("get_record", {"collection": coll, "id": i})["record"]
        d = [k for k in truth if rec[k] != truth[k]]
        assert len(d) == 1 and type(rec[d[0]]) is type(truth[d[0]])
        s, f = fs(Fault.SILENT_READ, seed=seed)
        row = f.call("list_records", {"collection": coll, "where": {}})["records"][0]
        assert [k for k in truth if row[k] != truth[k]] and s.snapshot()[coll][i] == truth


def test_silent_read_skips_empty_list():
    s, f = fs(Fault.SILENT_READ)
    f.call("list_records", {"collection": "orders", "where": {"status": "cancelled"}})
    assert not f.applied


@pytest.mark.parametrize("tool,args", [
    ("transfer", {**XFER, "amount": "30"}),
    ("transfer", {**XFER, "amount": True}),
    ("transfer", {**XFER, "amount": 0}),
    ("list_records", {"collection": "orders", "where": "x"}),
    ("update_field", {**UPD, "field": "qty", "value": "3"}),
    ("update_field", {**UPD, "field": "qty", "value": True}),
    ("create_record", {"collection": "orders", "fields": {"customer": 1, "item": "y", "qty": 1, "status": "pending"}}),
    ("create_record", {"collection": "orders", "fields": ["a"]}),
    ("get_record", {"collection": ["orders"], "id": "o1"}),
    ("nope", {}),
    ("get_record", {"collection": "orders"}),
])
def test_bad_args_raise_toolerror(tool, args):
    s = mk()
    before = s.snapshot()
    with pytest.raises(ToolError):
        s.call(tool, args)
    assert s.snapshot() == before


def test_list_records_where_id():
    s = mk()
    q = lambda w: s.call("list_records", {"collection": "accounts", "where": w})["records"]  # noqa: E731
    assert q({"id": "a2"}) == [{"id": "a2", "owner": "bob", "balance": 50, "tier": "basic"}]
    assert q({"id": "nope"}) == []
    with pytest.raises(ToolError):
        q({"id": 3})


def test_delayed_stale_view_applies_to_id_filter():
    s, f = fs(Fault.DELAYED)
    f.call("update_field", UPD)
    q = {"collection": "orders", "where": {"id": "o1"}}
    assert f.call("list_records", q)["records"][0]["status"] == "pending"
    assert f.call("list_records", q)["records"][0]["status"] == "pending"
    assert f.call("list_records", q)["records"][0]["status"] == "shipped"
