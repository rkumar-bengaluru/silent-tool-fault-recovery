"""Pre-freeze design fixes (task 9b)."""
import pytest

from stfr.contracts import check_contract
from stfr.faults import Fault
from stfr.svc import SCHEMAS
from tests.test_svc import UPD, XFER, fs, mk

CREATE = {"collection": "orders", "fields": {"customer": "x", "item": "y", "qty": 1, "status": "pending"}}


# ---- F1
def test_write_echoes_and_deepcopy():
    s = mk()
    r = s.call("update_field", UPD)
    assert r == {"ok": True, "record": {"customer": "ann", "item": "pen", "qty": 2, "status": "shipped"}}
    s.call("update_field", {**UPD, "value": "ready"})
    assert r["record"]["status"] == "shipped"
    assert s.call("transfer", XFER) == {"ok": True, "from_balance": 70, "to_balance": 80}
    c = s.call("create_record", CREATE)
    assert c == {"ok": True, "id": "o3", "record": CREATE["fields"]}
    s.state["orders"]["o3"]["qty"] = 9
    assert c["record"]["qty"] == 1
    assert s.call("delete_record", {"collection": "orders", "id": "o1"}) == {"ok": True}


@pytest.mark.parametrize("fault,tool,args", [
    (Fault.SILENT_DROP, "update_field", UPD), (Fault.SILENT_WRONG, "update_field", UPD),
    (Fault.DELAYED, "update_field", UPD), (Fault.SILENT_DROP, "transfer", XFER),
    (Fault.SILENT_PARTIAL, "transfer", XFER), (Fault.DELAYED, "transfer", XFER),
    (Fault.SILENT_PARTIAL, "create_record", CREATE), (Fault.SILENT_DROP, "create_record", CREATE),
])
def test_silent_write_faults_return_intended_echo(fault, tool, args):
    _, f = fs(Fault.CLEAN)
    want = f.call(tool, args)
    s, f = fs(fault)
    assert f.call(tool, args) == want and f.applied


def test_contract_echo_effect_and_field_type():
    ok = {"ok": True, "record": {"customer": "ann", "item": "pen", "qty": 2, "status": "shipped"}}
    assert check_contract("update_field", UPD, ok) is None
    assert check_contract("update_field", UPD, {"ok": True, "record": {**ok["record"], "status": "ready"}}) == "echo_effect"
    assert check_contract("update_field", UPD, {"ok": True}) == "echo_effect"
    assert check_contract("transfer", XFER, {"ok": True, "from_balance": 70, "to_balance": 80}) is None
    assert check_contract("transfer", XFER, {"ok": True, "from_balance": -1, "to_balance": 80}) == "echo_effect"
    assert check_contract("transfer", XFER, {"ok": True, "from_balance": "70", "to_balance": 80}) == "echo_effect"
    assert check_contract("create_record", CREATE, {"ok": True, "id": "o3", "record": CREATE["fields"]}) is None
    assert check_contract("create_record", CREATE, {"ok": True, "id": "o3", "record": {"customer": "x"}}) == "echo_effect"
    bad = {**ok["record"], "qty": "2"}
    assert check_contract("get_record", {"collection": "orders", "id": "o1"}, {"ok": True, "record": bad}) == "field_type"
    assert check_contract("get_record", {"collection": "orders", "id": "o1"},
                          {"ok": True, "record": {**ok["record"], "qty": True}}) == "field_type"


def test_prompt_documents_echoes():
    from stfr.agent import SYSTEM_PROMPT
    assert "from_balance" in SYSTEM_PROMPT


# ---- F3
def _read(seed, coll="orders", i="o1"):
    s, f = fs(Fault.SILENT_READ, seed=seed)
    res = f.call("get_record", {"collection": coll, "id": i})
    return f, res, {"collection": coll, "id": i}


def test_silent_read_kinds():
    seen = {}
    for seed in range(30):
        f, res, a = _read(seed)
        assert f.read_kind in ("plausible", "implausible")
        seen.setdefault(f.read_kind, []).append(check_contract("get_record", a, res))
        truth = mk().state["orders"]["o1"]
        assert [k for k in truth if res['record'][k] != truth[k]] == ['qty']
    assert set(seen) == {"plausible", "implausible"}
    assert all(x is None for x in seen["plausible"])
    assert all(x in ("non_negative", "status_domain") for x in seen["implausible"])


def test_silent_read_targets_int_field():
    for seed in range(30):
        f, res, _ = _read(seed, "accounts", "a1")
        truth = mk().state["accounts"]["a1"]
        assert [k for k in truth if res["record"][k] != truth[k]] == ["balance"]


def test_read_kind_in_episode():
    from tests.test_agent import DONE, call, ep, GET
    e, _ = ep([call("get_record", GET), DONE], fault=Fault.SILENT_READ)
    assert e.fault_applied and e.fault_read_kind in ("plausible", "implausible")
    e, _ = ep([call("get_record", GET), DONE])
    assert e.fault_read_kind is None
    from stfr.grade import grade
    from tests.test_agent import task
    assert grade(task(), e)["fault_read_kind"] is None


# ---- F4
def _rows(diffs):
    rows = []
    for t, d in enumerate(diffs):
        rows += [{"task_id": f"t{t}", "arm": "b", "success": False}, {"task_id": f"t{t}", "arm": "a", "success": bool(d)}]
    return rows


def test_permutation_p():
    from stfr.stats import mde_tasks, permutation_p
    assert permutation_p(_rows([0] * 5), "a", "b") == 1.0
    assert permutation_p(_rows([1] * 20), "a", "b", n=2000) < 0.01
    r = _rows([1, 0, 1, 1, 0, 1])
    assert permutation_p(r, "a", "b", n=500, seed=3) == permutation_p(r, "a", "b", n=500, seed=3)
    assert mde_tasks(100, 0.5) == pytest.approx((1.959964 + 0.841621) * 0.05, rel=1e-4)


def test_tests_emit_permutation_by_fault_and_delayed_harm(tmp_path):
    import json
    from stfr import run
    d = tmp_path / "r"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model": run.PRIMARY_MODEL, "seeds": [0], "split": "dev"}))
    (d / "summary.json").write_text(json.dumps({"run_id": "r", "mode": "final", "complete": True, "cost_usd": 0.0}))
    g = lambda ok: {"success": ok, "wrong_commit": False, "abstained": False, "false_flags": 0,
                    "extra_calls": 0, "readback_calls": 0, "duplicates": 0}
    rows = []
    for t in range(6):
        for f in run.ALL_FAULTS:
            for arm, ok in (("never", True), ("on_readback", f != "delayed"), ("on_contract", False), ("readback_contract", False)):
                rows.append({"task_id": f"t{t}", "fault": f, "seed": 0, "trigger": arm, "response": "recover",
                             "tokens_in": 1, "tokens_out": 1, "grade": g(ok)})
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    run.aggregate([d], tmp_path / "o.json")
    tests = json.loads((tmp_path / "o.json").read_text())["tests"]
    t = tests[0]
    assert "p_mcnemar" in t and 0 < t["p"] <= 1 and set(t["by_fault"]) == set(run.SILENT_WRITE)
    assert set(t["by_fault"]["silent_drop"]) == {"diff", "rescue", "harm"}
    p3 = tests[2]
    assert set(p3["by_fault"]) == set(run.ALL_FAULTS) and p3["delayed_harm"] == 6  # base ok, arm failed under delayed
    assert "delayed_harm" not in tests[0]


# ---- F5
def _run_dir(tmp_path, name, summary, split="dev"):
    import json
    from stfr import run
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model": run.PRIMARY_MODEL, "seeds": [0], "split": split}))
    (d / "summary.json").write_text(json.dumps({"run_id": name, "cost_usd": 0.0, **summary}))
    g = {"success": True, "wrong_commit": False, "abstained": False, "false_flags": 0,
         "extra_calls": 0, "readback_calls": 0, "duplicates": 0}
    rows = [{"task_id": "t0", "fault": "silent_read", "seed": 0, "trigger": a, "response": "recover",
             "tokens_in": 1, "tokens_out": 1, "grade": g} for a in ("never", "on_contract")]
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_aggregate_excludes_non_final_or_incomplete_from_tests(tmp_path):
    import json
    from stfr import run
    ds = [_run_dir(tmp_path, "dev", {"mode": "dev", "complete": True}),
          _run_dir(tmp_path, "part", {"mode": "final", "complete": False}),
          _run_dir(tmp_path, "nomode", {"complete": True})]
    run.aggregate(ds, tmp_path / "o.json")
    r = json.loads((tmp_path / "o.json").read_text())
    assert all(t["n_pairs"] == 0 for t in r["tests"]) and r["cells"]
    assert [x["in_tests"] for x in r["runs"]] == [False] * 3
    ok = _run_dir(tmp_path, "fin", {"mode": "final", "complete": True})
    run.aggregate(ds + [ok], tmp_path / "o.json")
    r = json.loads((tmp_path / "o.json").read_text())
    assert r["tests"][1]["n_pairs"] == 1 and r["runs"][-1]["in_tests"] is True


def test_aggregate_rejects_mixed_splits(tmp_path):
    from stfr import run
    ds = [_run_dir(tmp_path, "a", {"mode": "final", "complete": True}, "testA"),
          _run_dir(tmp_path, "b", {"mode": "final", "complete": True}, "testB")]
    with pytest.raises(SystemExit) as e:
        run.aggregate(ds, tmp_path / "o.json")
    assert e.value.code == 2


def test_cells_separate_test_and_non_test_rows(tmp_path):
    import json
    from stfr import run
    ds = [_run_dir(tmp_path, "dev", {"mode": "dev", "complete": True}),
          _run_dir(tmp_path, "fin", {"mode": "final", "complete": True})]
    run.aggregate(ds, tmp_path / "o.json")
    cells = [c for c in json.loads((tmp_path / "o.json").read_text())["cells"]
             if c["fault"] == "silent_read" and c["arm"] == "never/recover"]
    assert sorted(c["in_tests"] for c in cells) == [False, True] and all(c["n"] == 1 for c in cells)


def test_delayed_harm_present_without_pairs(tmp_path):
    import json
    from stfr import run
    run.aggregate([_run_dir(tmp_path, "fin", {"mode": "final", "complete": True})], tmp_path / "o.json")
    assert json.loads((tmp_path / "o.json").read_text())["tests"][2]["delayed_harm"] == 0


# ---- rng seeding: per (task, fault, seed), not per seed
def _read_kind_for(task_id, seed=0):
    from stfr.faults import FaultyService
    s = mk()
    f = FaultyService(s, Fault.SILENT_READ, 0, f"{task_id}|silent_read|{seed}")
    res = f.call("get_record", {"collection": "orders", "id": "o1"})
    return f.read_kind, res


def test_read_kinds_vary_across_tasks_and_repeat():
    kinds = [_read_kind_for(f"t{i}")[0] for i in range(40)]
    assert 10 <= kinds.count("implausible") <= 30
    assert _read_kind_for("t3") == _read_kind_for("t3")


def test_run_episode_seeds_fault_rng_per_task_not_seed_or_arm():
    from stfr.agent import run_episode
    from stfr.tasks import Task
    from tests.test_agent import DONE, GET, ScriptedChat, call
    def go(tid, trig):
        t = Task(tid, "p", mk().state, {"orders/o1/status": "shipped"}, [["orders", "o1"]], [], 1, 0)
        e = run_episode(t, ScriptedChat([call("get_record", GET), DONE]), fault=Fault.SILENT_READ, slot=0,
                        trigger=trig, response="recover", threshold=0.5, seed=0)
        return e.fault_read_kind, e.steps[0]["result"]
    assert go("t1", "never") == go("t1", "on_contract")
    assert len({go(f"t{i}", "never")[0] for i in range(40)}) == 2


def test_fault_slot_varies_across_tasks_with_same_seed():
    from collections import defaultdict
    from stfr.svc import TOOLS
    from stfr.tasks import fault_slot, generate
    by = defaultdict(set)
    for i in range(80):
        t = generate(i)
        n = sum(TOOLS[s["tool"]].kind == "write" for s in t.ref_steps)  # every write is eligible for SILENT_DROP
        if n >= 2:
            by[n].add(fault_slot(t, Fault.SILENT_DROP, 0))
    assert by and any(len(v) > 1 for v in by.values())


def test_empty_reply_ends_episode_without_crash():
    from tests.test_agent import ep
    e, _ = ep(["", "", ""])
    assert e.status != "done" and e.status == "parse_failure"
