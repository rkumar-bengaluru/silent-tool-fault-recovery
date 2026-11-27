import random

from stfr.agent import Episode
from stfr.grade import grade
from stfr.stats import cluster_bootstrap_diff, holm, mcnemar_exact, mde_paired, paired_counts
from stfr.tasks import Task

INIT = {"orders": {"o1": {"customer": "ann", "item": "pen", "qty": 2, "status": "pending"}},
        "inventory": {"i1": {"sku": "pen", "stock": 10, "price": 2}}}
NEW = {"customer": "bob", "item": "pen", "qty": 1, "status": "pending"}
TASK = Task("t", "p", INIT, {"orders/o1/status": "shipped"}, [["orders", "o1"]],
            [{"tool": "update_field", "args": {}}], 1, 0)


def state(**ch):
    s = {c: {i: dict(r) for i, r in rs.items()} for c, rs in INIT.items()}
    for path, v in ch.items():
        c, i, f = path.split("__")
        s[c][i][f] = v
    return s


def step(k, tool="update_field"):
    return {"k": k, "tool": tool, "args": {}, "error": None}


def mkep(final, status="done", steps=(1,), flags=(), fault="clean", applied=False, call=None, rb=0):
    return Episode("t", fault, "never", "recover", 0, status, [step(k) for k in steps], list(flags), final,
                   0, 0, 0.0, rb, applied, call)


def test_success_requires_goal_and_no_side_effects():
    assert grade(TASK, mkep(state(orders__o1__status="shipped")))["success"] is True
    g = grade(TASK, mkep(state(orders__o1__status="shipped", inventory__i1__stock=9)))
    assert (g["success"], g["unintended_changes"], g["wrong_commit"]) == (False, 1, True)
    # a change to another field of an allowed record is unintended but not outside `allowed`
    g = grade(TASK, mkep(state(orders__o1__status="shipped", orders__o1__qty=3)))
    assert (g["success"], g["unintended_changes"]) == (False, 1)


def test_wrong_commit_when_done_but_goal_unmet():
    g = grade(TASK, mkep(state()))
    assert (g["success"], g["wrong_commit"], g["abstained"]) == (False, True, False)
    g = grade(TASK, mkep(state(), status="abstained"))
    assert (g["success"], g["wrong_commit"], g["abstained"]) == (False, False, True)


def test_delayed_readback_duplicate_counted():
    final = state(orders__o1__status="shipped")
    final["orders"]["o2"], final["orders"]["o3"] = dict(NEW), dict(NEW)
    t = Task("t", "p", INIT, {"orders/o1/status": "shipped", **{f"orders/o2/{k}": v for k, v in NEW.items()}},
             [["orders", "o1"], ["orders", "o2"]],
             [{"tool": "create_record", "args": {"collection": "orders"}}, {"tool": "update_field", "args": {}}], 2, 0)
    ep = mkep(final, steps=(1, 2, 3, 4), flags=(2,), fault="delayed", applied=True, call=0)
    g = grade(t, ep)
    assert (g["duplicates"], g["success"], g["false_flags"]) == (1, False, 1)
    assert g["extra_calls"] == 2


def test_never_applied_episode_flags_all_false():
    g = grade(TASK, mkep(state(orders__o1__status="shipped"), steps=(1, 2), flags=(1, 2),
                         fault="silent_drop", applied=False, rb=3))
    assert (g["false_flags"], g["fault_applied"], g["readback_calls"]) == (2, False, 3)


def test_faulted_step_flag_is_true_flag():
    ep = mkep(state(), steps=(1, 2, 3), flags=(2, 3), fault="silent_drop", applied=True, call=1)
    assert grade(TASK, ep)["false_flags"] == 1
    assert grade(TASK, mkep(state(), flags=(1,), fault="clean"))["false_flags"] == 1


def test_mcnemar_known_values():
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-12
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert paired_counts([False, False, True, True], [True, False, False, True]) == (1, 1)


def test_bootstrap_ci_contains_true_diff_on_synthetic():
    rng = random.Random(1)
    rows = []
    for i in range(200):
        rows.append({"task_id": f"t{i}", "arm": "base", "success": rng.random() < 0.4})
        rows.append({"task_id": f"t{i}", "arm": "arm", "success": rng.random() < 0.7})
    d, lo, hi = cluster_bootstrap_diff(rows, "arm", "base", n=500)
    assert lo <= 0.3 <= hi and lo < d < hi
    assert 0 < mde_paired(100, 0.3) < 0.3


def test_holm_orders_and_caps_at_one():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.9})
    assert out == {"a": 0.04, "b": 0.09, "c": 0.09, "d": 0.9}
    assert holm({"a": 0.6, "b": 0.7})["b"] == 1.0
