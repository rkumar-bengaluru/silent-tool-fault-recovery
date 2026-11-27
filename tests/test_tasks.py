import json

import pytest

from stfr import tasks
from stfr.faults import Fault, _eligible
from stfr.svc import TOOLS, Service
from stfr.tasks import SealBroken, SealedSplit, fault_slot, generate

DEV = range(30)
ALL = list(range(30)) + list(range(10000, 10060))


def test_generate_is_deterministic():
    assert generate(7) == generate(7)
    assert generate(7) != generate(8)


@pytest.mark.parametrize("seed", ALL)
def test_every_task_is_solvable_by_its_reference(seed):
    t = generate(seed)
    s = Service(t.initial)
    for st in t.ref_steps:
        s.call(st["tool"], st["args"])
    after = s.snapshot()
    for key, v in t.goal.items():
        c, i, f = key.split("/")
        assert after[c][i][f] == v, key
    allowed = {tuple(a) for a in t.allowed}
    for c in set(after) | set(t.initial):
        for i in set(after[c]) | set(t.initial[c]):
            if after[c].get(i) != t.initial[c].get(i):
                assert (c, i) in allowed, (c, i)
    assert 3 <= len(t.ref_steps) <= 6


def test_every_task_has_two_writes():
    for seed in ALL:
        t = generate(seed)
        assert t.n_writes >= 2
        assert t.n_writes == sum(TOOLS[s["tool"]].kind == "write" for s in t.ref_steps)
        assert t.n_reads == len(t.ref_steps) - t.n_writes


def _ref_result(t, kind, idx, fault):
    s, n = Service(t.initial), 0
    for st in t.ref_steps:
        res = s.call(st["tool"], st["args"])
        if TOOLS[st["tool"]].kind == kind:
            if n == idx:
                return st["tool"], res
            n += 1


def test_fault_slot_points_at_eligible_ref_call():
    for seed in DEV:
        t = generate(seed)
        for f in Fault:
            slot = fault_slot(t, f, seed)
            if f == Fault.CLEAN:
                assert slot == 0
            elif slot is not None:
                tool, res = _ref_result(t, "read" if f == Fault.SILENT_READ else "write", slot, f)
                assert _eligible(f, tool, res)
            else:
                assert f == Fault.SILENT_PARTIAL or f == Fault.SILENT_READ
            assert fault_slot(t, f, seed) == slot


def test_partial_has_coverage():
    n = sum(fault_slot(generate(s), Fault.SILENT_PARTIAL, s) is not None for s in DEV)
    assert n >= len(DEV) / 3


def test_prompts_do_not_mention_faults():
    for s in DEV:
        assert "fault" not in generate(s).prompt.lower()


@pytest.fixture
def splits(tmp_path, monkeypatch):
    tasks.write_splits(tmp_path)
    monkeypatch.setattr(tasks, "DATA", tmp_path)
    return tmp_path


def test_sealed_split_refused_without_flag(splits):
    assert len(tasks.load_split("dev")) == 30
    with pytest.raises(SealedSplit):
        tasks.load_split("testA")
    assert len(tasks.load_split("testA", allow_sealed=True)) == 60
    assert len(tasks.load_split("testB", allow_sealed=True)) == 60


def test_tampered_split_raises_seal_broken(splits):
    p = splits / "testA.jsonl"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(SealBroken):
        tasks.load_split("testA", allow_sealed=True)


def test_splits_disjoint(splits):
    key = lambda t: json.dumps([t.prompt, t.initial], sort_keys=True)  # noqa: E731
    dev = {key(t) for t in tasks.load_split("dev")}
    for n in ("testA", "testB"):
        assert not dev & {key(t) for t in tasks.load_split(n, allow_sealed=True)}
