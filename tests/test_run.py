import json
from pathlib import Path

import pytest

from stfr import run
from stfr.triggers import TRIGGERS
from stfr.llm import BudgetExceeded, ModelUnavailable, Reply, cache_key

DONE = json.dumps({"done": True, "summary": "ok"})
M8 = "qwen3:8b-q4_K_M"


class Chat:
    """Caching fake: identical request -> cached reply (cost 0), like the real caches."""
    def __init__(self, fail_after=None, seen=None):
        self.n, self.fail_after, self.seen = 0, fail_after, seen if seen is not None else {}

    def chat(self, messages, *, seed, temperature=0.0):
        if self.fail_after is not None and self.n >= self.fail_after:
            raise BudgetExceeded("stop")
        self.n += 1
        k = json.dumps([messages, seed])
        hit = k in self.seen
        self.seen[k] = 1
        return Reply(DONE, 10, 5, 0.0 if hit else 0.01, cached=hit)


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)


def cfg(tmp, name="dev.json", **kw):
    d = {"model": "qwen3:4b-q4_K_M", "provider": "ollama", "split": "dev", "task_limit": 2, "faults": ["clean", "silent_drop"],
         "arms": [{"trigger": "never", "response": "recover"}, {"trigger": "on_readback", "response": "recover"}],
         "seeds": [0], "threshold": 0.7, "max_steps": 5, "est_usd_per_episode": 0.0, "frozen": False, **kw}
    p = Path(tmp) / "stfr" / "configs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))
    return p


def go(tmp, mode, p, chat=None, factory=None, extra=(), **kw):
    chat = chat or Chat()
    return run.main([mode, "--config", str(p), *extra], chat_factory=factory or (lambda *a, **k: chat), **kw)


def eps(tmp):
    (d,) = (Path(tmp) / "results" / "runs").iterdir()
    return d, [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines()]


def test_matched_design_same_slot_across_arms(tmp_path, monkeypatch):
    seen = []
    real = run.run_episode
    monkeypatch.setattr(run, "run_episode", lambda task, chat, **kw: seen.append(
        (task.id, kw["fault"].value, kw["seed"], kw["slot"], kw["trigger"])) or real(task, chat, **kw))
    assert go(tmp_path, "dev", cfg(tmp_path)) == 0
    slots = {}
    for t, f, s, slot, trig in seen:
        slots.setdefault((t, f, s), set()).add(slot)
    assert len(seen) == 2 * 2 * 2 and all(len(v) == 1 for v in slots.values())
    _, rows = eps(tmp_path)
    assert len({(r["task_id"], r["fault"], r["seed"], r["trigger"], r["response"]) for r in rows}) == 8


def test_dev_mode_refuses_sealed_split(tmp_path):
    assert go(tmp_path, "dev", cfg(tmp_path, split="testA")) == 2
    assert not (tmp_path / "results" / "runs").exists()


def final_cfg(tmp, **kw):
    cfg(tmp, "dev.json", seeds=[0, 1])
    return cfg(tmp, "frozen_x.json", **{"split": "testA", "seeds": [100], "frozen": True, **kw})


def test_final_refuses_unfrozen_or_dirty_or_repeat(tmp_path):
    clean = lambda: ""
    assert go(tmp_path, "final", final_cfg(tmp_path, frozen=False), git_status=clean) == 2
    assert go(tmp_path, "final", final_cfg(tmp_path), git_status=lambda: " M stfr/agent.py") == 2
    assert go(tmp_path, "final", cfg(tmp_path, "other.json", split="testA", seeds=[100], frozen=True), git_status=clean) == 2
    lock = tmp_path / "results" / "final_testA_qwen3_4b-q4_K_M.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("x")
    assert go(tmp_path, "final", final_cfg(tmp_path), git_status=clean) == 2
    assert not (tmp_path / "results" / "runs").exists()


def test_final_refuses_dev_seeds(tmp_path):
    assert go(tmp_path, "final", final_cfg(tmp_path, seeds=[1, 100]), git_status=lambda: "") == 2
    assert not (tmp_path / "results" / "final_testA_qwen3_4b-q4_K_M.lock").exists()


def test_final_reserves_and_refuses_when_budget_short(tmp_path):
    assert go(tmp_path, "final", final_cfg(tmp_path, est_usd_per_episode=1000.0), git_status=lambda: "") == 3
    assert not (tmp_path / "results" / "runs").exists()
    assert not list((tmp_path / "results").glob("*.lock"))


def test_final_happy_path_writes_lock(tmp_path):
    assert go(tmp_path, "final", final_cfg(tmp_path, task_limit=1), git_status=lambda: "") == 0
    assert (tmp_path / "results" / "final_testA_qwen3_4b-q4_K_M.lock").exists()


def test_budget_stop_marks_incomplete_and_keeps_episodes(tmp_path):
    assert go(tmp_path, "dev", cfg(tmp_path), Chat(fail_after=3)) == 3
    d, rows = eps(tmp_path)
    assert len(rows) == 3
    assert json.loads((d / "summary.json").read_text())["complete"] is False


class CachedChat:
    """Uses the runner-provided disk Cache; multi-step replies; uncached replies cost money."""
    def __init__(self, cache, log):
        self.cache, self.log = cache, log

    def chat(self, messages, *, seed, temperature=0.0):
        key = cache_key({"m": messages, "seed": seed})
        if hit := self.cache.get(key):
            self.log.append(True)
            return Reply(**hit, cached=True)
        n = sum(m["role"] == "assistant" for m in messages)
        text = DONE if n else json.dumps({"thought": f"s{seed}", "tool": "list_records",
                                          "args": {"collection": "orders", "where": {}}, "confidence": 0.9})
        r = Reply(text, 10 + n, 5, 0.01)
        self.cache.put(key, {"text": r.text, "input_tokens": r.input_tokens, "output_tokens": 5, "cost_usd": 0.01})
        self.log.append(False)
        return r


def test_rerun_from_cache_is_identical_and_free(tmp_path):
    p, logs = cfg(tmp_path), []
    fac = lambda c, *, cache, **k: CachedChat(cache, logs)  # a fresh fake per run: hits must come from disk
    assert go(tmp_path, "dev", p, factory=fac) == 0
    assert not all(logs) and logs  # first run: some replies are uncached (arms share prefixes)
    cost = (tmp_path / "results" / "cost.json").read_text()
    assert json.loads(cost)[0]["usd"] > 0
    first = next((tmp_path / "results" / "runs").iterdir())
    logs.clear()
    assert go(tmp_path, "dev", p, factory=fac) == 0
    second = [d for d in (tmp_path / "results" / "runs").iterdir() if d != first][0]
    assert all(logs) and logs
    assert (second / "episodes.jsonl").read_bytes() == (first / "episodes.jsonl").read_bytes()
    assert max(len(json.loads(l)["steps"]) for l in (first / "episodes.jsonl").read_text().splitlines()) > 1
    assert (tmp_path / "results" / "cost.json").read_text() == cost
    assert json.loads((second / "summary.json").read_text())["cost_usd"] == 0


def test_ollama_down_exits_before_any_episode(tmp_path):
    def down(*a, **k):
        raise ModelUnavailable("Ollama not reachable at http://localhost:11434")
    p = cfg(tmp_path)
    assert run.main(["dev", "--config", str(p)], chat_factory=down) == 4
    assert not (tmp_path / "results" / "runs").exists() and not (tmp_path / "results" / "results.json").exists()


def test_aggregate_copies_config_and_computes_cells(tmp_path):
    assert go(tmp_path, "dev", cfg(tmp_path)) == 0
    d, rows = eps(tmp_path)
    out = tmp_path / "results.json"
    run.aggregate([d], out)
    r = json.loads(out.read_text())
    assert r["runs"][0]["config"]["seeds"] == [0] and r["runs"][0]["complete"] is True
    c = next(c for c in r["cells"] if c["fault"] == "clean" and c["arm"] == "never/recover")
    assert c["n"] == 2 and c["model"] == "qwen3:4b-q4_K_M" and "success" in c and "duplicates" in c
    assert r["tests"] == []  # primary model absent


def test_primary_tests_pair_and_holm(tmp_path):
    d = tmp_path / "runs" / "r1"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model": M8, "seeds": [0], "split": "dev"}))
    (d / "summary.json").write_text(json.dumps({"run_id": "r1", "mode": "final", "complete": True, "cost_usd": 0.0}))
    g = lambda ok: {"success": ok, "wrong_commit": False, "abstained": False, "false_flags": 0,
                    "extra_calls": 0, "readback_calls": 0, "duplicates": 0}
    rows = []
    for t in range(6):
        for f in run.ALL_FAULTS:
            for arm, ok in (("never", False), ("on_readback", True), ("on_contract", False), ("readback_contract", True)):
                rows.append({"task_id": f"t{t}", "fault": f, "seed": 0, "trigger": arm, "response": "recover",
                             "tokens_in": 1, "tokens_out": 1, "grade": g(ok)})
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    run.aggregate([d], tmp_path / "o.json")
    tests = json.loads((tmp_path / "o.json").read_text())["tests"]
    assert len(tests) == 3
    t1 = tests[0]
    assert t1["n_pairs"] == 6 * 3 and t1["rescue"] == 18 and t1["harm"] == 0 and t1["holm_p"] >= t1["p"]
    assert tests[2]["n_pairs"] == 6 * 7


CLEAN = lambda: ""


def test_results_flag_refused_outside_aggregate(tmp_path):
    p = final_cfg(tmp_path)
    for mode in ("final", "dev", "pilot"):
        assert go(tmp_path, mode, p, extra=["--results", str(tmp_path / "elsewhere")], git_status=CLEAN) == 2
    assert not (tmp_path / "elsewhere").exists()


def test_final_config_must_be_in_repo_configs_dir(tmp_path):
    other = tmp_path / "elsewhere"
    p = final_cfg(other)  # frozen config outside ROOT/stfr/configs, no dev configs next to it with seeds [0, 1]
    assert go(tmp_path, "final", cfg(other, "frozen_y.json", split="testA", seeds=[0, 1], frozen=True), git_status=CLEAN) == 2
    assert go(tmp_path, "final", p, git_status=CLEAN) == 2


def test_final_dev_seeds_read_from_repo_configs_even_if_config_elsewhere(tmp_path):
    cfg(tmp_path, "dev.json", seeds=[0, 1])
    q = cfg(tmp_path / "x", "frozen_y.json", split="testA", seeds=[0, 1], frozen=True)
    assert go(tmp_path, "final", q, git_status=CLEAN) == 2


def test_final_claude_needs_positive_estimate(tmp_path):
    assert go(tmp_path, "final", final_cfg(tmp_path, provider="claude", est_usd_per_episode=0.0), git_status=CLEAN) == 2
    assert not list((tmp_path / "results").glob("*.lock"))


def test_failed_git_status_counts_as_dirty(tmp_path, monkeypatch):
    class R:
        returncode, stdout = 128, ""
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: R)
    assert go(tmp_path, "final", final_cfg(tmp_path)) == 2
    def boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(run.subprocess, "run", boom)
    assert go(tmp_path, "final", final_cfg(tmp_path)) == 2


def spy(monkeypatch):
    calls = []
    real = run.load_split
    monkeypatch.setattr(run, "load_split", lambda name, allow_sealed=False: calls.append((name, allow_sealed)) or real(name, allow_sealed=allow_sealed))
    return calls


def test_refused_reservation_never_loads_sealed_split(tmp_path, monkeypatch):
    calls = spy(monkeypatch)
    assert go(tmp_path, "final", final_cfg(tmp_path, est_usd_per_episode=1000.0), git_status=CLEAN) == 3
    assert not calls and not list((tmp_path / "results").glob("*.lock"))


def test_final_refuses_dev_split(tmp_path):
    assert go(tmp_path, "final", final_cfg(tmp_path, split="dev"), git_status=CLEAN) == 2


def test_pilot_and_dev_never_load_sealed(tmp_path, monkeypatch):
    calls = spy(monkeypatch)
    proj = cfg(tmp_path, "proj.json", split="testA", task_limit=None)
    assert go(tmp_path, "pilot", cfg(tmp_path), extra=["--project-config", str(proj)]) == 0
    assert go(tmp_path, "dev", cfg(tmp_path)) == 0
    assert calls and not any(sealed for _, sealed in calls)


def test_configs_name_only_known_triggers_and_responses():
    for f in (Path(run.__file__).parent / "configs").glob("*.json"):
        for arm in json.loads(f.read_text())["arms"]:
            assert arm["trigger"] in TRIGGERS and arm["response"] in ("recover", "abstain"), (f, arm)


def test_unknown_trigger_exits_2_before_any_episode(tmp_path):
    assert go(tmp_path, "dev", cfg(tmp_path, arms=[{"trigger": "on_low_conf", "response": "recover"}])) == 2
    assert not (tmp_path / "results" / "runs").exists()


def test_summary_written_on_unexpected_error(tmp_path):
    class Boom(Chat):
        def chat(self, *a, **k):
            raise RuntimeError("kaput")
    with pytest.raises(RuntimeError):
        go(tmp_path, "dev", cfg(tmp_path), Boom())
    (d,) = (tmp_path / "results" / "runs").iterdir()
    s = json.loads((d / "summary.json").read_text())
    assert s["complete"] is False and "kaput" in s["error"]


def test_mid_run_model_unavailable_exits_4_with_summary(tmp_path):
    class Gone(Chat):
        def chat(self, *a, **k):
            raise ModelUnavailable("gone")
    assert go(tmp_path, "dev", cfg(tmp_path), Gone()) == 4
    (d,) = (tmp_path / "results" / "runs").iterdir()
    assert json.loads((d / "summary.json").read_text())["complete"] is False


def test_duplicate_episodes_raise_value_error():
    r = {"model": run.PRIMARY_MODEL, "task_id": "t", "fault": "clean", "seed": 0, "arm": "never/recover", "grade": {"success": True}}
    with pytest.raises(ValueError):
        run._tests([r, dict(r)])


def test_final_claude_stops_mid_run_when_spend_passes_reservation(tmp_path):
    from types import SimpleNamespace

    class Client:  # each call really costs $0.001, far above the $0.0001 reservation
        messages = None
        def __init__(self):
            self.messages = self
        def create(self, **kw):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=DONE)],
                                   usage=SimpleNamespace(input_tokens=10_000, output_tokens=0))

    from stfr.llm import ClaudeChat
    fac = lambda c, *, cache, tracker, run_id, final: ClaudeChat(c["model"], cache=cache, tracker=tracker, run_id=run_id,
                                                                final=final, client=Client())
    p = final_cfg(tmp_path, provider="claude", model="claude-haiku-5-5", est_usd_per_episode=0.00001)
    assert go(tmp_path, "final", p, factory=fac, git_status=lambda: "") == 3
    (d,) = (tmp_path / "results" / "runs").iterdir()
    assert json.loads((d / "summary.json").read_text())["complete"] is False
    assert list((tmp_path / "results").glob("final_testA_*.lock"))
