import json
from pathlib import Path

import pytest

from stfr import paper_assets, run

RESULTS = Path(__file__).resolve().parents[1] / "results" / "results.json"
M8 = "qwen3:8b-q4_K_M"
HASH = "379f948f935a96c63c23d3401e4b42ba6a46cca2732f8e0615d4e01e0ca4cf5a"


def test_frozen_confirmatory_values():
    tests = json.loads(RESULTS.read_text())["tests"]
    assert paper_assets.tests_hash(tests) == HASH
    want = {"on_readback_vs_on_contract_silent_write": (0.133333, 9.999e-05, 2.9997e-04, [0.0972222, 0.173611]),
            "on_contract_vs_never_silent_read": (0.0583333, 0.0638936, 0.0638936, [0.00833333, 0.116667]),
            "readback_contract_vs_never_all": (0.0436508, 6.9993e-04, 1.39986e-03, [0.0226190, 0.0646825])}
    for t in tests:
        d, p, h, ci = want[t["name"]]
        assert t["diff"] == pytest.approx(d, rel=1e-5) and t["p"] == pytest.approx(p, rel=1e-5)
        assert t["holm_p"] == pytest.approx(h, rel=1e-5) and t["ci"] == pytest.approx(ci, rel=1e-5)


def row(task="t0", fault="clean", trig="never", ok=True, status="done", kind=None, applied=False, steps=(), seed=0):
    return {"task_id": task, "fault": fault, "seed": seed, "trigger": trig, "response": "recover", "status": status,
            "tokens_in": 1, "tokens_out": 1, "steps": list(steps),
            "grade": {"success": ok, "wrong_commit": False, "abstained": False, "false_flags": 0, "extra_calls": 0,
                      "readback_calls": 0, "duplicates": 0, "fault_applied": applied, "fault_read_kind": kind}}


def mk(tmp, name, rows, model=M8, split="dev", mode="final", complete=True):
    d = tmp / "runs" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model": model, "seeds": [0], "split": split}))
    (d / "summary.json").write_text(json.dumps({"run_id": name, "mode": mode, "complete": complete, "cost_usd": 0.0}))
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def agg(tmp, dirs):
    run.aggregate(dirs, tmp / "o.json")
    return json.loads((tmp / "o.json").read_text())


def test_exploratory_excludes_incomplete_run(tmp_path):
    a = mk(tmp_path, "a", [row()])
    b = mk(tmp_path, "b", [row()], model="qwen3:4b", complete=False)
    r = agg(tmp_path, [a, b])
    assert [e["run_id"] for e in r["exploratory"]] == ["a"]
    assert r["exploratory"][0]["episodes"] == 1 and r["exploratory"][0]["split"] == "dev"
    assert r["exploratory_excluded"][0]["run_id"] == "b" and r["exploratory_excluded"][0]["reason"]


def test_exploratory_duplicate_model_split_raises(tmp_path):
    with pytest.raises(ValueError):
        agg(tmp_path, [mk(tmp_path, "a", [row()]), mk(tmp_path, "b", [row(task="t9")])])


def test_silent_read_by_kind_counts(tmp_path):
    fl = [{"tool": "get", "flagged": True}]
    rows = [row(fault="silent_read", kind="plausible", ok=True, steps=fl),
            row(fault="silent_read", kind="plausible", ok=False, task="t1"),
            row(fault="silent_read", kind="implausible", ok=True, task="t2"),
            row(fault="loud", task="t3")]
    e = agg(tmp_path, [mk(tmp_path, "a", rows)])["exploratory"][0]["silent_read_by_kind"]["never/recover"]
    assert e["plausible"] == {"n": 2, "success": 1, "flagged": 1} and e["implausible"] == {"n": 1, "success": 1, "flagged": 0}
    assert agg(tmp_path, [tmp_path / "runs" / "a"])["exploratory"][0]["silent_read_by_kind"]["on_contract/recover"]["plausible"]["n"] == 0


def test_fault_applied_rate_denominator_includes_parse_failures(tmp_path):
    rows = [row(fault="loud", applied=True), row(fault="loud", applied=False, status="parse_failure", task="t1")]
    e = agg(tmp_path, [mk(tmp_path, "a", rows)])["exploratory"][0]
    assert e["fault_applied_rate"]["never/recover"]["loud"] == 0.5
    assert "clean" not in e["fault_applied_rate"]["never/recover"]
    assert e["status_counts"] == {"done": 1, "parse_failure": 1}


def test_flag_rate_counts_only_service_steps(tmp_path):
    st = [{"tool": "get", "flagged": True}, {"tool": "get", "flagged": False}, {"tool": None, "flagged": True}]
    rows = [row(steps=st), row(fault="delayed", steps=st, task="t1"), row(fault="loud", steps=st, task="t2")]
    e = agg(tmp_path, [mk(tmp_path, "a", rows)])["exploratory"][0]
    assert e["flag_rate_clean_delayed"]["never/recover"] == 0.5  # 2 flagged / 4 service steps; loud excluded


FMT = [("prop", 0, "0.000"), ("prop", 1, "1.000"), ("prop", 0.00049, "0.000"), ("prop", 0.9999, "1.000"), ("prop", -0.0769, "$-$0.077"),
       ("p", 9.999e-05, "$<$0.001"), ("p", 0.0639, "0.064"), ("p", 1.0, "1.000"), ("count", 6840, "6{,}840"), ("count", 12, "12"),
       ("mean", 13442.85, "13442.9"), ("mean", 123.456, "123.5"), ("usd", 0.9126, "\\$0.91"),
       ("text", "qwen3:8b-q4_K_M", "qwen3:8b-q4\\_K\\_M")]


def test_fmt_table():
    for kind, x, want in FMT:
        assert paper_assets.fmt(x, kind) == want
    for kind, x in [("prop", 1.2), ("prop", -1.01), ("p", 1.1), ("p", -0.1), ("count", 12.5)]:
        with pytest.raises(ValueError):
            paper_assets.fmt(x, kind)


def _res():
    return json.loads(RESULTS.read_text())


def test_macro_names_pattern_and_unique():
    import re
    names = [s["name"] for s in paper_assets.build_specs(_res())]
    assert len(names) == len(set(names)) > 400 and all(re.fullmatch(r"R[A-Za-z]+", n) for n in names)
    assert {"RPOneDiff", "RPThreeDelayedHarmTotal", "RTasksPerSplit", "RHaikuCostUsd"} <= set(names)


def test_cell_size_wrong_commit_and_abstain_macros(tmp_path):
    nums = paper_assets.write_assets(_res(), tmp_path)
    assert nums["RQwenEightDropReadbackN"]["formatted"] == "120" and nums["RHaikuPartialReadbackN"]["formatted"] == "20"
    assert nums["RQwenEightDropReadbackWrongCommit"]["kind"] == "prop"
    assert nums["RQwenEightDropReadbackAbstainAbstainRate"]["formatted"] == "0.917"
    assert nums["RQwenEightDropReadbackAbstainWrongCommit"]["formatted"] == "0.050"


def test_assets_written_with_lf(tmp_path):
    paper_assets.write_assets(_res(), tmp_path)
    assert all(b"\r\n" not in f.read_bytes() for f in tmp_path.iterdir())


def test_duplicate_macro_name_raises(monkeypatch):
    s = paper_assets.spec(["p_one"], "x", "p")
    monkeypatch.setattr(paper_assets, "build_specs", lambda r: [s, s])
    with pytest.raises(ValueError, match="duplicate"):
        paper_assets.build_numbers({})


def test_numbers_json_value_matches_source_path(tmp_path):
    res = _res()
    nums = paper_assets.write_assets(res, tmp_path)
    assert json.loads((tmp_path / "numbers.json").read_text()).keys() == nums.keys()
    for n, d in nums.items():
        assert paper_assets.resolve(res, d["source_path"]) == d["value"], n
    assert nums["RTasksPerSplit"]["formatted"] == "60" and nums["RPOneDiff"]["source_path"] == "tests[0].diff"
    assert not any(n.endswith(("HolmP", "PMcnemar")) for n in nums if n.startswith("RExp"))
    assert nums["RPOneHolmP"]["formatted"] == "$<$0.001" and nums["RPTwoHolmP"]["formatted"] == "0.064"
    assert "\\newcommand{\\RPOneDiff}{0.133}" in (tmp_path / "numbers.tex").read_text()


def test_success_table_bolds_row_max_and_ties(tmp_path):
    res = _res()
    paper_assets.write_assets(res, tmp_path)
    rows = (tmp_path / "tab_success_QwenEight.tex").read_text().splitlines()
    cells = {(c["fault"], c["arm"]): c["success"] for c in res["cells"] if c["model"] == M8}
    n_bold = []
    for f, line in zip(paper_assets.FAULTS, [r for r in rows if r.startswith(tuple(w.capitalize() for w in ["clean", "loud", "drop", "wrong", "partial", "read", "delayed"]))]):
        vals = [paper_assets.fmt(cells[(f, a)], "prop") for a in paper_assets.ARM_ORDER]
        top = max(vals, key=float)
        assert line.count("\\textbf") == vals.count(top)
        n_bold.append(line.count("\\textbf"))
    assert len(n_bold) == 7 and max(n_bold) > 1  # real data has at least one tie (loud/clean all-success rows)


def test_fig_dat_header_order_and_values(tmp_path):
    paper_assets.write_assets(_res(), tmp_path)
    lines = (tmp_path / "fig_success.dat").read_text().splitlines()
    assert lines[0] == "model arm drop wrong partial read" and len(lines) == 22
    assert [l.split()[:2] for l in lines[1:3]] == [["QwenFour", "Never"], ["QwenFour", "OnError"]]
    assert lines[8].split()[:2] == ["QwenEight", "Never"] and lines[-1].split()[:2] == ["Haiku", "Always"]
    c = next(c for c in _res()["cells"] if (c["model"], c["fault"], c["arm"]) == (M8, "silent_drop", "never/recover"))
    assert lines[8].split()[2] == f"{c['success']:.3f}"


def test_fig_dat_missing_cell_raises():
    r = _res()
    r["cells"] = [c for c in r["cells"] if not (c["model"] == M8 and c["fault"] == "silent_read" and c["arm"] == "always/recover")]
    with pytest.raises(ValueError, match="missing"):
        paper_assets._fig_dat(r)


def test_assets_cli_refuses_wrong_hash(tmp_path, capsys):
    r = _res()
    r["tests"][0]["diff"] += 0.001
    (tmp_path / "results.json").write_text(json.dumps(r))
    assert run.main(["assets", "--results", str(tmp_path)]) == 2
    assert "frozen hash" in capsys.readouterr().err


def test_primary_test_index_mapping():
    assert [t["name"] for t in _res()["tests"]] == ["on_readback_vs_on_contract_silent_write", "on_contract_vs_never_silent_read",
                                                    "readback_contract_vs_never_all"]


def test_no_bare_lt_in_generated(tmp_path):
    nums = paper_assets.write_assets(_res(), tmp_path)
    assert not any("<" in d["formatted"].replace("$<$", "") for d in nums.values())
    assert all("<" not in f.read_text().replace("$<$", "") for f in tmp_path.glob("*.tex"))
