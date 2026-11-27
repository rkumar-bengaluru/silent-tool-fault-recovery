"""Turn the frozen results.json into paper/generated/: \\R macros, tables, pgfplots data. The paper types no result by hand."""
from __future__ import annotations

import hashlib
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path



def escape_latex(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", s)

FROZEN_HASH = "379f948f935a96c63c23d3401e4b42ba6a46cca2732f8e0615d4e01e0ca4cf5a"
TESTS = ["POne", "PTwo", "PThree"]
MODELS = {"qwen3:8b-q4_K_M": "QwenEight", "qwen3:4b-q4_K_M": "QwenFour", "claude-haiku-5-5": "Haiku"}
ARMS = {"never/recover": "Never", "always/recover": "Always", "on_error/recover": "OnError", "on_uncertainty/recover": "OnUnc",
        "on_contract/recover": "Contract", "on_readback/recover": "Readback", "readback_contract/recover": "Both",
        "on_contract/abstain": "ContractAbstain", "on_readback/abstain": "ReadbackAbstain"}
FAULTS = {"clean": "Clean", "loud": "Loud", "silent_drop": "Drop", "silent_wrong": "Wrong", "silent_partial": "Partial",
          "silent_read": "Read", "delayed": "Delayed"}
FIELDS = {"diff": "Diff", "p": "P", "holm_p": "HolmP", "p_mcnemar": "PMcnemar", "ci_lo": "CiLo", "ci_hi": "CiHi", "rescue": "Rescue",
          "harm": "Harm", "n_pairs": "NPairs", "success": "Success", "delayed_harm": "DelayedHarmTotal",
          "n": "N", "wrong_commit": "WrongCommit", "abstain": "AbstainRate"}
TAGS = {**MODELS, **ARMS, **FAULTS, **FIELDS}
ARM_ORDER = list(ARMS)
FIG_MODELS = ["qwen3:4b-q4_K_M", "qwen3:8b-q4_K_M", "claude-haiku-5-5"]
FIG_ARMS = ["never", "on_error", "on_uncertainty", "on_contract", "on_readback", "readback_contract", "always"]
FIG_FAULTS = ["silent_drop", "silent_wrong", "silent_partial", "silent_read"]
ORD = ["first", "second", "third", "fourth"]


def tests_hash(tests) -> str:
    return hashlib.sha256(json.dumps(tests, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dec(x, places: str) -> str:
    return str(Decimal(repr(float(x))).quantize(Decimal(places), rounding=ROUND_HALF_UP))


def fmt(x, kind: str) -> str:
    if kind == "text":
        return escape_latex(str(x))
    if kind == "count":
        if float(x) != int(x):
            raise ValueError(f"count must be an integer, got {x!r}")
        return f"{int(x):,}".replace(",", "{,}")
    if kind == "mean":
        return _dec(x, "0.1")
    if kind == "usd":
        return "\\$" + _dec(x, "0.01")
    if kind == "p":
        if not 0 <= x <= 1:
            raise ValueError(f"p out of [0, 1]: {x!r}")
        return "$<$0.001" if x < 0.001 else _dec(x, "0.001")
    if kind == "prop":
        if not -1 <= x <= 1:
            raise ValueError(f"prop out of [-1, 1]: {x!r}")
        s = _dec(abs(x), "0.001")
        return "$-$" + s if x < 0 and s != "0.000" else s
    raise ValueError(f"unknown kind {kind!r}")


def tag(p) -> str:
    if isinstance(p, int):
        return TESTS[p]
    return TAGS.get(p) or "".join(w.capitalize() for w in re.findall(r"[A-Za-z]+", p))


def spec(parts, source_path: str, kind: str) -> dict:
    return {"name": "R" + "".join(tag(p) for p in parts), "source_path": source_path, "kind": kind}


def _name(*parts) -> str:
    return "R" + "".join(tag(p) for p in parts)


def _test_specs(parts, path, t, exploratory):
    skip = {"name", "arm", "base", "faults", "by_fault", "ci"} | ({"holm_p", "p_mcnemar"} if exploratory else set())
    kinds = {"p": "p", "holm_p": "p", "p_mcnemar": "p", "diff": "prop"}
    out = [spec(parts + [k], f"{path}.{k}", kinds.get(k, "count")) for k in t if k not in skip]
    out += [spec(parts + [n], f"{path}.ci[{i}]", "prop") for i, n in enumerate(["ci_lo", "ci_hi"])]
    for f, d in t["by_fault"].items():
        out += [spec(parts + [f, k], f"{path}.by_fault.{f}.{k}", "prop" if k == "diff" else "count") for k in d]
    return out


def build_specs(res: dict) -> list[dict]:
    S = []
    for i, t in enumerate(res["tests"]):
        S += _test_specs([i], f"tests[{i}]", t, False)
    for c in res["cells"]:
        if c["in_tests"]:
            cp = f"cells[model={c['model']},fault={c['fault']},arm={c['arm']}]"
            S += [spec([c["model"], c["fault"], c["arm"], f], f"{cp}.{f}", kind)
                  for f, kind in (("success", "prop"), ("n", "count"), ("wrong_commit", "prop"), ("abstain", "prop"))]
    for e in res["exploratory"]:
        rp, m = f"exploratory[run_id={e['run_id']}]", e["model"]
        S.append(spec(["exp", m, "episodes"], f"{rp}.episodes", "count"))
        for i, t in enumerate(e["tests"]):
            S += _test_specs(["exp", m, i], f"{rp}.tests[{i}]", t, True)
        for arm, kinds in e["silent_read_by_kind"].items():
            for k, d in kinds.items():
                S += [spec(["exp", m, "by_kind", arm, k, f], f"{rp}.silent_read_by_kind.{arm}.{k}.{f}", "count") for f in d]
        S += [spec(["exp", m, "status", s], f"{rp}.status_counts.{s}", "count") for s in e["status_counts"]]
        for arm, fr in e["fault_applied_rate"].items():
            S += [spec(["exp", m, "applied", arm, f], f"{rp}.fault_applied_rate.{arm}.{f}", "prop") for f in fr]
        S += [spec(["exp", m, "flag", arm], f"{rp}.flag_rate_clean_delayed.{arm}", "prop") for arm in e["flag_rate_clean_delayed"]]
    for r in res["runs"]:
        rp, c, m = f"runs[run_id={r['run_id']}]", r["config"], r["model"]
        S += [spec([m, "seed", ORD[j]], f"{rp}.config.seeds[{j}]", "count") for j in range(len(c["seeds"]))]
        S += [spec([m, "threshold"], f"{rp}.config.threshold", "prop"), spec([m, "max_steps"], f"{rp}.config.max_steps", "count"),
              spec([m, "model"], f"{rp}.config.model", "text")]
        if c["provider"] == "claude":
            S.append(spec([m, "cost_usd"], f"{rp}.cost_usd", "usd"))
    # tasks per split, derived: clean/never cell n divided by the number of seeds; all models must agree
    per = {r["model"]: f"cells[model={r['model']},fault=clean,arm=never/recover].n / len(runs[run_id={r['run_id']}].config.seeds)"
           for r in res["runs"]}
    vals = {resolve(res, p) for p in per.values()}
    if len(vals) != 1:
        raise ValueError(f"tasks per split differ across models: {vals}")
    S.append(spec(["tasks_per_split"], per[next(iter(per))], "count"))
    return S


_TOK = re.compile(r"\[(.*?)\]|([^.\[\]]+)")


def resolve(res, path: str):
    """Resolve a source_path (`tests[0].ci[1]`, `x[k=v].y`, optional ` / len(path)`) against results.json."""
    if " / len(" in path:
        a, b = path.split(" / len(")
        return resolve(res, a) / len(resolve(res, b[:-1]))
    cur = res
    for idx, key in _TOK.findall(path):
        if key:
            cur = cur[key]
        elif "=" in idx:
            want = [kv.split("=", 1) for kv in idx.split(",")]
            hit = [x for x in cur if all(x[k] == v for k, v in want)]
            if len(hit) != 1:
                raise KeyError(f"{path}: [{idx}] matched {len(hit)}")
            cur = hit[0]
        else:
            cur = cur[int(idx)]
    return cur


def build_numbers(res: dict) -> dict:
    out, specs = {}, build_specs(res)
    seen = set()
    for s in specs:
        if s["name"] in seen:
            raise ValueError(f"duplicate macro name {s['name']}")
        if not re.fullmatch(r"R[A-Za-z]+", s["name"]):
            raise ValueError(f"bad macro name {s['name']}")
        seen.add(s["name"])
    for s in specs:
        v = resolve(res, s["source_path"])
        out[s["name"]] = {"value": v, "formatted": fmt(v, s["kind"]), "source_path": s["source_path"], "kind": s["kind"]}
    return dict(sorted(out.items()))


def _success_table(res, model, nums):
    cells = {(c["fault"], c["arm"]): c["success"] for c in res["cells"] if c["model"] == model and c["in_tests"]}
    hdr = ["Never", "Always", "OnErr", "OnUnc", "Contr.", "Readb.", "Both", "Contr.-Ab.", "Readb.-Ab."]
    rows = ["\\begin{tabular}{@{}l" + "r" * 9 + "@{}}", "\\toprule", "Fault & " + " & ".join(hdr) + " \\\\", "\\midrule"]
    for f, ft in FAULTS.items():
        vals = [cells[(f, a)] for a in ARM_ORDER]
        best = fmt(max(vals), "prop")
        cs = []
        for a, v in zip(ARM_ORDER, vals):
            m = "\\" + _name(model, f, a, "success")
            assert m[1:] in nums
            cs.append(f"\\textbf{{{m}}}" if fmt(v, "prop") == best else m)
        rows.append(f"{ft} & " + " & ".join(cs) + " \\\\")
    return "\n".join(rows + ["\\bottomrule", "\\end{tabular}", ""])


def _test_row(label, parts, n_exploratory=False):
    m = lambda *f: "\\" + _name(*parts, *f)
    return (f"{label} & {m('diff')} & [{m('ci_lo')}, {m('ci_hi')}] & {m('p')} & "
            + ("" if n_exploratory else f"{m('holm_p')} & ") + f"{m('rescue')}/{m('harm')} \\\\")


def _tab_primary():
    rows = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule", "Test & Diff. & 95\\% CI & $p$ & Holm $p$ & Rescue/harm \\\\", "\\midrule"]
    rows += [_test_row(f"P{i + 1}", [i]) for i in range(3)]
    return "\n".join(rows + ["\\bottomrule", "\\end{tabular}", ""])


def _tab_exploratory(res):
    rows = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule", "Model & Test & Diff. & 95\\% CI & $p$ & Rescue/harm \\\\", "\\midrule"]
    for e in res["exploratory"]:
        for i in range(3):
            r = _test_row(f"P{i + 1}", ["exp", e["model"], i], True)
            rows.append(f"\\{_name(e['model'], 'model')} & " + r)
    return "\n".join(rows + ["\\bottomrule", "\\end{tabular}", ""])


def _fig_dat(res):
    cells = {(c["model"], c["fault"], c["arm"]): c["success"] for c in res["cells"] if c["in_tests"]}
    lines = ["model arm drop wrong partial read"]
    for m in FIG_MODELS:
        for a in FIG_ARMS:
            try:
                v = [fmt(cells[(m, f, a + "/recover")], "prop") for f in FIG_FAULTS]
            except KeyError as e:
                raise ValueError(f"missing cell for fig_success.dat: {e}") from e
            lines.append(" ".join([tag(m), tag(a + "/recover"), *v]))
    return "\n".join(lines) + "\n"


def write_assets(res: dict, out: Path) -> dict:
    nums = build_numbers(res)
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.tex").write_text("".join(f"\\newcommand{{\\{n}}}{{{d['formatted']}}}\n" for n, d in nums.items()), newline="\n")
    (out / "numbers.json").write_text(json.dumps(nums, indent=1, sort_keys=True) + "\n", newline="\n")
    (out / "tab_primary.tex").write_text(_tab_primary(), newline="\n")
    for m in FIG_MODELS:
        (out / f"tab_success_{tag(m)}.tex").write_text(_success_table(res, m, nums), newline="\n")
    (out / "tab_exploratory.tex").write_text(_tab_exploratory(res), newline="\n")
    (out / "fig_success.dat").write_text(_fig_dat(res), newline="\n")
    return nums
