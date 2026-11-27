"""CLI: `stfr <pilot|dev|final|aggregate|assets> --config <path>`. results/ and paper/ are at the repository root."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from stfr.llm import BudgetExceeded, Cache, ClaudeChat, CostTracker, ModelUnavailable, OllamaChat

from . import paper_assets
from .agent import run_episode
from .faults import Fault
from .grade import grade
from .stats import cluster_bootstrap_diff, holm, mcnemar_exact, paired_counts, permutation_p
from .tasks import DATA, SealBroken, SealedSplit, fault_slot, load_split
from .triggers import TRIGGERS

ROOT = Path(__file__).resolve().parent.parent  # repository root
ALL_FAULTS = [f.value for f in Fault]
SILENT_WRITE = ["silent_drop", "silent_wrong", "silent_partial"]
PRIMARY_MODEL = "qwen3:8b-q4_K_M"
# Fixed before any result is seen; all use the arms' "recover" response. Holm runs over exactly these three.
PRIMARY = [
    {"name": "on_readback_vs_on_contract_silent_write", "arm": "on_readback/recover", "base": "on_contract/recover", "faults": SILENT_WRITE},
    {"name": "on_contract_vs_never_silent_read", "arm": "on_contract/recover", "base": "never/recover", "faults": ["silent_read"]},
    {"name": "readback_contract_vs_never_all", "arm": "readback_contract/recover", "base": "never/recover", "faults": ALL_FAULTS},
]


def _safe(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def default_chat(cfg, *, cache, tracker, run_id, final):
    if cfg["provider"] == "claude":
        return ClaudeChat(cfg["model"], cache=cache, tracker=tracker, run_id=run_id, final=final)
    url = "http://localhost:11434"
    try:  # cheap availability probe: the server answers and lists the model
        tags = httpx.get(f"{url}/api/tags", timeout=5).json()
    except (httpx.HTTPError, ValueError) as e:
        raise ModelUnavailable(f"Ollama not reachable at {url}") from e
    if cfg["model"] not in {m.get("name") for m in tags.get("models", [])}:
        raise ModelUnavailable(f"model {cfg['model']} not installed in Ollama")
    return OllamaChat(cfg["model"], base_url=url, cache=cache)


def _git_dirty() -> str:
    """Non-empty when stfr/ is dirty OR git itself failed (unknown state counts as dirty)."""
    try:
        r = subprocess.run(["git", "status", "--porcelain", str(ROOT / "stfr")], capture_output=True, text=True, cwd=ROOT)
    except OSError as e:
        return f"git unavailable: {e}"
    return r.stdout.strip() or ("git status failed" if r.returncode else "")


def _split_size(name: str) -> int:
    return sum(1 for l in (DATA / f"{name}.jsonl").read_text().splitlines() if l)  # counts without loading a sealed split


def _episodes(cfg, tasks, seeds):
    for t in tasks:
        for f in cfg["faults"]:
            for s in seeds:
                yield t, Fault(f), s, fault_slot(t, Fault(f), s)


def _n_episodes(cfg, n_tasks, seeds=None):
    return n_tasks * len(cfg["faults"]) * len(cfg["arms"]) * len(seeds if seeds is not None else cfg["seeds"])


def main(argv=None, chat_factory=None, git_status=None) -> int:
    ap = argparse.ArgumentParser(prog="stfr")
    ap.add_argument("mode", choices=["pilot", "dev", "final", "aggregate", "assets"])
    ap.add_argument("--config")
    ap.add_argument("--project-config", help="pilot: config to project the cost and time of")
    ap.add_argument("--results", help="aggregate: output directory; assets: directory holding results.json")
    ap.add_argument("runs", nargs="*", help="aggregate: run directories")
    a = ap.parse_args(argv)
    if a.mode == "aggregate":
        aggregate([Path(r) for r in a.runs], Path(a.results or ROOT / "results") / "results.json")
        return 0

    def refuse(msg, code=2):
        print(f"REFUSED: {msg}", file=sys.stderr)
        return code

    if a.mode == "assets":
        res = json.loads((Path(a.results or ROOT / "results") / "results.json").read_text())
        if paper_assets.tests_hash(res["tests"]) != paper_assets.FROZEN_HASH:
            return refuse("results.json tests block does not match the frozen hash")
        paper_assets.write_assets(res, ROOT / "paper" / "generated")
        return 0

    if a.results is not None:  # a different results dir would mean a new lock and a fresh $0 cost.json
        return refuse("--results is only for aggregate")
    res = ROOT / "results"
    cfgp = Path(a.config)
    cfg = json.loads(cfgp.read_text())
    seeds = cfg["seeds"][:1] if a.mode == "pilot" else cfg["seeds"]
    for arm in cfg["arms"]:
        if arm["trigger"] not in TRIGGERS or arm["response"] not in ("recover", "abstain"):
            return refuse(f"unknown arm {arm}")
    if bad := [f for f in cfg["faults"] if f not in ALL_FAULTS]:
        return refuse(f"unknown faults {bad}")
    final = a.mode == "final"
    lock = res / f"final_{cfg['split']}_{_safe(cfg['model'])}.lock"
    est = cfg.get("est_usd_per_episode", 0.0)
    if final:
        if cfg.get("frozen") is not True:
            return refuse("config is not frozen")
        cdir = (ROOT / "stfr" / "configs").resolve()
        if not (cfgp.resolve().parent == cdir and cfgp.name.startswith("frozen") and cfgp.suffix == ".json"):
            return refuse("config must be stfr/configs/frozen*.json")
        if cfg["split"] not in ("testA", "testB"):
            return refuse("final runs only on a sealed test split")
        if cfg["provider"] == "claude" and not est > 0:
            return refuse("a Claude final run needs est_usd_per_episode > 0 to reserve")
        if (git_status or _git_dirty)():
            return refuse("stfr/ has uncommitted changes (or git status failed)")
        if lock.exists():
            return refuse(f"{lock.name} exists: the final run happens once")
        dev_seeds = {s for p in cdir.glob("dev*.json") for s in json.loads(p.read_text())["seeds"]}
        if dev_seeds & set(seeds):
            return refuse(f"final seeds overlap dev seeds: {sorted(dev_seeds & set(seeds))}")
    elif cfg["split"] != "dev":
        return refuse("only the dev split may be used outside final mode")
    run_id = f"{a.mode}-{_safe(cfg['model'])}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    tracker, cache = CostTracker(res / "cost.json"), Cache(res / "cache")
    try:
        chat = (chat_factory or default_chat)(cfg, cache=cache, tracker=tracker, run_id=run_id, final=final)
    except ModelUnavailable as e:
        return refuse(str(e), 4)
    limit = 3 if a.mode == "pilot" else cfg["task_limit"]
    if final:  # reserve and lock BEFORE the sealed split is read; reserving first keeps a refusal from burning the run
        n_tasks = min(_split_size(cfg["split"]), limit or _split_size(cfg["split"]))
        proj = _n_episodes(cfg, n_tasks) * est * 1.25
        try:
            tracker.check(proj)
        except BudgetExceeded as e:
            return refuse(str(e), 3)
        tracker.reserve(proj, run_id)
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(run_id)
    try:
        tasks = load_split(cfg["split"], allow_sealed=final)[:limit]
    except (SealedSplit, SealBroken) as e:
        tracker.release(run_id)
        return refuse(str(e))

    out = res / "runs" / run_id
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1), newline="\n")
    skipped, n, code, err, cost, t0 = defaultdict(int), 0, 0, None, 0.0, time.time()
    try:
        with open(out / "episodes.jsonl", "w", newline="\n") as fh:
            for task, fault, seed, slot in _episodes(cfg, tasks, seeds):
                if slot is None:  # skipped for every arm so the design stays matched
                    skipped[fault.value] += 1
                    continue
                for arm in cfg["arms"]:
                    ep = run_episode(task, chat, fault=fault, slot=slot, trigger=arm["trigger"], response=arm["response"],
                                     threshold=cfg["threshold"], seed=seed, max_steps=cfg["max_steps"])
                    cost += ep.cost_usd
                    n += 1
                    row = {**asdict(ep), "grade": grade(task, ep)}
                    row.pop("cost_usd")  # differs between a first run and a cached rerun; the total is in summary.json
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    fh.flush()
    except BudgetExceeded as e:
        code, err = 3, f"BudgetExceeded: {e}"
    except ModelUnavailable as e:
        code, err = 4, f"ModelUnavailable: {e}"
    except BaseException as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        if cost > 0 and getattr(chat, "tracker", None) is None:  # ClaudeChat charges its own calls
            tracker.charge(cost, run_id)
        tracker.release(run_id)
        secs = time.time() - t0
        summ = {"run_id": run_id, "mode": a.mode, "model": cfg["model"], "complete": err is None, "episodes": n,
                "skipped_pairs": dict(skipped), "cost_usd": cost, "seconds": secs}
        if err:
            summ["error"] = err
            print(f"STOPPED: {err}", file=sys.stderr)
        (out / "summary.json").write_text(json.dumps(summ, indent=1), newline="\n")
    if a.mode == "pilot" and n:
        print(f"pilot: {n} episodes, {secs / n:.2f} s/episode, ${cost / n:.5f}/episode")
        if a.project_config:
            pc = json.loads(Path(a.project_config).read_text())
            nt = _split_size(pc["split"]) if pc["task_limit"] is None else pc["task_limit"]
            m = _n_episodes(pc, nt)
            print(f"projection for {a.project_config}: {m} episodes, {m * secs / n / 3600:.1f} h, ${m * cost / n:.2f}")
    print(f"{run_id}: {n} episodes, complete={err is None}")
    return code


def aggregate(run_dirs: list[Path], out: Path) -> None:
    runs, rows, expl, excluded = [], [], [], []
    for d in map(Path, run_dirs):
        cfg, s = json.loads((d / "config.json").read_text()), json.loads((d / "summary.json").read_text())
        in_tests = s.get("mode") == "final" and s.get("complete") is True  # only final, complete runs feed the tests
        runs.append({"run_id": s["run_id"], "config": cfg, "model": cfg["model"], "seeds": cfg["seeds"],
                     "cost_usd": s["cost_usd"], "complete": s["complete"], "mode": s.get("mode"), "in_tests": in_tests})
        mine = []
        for l in (d / "episodes.jsonl").read_text().splitlines():
            r = json.loads(l)
            mine.append({**r, "model": cfg["model"], "arm": f"{r['trigger']}/{r['response']}", "_in_tests": in_tests})
        rows += mine
        if in_tests:
            expl.append(_exploratory(s["run_id"], cfg, mine))
        else:
            excluded.append({"run_id": s["run_id"], "reason": f"mode={s.get('mode')!r} complete={s.get('complete')!r}: not a final, complete run"})
    splits = {r["config"]["split"] for r in runs if r["in_tests"]}
    if len(splits) > 1:
        print(f"REFUSED: tests would mix splits {sorted(splits)}", file=sys.stderr)
        raise SystemExit(2)
    keys = [(e["model"], e["split"]) for e in expl]
    if len(keys) != len(set(keys)):
        raise ValueError(f"two exploratory runs share (model, split): {keys}")
    cells = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["fault"], r["arm"], r["_in_tests"])].append(r)
    for (m, f, arm, it), g in sorted(groups.items()):
        n = len(g)
        mean = lambda k: sum(r["grade"][k] for r in g) / n
        cells.append({"model": m, "fault": f, "arm": arm, "in_tests": it, "n": n, "success": mean("success"),
                      "wrong_commit": mean("wrong_commit"), "abstain": mean("abstained"),
                      "false_flags_per_episode": mean("false_flags"), "extra_calls": mean("extra_calls"),
                      "readback_calls": mean("readback_calls"), "tokens": sum(r["tokens_in"] + r["tokens_out"] for r in g) / n,
                      "duplicates": sum(r["grade"]["duplicates"] for r in g)})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"runs": runs, "cells": cells, "tests": _tests([r for r in rows if r["_in_tests"]]),
                                         "exploratory": expl, "exploratory_excluded": excluded}, indent=1), newline="\n")


def _exploratory(run_id, cfg, rows) -> dict:
    """Per-run, post-hoc numbers; every episode counts, whatever its status."""
    arms = sorted({r["arm"] for r in rows})
    sk = {a: {k: {"n": 0, "success": 0, "flagged": 0} for k in ("plausible", "implausible")}
          for a in ("never/recover", "on_contract/recover", "on_readback/recover")}
    for r in rows:
        if r["fault"] == "silent_read" and r["arm"] in sk and r["grade"].get("fault_read_kind") in sk[r["arm"]]:
            c = sk[r["arm"]][r["grade"].get("fault_read_kind")]
            c["n"] += 1
            c["success"] += bool(r["grade"]["success"])
            c["flagged"] += any(s["flagged"] for s in r.get("steps", []))
    far = {a: {} for a in arms}
    for a in arms:
        for f in ALL_FAULTS:
            g = [r for r in rows if r["arm"] == a and r["fault"] == f]
            if f != "clean" and g:
                far[a][f] = sum(bool(r["grade"].get("fault_applied")) for r in g) / len(g)
    fr = {}
    for a in arms:
        st = [s for r in rows if r["arm"] == a and r["fault"] in ("clean", "delayed") for s in r.get("steps", []) if s["tool"] is not None]
        fr[a] = sum(bool(s["flagged"]) for s in st) / len(st) if st else None
    sc = defaultdict(int)
    for r in rows:
        sc[r.get("status")] += 1
    return {"run_id": run_id, "model": cfg["model"], "split": cfg["split"], "episodes": len(rows),
            "tests": _tests(rows, cfg["model"]), "silent_read_by_kind": sk, "status_counts": dict(sc),
            "fault_applied_rate": far, "flag_rate_clean_delayed": fr}


def _tests(rows, model=PRIMARY_MODEL) -> list[dict]:
    rows = [r for r in rows if r["model"] == model]
    if not rows:
        return []
    keys = [(r["task_id"], r["fault"], r["seed"], r["arm"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (task, fault, seed, arm) episodes: runs overlap")
    tests = []
    for c in PRIMARY:
        # pair on (task, fault, seed); the bootstrap clusters on task_id
        sel = [{"task_id": r["task_id"], "seed": (r["fault"], r["seed"]), "arm": r["arm"], "success": r["grade"]["success"]}
               for r in rows if r["fault"] in c["faults"] and r["arm"] in (c["arm"], c["base"])]
        pa = {(r["task_id"], r["seed"]): r["success"] for r in sel if r["arm"] == c["arm"]}
        pb = {(r["task_id"], r["seed"]): r["success"] for r in sel if r["arm"] == c["base"]}
        ks = sorted(pa.keys() & pb.keys(), key=str)
        t = {"name": c["name"], "arm": c["arm"], "base": c["base"], "faults": c["faults"], "n_pairs": len(ks)}
        if ks:
            rescue, harm = paired_counts([pb[k] for k in ks], [pa[k] for k in ks])
            d, lo, hi = cluster_bootstrap_diff(sel, c["arm"], c["base"])
            t.update(rescue=rescue, harm=harm, p=permutation_p(sel, c["arm"], c["base"]),
                     p_mcnemar=mcnemar_exact(rescue, harm), diff=d, ci=[lo, hi])
            # by_fault diff is pair-level: equals the clustered diff only when seeds per task are equal
            t["by_fault"] = {}
            for f in c["faults"]:
                kf = [k for k in ks if k[1][0] == f]
                if kf:
                    rf, hf = paired_counts([pb[k] for k in kf], [pa[k] for k in kf])
                    t["by_fault"][f] = {"diff": sum(pa[k] - pb[k] for k in kf) / len(kf), "rescue": rf, "harm": hf}
            if "delayed" in c["faults"]:
                t["delayed_harm"] = sum(pb[k] and not pa[k] for k in ks if k[1][0] == "delayed")
        else:
            t.update(rescue=0, harm=0, p=1.0, p_mcnemar=1.0, diff=None, ci=None, by_fault={})
            if "delayed" in c["faults"]:
                t["delayed_harm"] = 0
        tests.append(t)
    hp = holm({t["name"]: t["p"] for t in tests})
    for t in tests:
        t["holm_p"] = hp[t["name"]]
    return tests
