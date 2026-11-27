"""Paired statistics (standard library only)."""
from __future__ import annotations

import random
from collections import defaultdict
from math import comb, sqrt
from statistics import NormalDist


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def paired_counts(base: list[bool], arm: list[bool]) -> tuple[int, int]:
    return (sum(not a and b for a, b in zip(base, arm)), sum(a and not b for a, b in zip(base, arm)))


def _task_diffs(rows: list[dict], arm: str, base: str) -> list[float]:
    """Per-task mean success(arm) - success(base), sorted by task_id.

    Rows pair on (task_id, seed) when a "seed" key is present, else on task_id; only pairs present in both
    arms count. Each task's pairs average to one cluster difference, and clusters are resampled by task_id.
    """
    by = {a: {(r["task_id"], r.get("seed")): r["success"] for r in rows if r["arm"] == a} for a in (arm, base)}
    per = defaultdict(list)
    for k in by[arm].keys() & by[base].keys():
        per[k[0]].append(int(by[arm][k]) - int(by[base][k]))
    diffs = [sum(v) / len(v) for _, v in sorted(per.items())]
    if not diffs:
        raise ValueError("no paired rows")
    return diffs


def cluster_bootstrap_diff(rows: list[dict], arm: str, base: str, *, n=10_000, seed=0) -> tuple[float, float, float]:
    """Mean success(arm) - success(base) with a 95% percentile CI, resampling tasks (pairing: see _task_diffs)."""
    diffs = _task_diffs(rows, arm, base)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), means[int(0.025 * (n - 1))], means[int(0.975 * (n - 1) + 0.5)]


def permutation_p(rows: list[dict], arm: str, base: str, *, n=10_000, seed=0) -> float:
    """Two-sided sign-flip permutation test on the per-task mean differences."""
    diffs = _task_diffs(rows, arm, base)
    if not any(diffs):
        return 1.0
    rng, obs = random.Random(seed), abs(sum(diffs))
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12 for _ in range(n))
    return (1 + hits) / (n + 1)


def mde_tasks(n_tasks: int, sd_task_diff: float, alpha=0.05, power=0.8) -> float:
    """Smallest mean per-task difference detectable (normal approximation)."""
    z = NormalDist().inv_cdf
    return (z(1 - alpha / 2) + z(power)) * sd_task_diff / sqrt(n_tasks)


def mde_paired(n_pairs: int, p_discordant: float, alpha=0.05, power=0.8) -> float:
    """Smallest absolute success-proportion difference detectable by a two-sided paired test.

    Normal approximation: the per-pair difference has variance ~ p_discordant (small-difference limit),
    so MDE = (z_{1-alpha/2} + z_power) * sqrt(p_discordant / n_pairs).
    """
    z = NormalDist().inv_cdf
    return (z(1 - alpha / 2) + z(power)) * sqrt(p_discordant / n_pairs)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    out, run, m = {}, 0.0, len(pvals)
    for i, (k, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out
