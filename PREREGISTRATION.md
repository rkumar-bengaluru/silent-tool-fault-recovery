> **Note for this repository.** This is the pre-registration as it was fixed before the test split
> was run, kept verbatim. Paths in it refer to the original research layout: `poc/` is now `stfr/`,
> and `../gaps.md` (the literature-gap notes) is not included. The paper reports the results.

# POC plan (pre-registration, revision 2): when should a tool agent recover from silent faults?

Gap: G3 + G7 in `../gaps.md`. The study compares what *triggers* recovery (G7) while holding the recovery step fixed. Among the triggers are return-value checks and independent world-state read-back (G3). The faults are tool calls that return success but did the wrong thing. Everything below is fixed before testA is run. testB stays untouched until a Stage 5 send-back.

Revision 2 replaces the version the owner approved at the Task 9 checkpoint. An architect review and the full dev run changed the success definition, the SILENT_READ fault, the statistical test and H5. Each change is listed under "Changes since revision 1".

**Claims this study must state (from gaps.md):**
1. It is a narrow, controlled comparison of oracles on faults where the tool reported success. It is not a new method.
2. Read-back for ambiguous outcomes (2608.02645, 2609.29095), output-contract recovery (2608.19303) and a state checker for silent wrong reads (2610.27777) already exist. The novelty is only the head-to-head comparison with recovery held fixed.
3. A delayed-visibility control (`DELAYED`) is included to expose read-back false alarms.
4. The results come from a mock service with 4B/8B local models and one small Claude model. The tasks are 6 templates × 10 instances per split, not 60 independent tasks.

## Design
- **Faults (7):** `CLEAN`, `LOUD`, `SILENT_DROP`, `SILENT_WRONG`, `SILENT_PARTIAL`, `SILENT_READ`, `DELAYED`.
  - A silent write fault returns the same echo a correct write would return. This echo is computed before the corruption is applied.
  - `SILENT_READ` corrupts one field of a returned record, and it prefers int fields. With probability 0.5 the value is implausible (an int `v` becomes `-(v+1)`). Otherwise it is a plausible wrong value.
  - Fault randomness is seeded per `(task, fault, seed)`.
  - `DELAYED` stays stale for 2 further agent calls. This is a design parameter.
- **Tool results:** write tools echo their effect.
  - `update_field` returns the record after the write.
  - `transfer` returns both balances.
  - `create_record` returns the id and the record.
  - `delete_record` returns `{"ok": true}`.
- **Oracles:**
  - The contract sees only `(tool, args, result)`. It checks the schema, field types, and that the echo matches the request.
  - Read-back is an independent `get` of the real state after a write.
- **Arms (9):**
  - The triggers `never`, `always`, `on_error`, `on_uncertainty`, `on_contract`, `on_readback` and `readback_contract`, each with `recover`.
  - `on_contract` and `on_readback` with `abstain`.
  - Every flag injects the same NOTICE, whatever the trigger.
- **Splits:**
  - dev has 30 tasks, testA 60 and testB 60.
  - Dev seeds are [0, 1] and final seeds are [100, 101], so they don't overlap.
  - Temperature is 0.7 for Ollama. Claude takes no temperature, so each seed is a separate cached sample.
- **Success:** the agent ends with status `done`, every goal entry holds in the final state, nothing outside the allowed records changed, and no unintended field changed. **Wrong commit:** status `done` without success. Abstaining counts as failure.
- **Exclusions:**
  - A (task, fault) pair with no eligible call is skipped in every arm.
  - SILENT_PARTIAL needs a transfer or a create, which only 2 of the 6 templates have. So testA skips exactly 40 of its 60 SILENT_PARTIAL pairs, and dev skips 20 of 30. This count comes from the templates and needed no look at testA.
  - Episodes where the fault never applied are kept and flagged `fault_applied`. Any flag in them is a false flag. `fault_applied` rates are reported per arm.
- **Model output errors:** an Ollama "token repeat limit" 500 is treated as an empty reply. Three unreadable replies in a row end the episode as `parse_failure`, which counts as failure. Other 5xx errors and timeouts are retried, and a model that stays unavailable stops the run (exit 4).

## Hypotheses
- H1: on silent-write faults (drop, wrong, partial), `on_readback` has higher success than `on_contract`.
- H2: on `SILENT_READ`, `on_contract` beats `never`, since read-back cannot see a wrong read. The implausible and plausible halves are reported separately.
- H3: `readback_contract` has the best success averaged over all faults.
- H4: `always` and `on_uncertainty` produce the most false flags on `CLEAN`/`DELAYED` and the most harm relative to `never`.
- H5 (descriptive only): after a flag, compare the success and wrong-commit rates of `recover` and `abstain`. Because abstaining is a failure by definition, this hypothesis is not tested.

## Primary comparisons
All primaries use `recover` on `qwen3:8b-q4_K_M`.
- **Test:** a two-sided sign-flip permutation test on per-task mean differences (10,000 flips, seed 0), with Holm correction across P1–P3 only.
- **Descriptive statistics:** the 95% bootstrap CI resamples tasks. The McNemar exact p on (task, fault, seed) pairs is reported as `p_mcnemar` and is descriptive only.
- **Data:** tests use only complete `mode == "final"` runs. A split that mixes run types is refused.
- Everything else is exploratory.

| | Comparison | Faults | H | Also reported |
|---|---|---|---|---|
| P1 | `on_readback` vs `on_contract` | SILENT_DROP + SILENT_WRONG + SILENT_PARTIAL, pooled | H1 | per-fault differences |
| P2 | `on_contract` vs `never` | SILENT_READ | H2 | implausible vs plausible halves |
| P3 | `readback_contract` vs `never` | all 7 faults, CLEAN included | H3 | per-fault differences, `delayed_harm` |

## Dev-only tuning (done)
- **`on_uncertainty` threshold:** the grid {0.5, 0.6, 0.7, 0.8, 0.9} was searched to match `on_readback`'s per-step flag rate on dev CLEAN+DELAYED, which is 0.087. The best match is **0.9**, which gives a flag rate of 0.009. Every lower threshold flags 0%.
  - qwen3:8b nearly always reports confidence ≥ 0.9, so this arm under-fires. That is reported as a finding, not fixed.
  - The dev run used 0.7, so on dev this arm behaved like `never`.
  - The frozen configs use 0.9.
- **`SYSTEM_PROMPT`:** the wording was updated only for the new result shapes, and is frozen from here.

## Full dev run (qwen3:8b, 30 tasks, 190 pairs × 9 arms × 2 seeds = 3,420 episodes, $0, 61 min of new compute)
Run `dev-qwen3_8b-q4_K_M-20261122T150851131985Z`. Overall success is 0.446. Statuses: 3,118 done, 208 abstained, 53 parse failures, 41 hit the step budget.

| fault | never | always | on_error | on_unc. (0.7) | on_contract | on_readback | rb+contract | contract/abstain | readback/abstain |
|---|---|---|---|---|---|---|---|---|---|
| clean | .77 | .77 | .80 | .77 | .77 | .77 | .77 | .77 | .77 |
| loud | .72 | .80 | .77 | .68 | .72 | .72 | .72 | .72 | .72 |
| silent_drop | .00 | .05 | .00 | .00 | .00 | **.37** | **.37** | .00 | .00 |
| silent_wrong | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 |
| silent_partial | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 |
| silent_read | .60 | .47 | .62 | .60 | .62 | .60 | .62 | .42 | .60 |
| delayed | .77 | .75 | .80 | .77 | .77 | .72 | .72 | .77 | **.00** |

Dev values of the primaries (dev is not a result):

| | diff | 95% CI (tasks) | perm. p | task SD | MDE, 60 testA tasks |
|---|---|---|---|---|---|
| P1 | +0.161 | [0.094, 0.233] | 0.0001 | 0.194 | 0.070 |
| P2 | +0.017 | [−0.067, 0.100] | 1.00 | 0.241 | 0.087 |
| P3 | +0.053 | [0.017, 0.089] | 0.010 | 0.100 | 0.036 |

MDE is `mde_tasks(60, task SD, alpha 0.05, power 0.8)`.

What dev shows:
- **P1 comes entirely from SILENT_DROP** (+0.37). On SILENT_WRONG and SILENT_PARTIAL, no arm recovers anything.
  - With SILENT_WRONG, read-back flags the error, but the agent re-applies the field it meant to write and misses the corrupted sibling field.
  - With SILENT_PARTIAL, retrying the transfer would debit twice.
  - These zeros are a finding about recovery, not detection.
- **P2 looks underpowered.** The contract flags 21 of 21 implausible reads, so detection works. But success moves only from 11/21 to 12/21, because most tasks succeed even with the wrong value, and on the others the agent does not fix the read. No arm flags a plausible wrong read.
- **DELAYED false alarms:**
  - Read-back flags 8.7% of CLEAN+DELAYED steps.
  - `on_readback/abstain` drops DELAYED success to 0.00.
  - `on_readback/recover` drops it from 0.77 to 0.72, which is the `delayed_harm` P3 reports.
- **`always` hurts on SILENT_READ** (0.47 vs 0.60 for `never`). It is the best arm on LOUD (0.80).

## Cost and time projection (testA)
- **Pairs:** 60 × 7 − 40 skipped = **380**.
- **Local models (4b, 8b):** 380 × 9 arms × 2 seeds = 6,840 episodes each, about 2.0 h for 4b and 5.0 h for 8b. Both are under the 12 h rule, so there are no cuts. (The rule: cut seeds to 1 first, then tasks to 40.)
- **Claude:** 380 × 9 × 1 seed = 3,420 episodes, about $0.68. With a 1.25× reserve that is about $0.86.
- **Optional dev on Claude** is capped at $3. Claude spend so far is $0.068. The total is far below the $40 warning and the $50 hard stop.

## Models and frozen configs (written at freeze, threshold 0.9)
- `qwen3:8b-q4_K_M`, full design: `poc/configs/frozen_qwen3_8b.json`, seeds [100, 101].
- `qwen3:4b-q4_K_M`, full design, as a size contrast: `poc/configs/frozen_qwen3_4b.json`, seeds [100, 101].
- `claude-haiku-5-5`, 9 arms × 7 faults, 1 seed: `poc/configs/frozen_claude.json`, seed [100], `est_usd_per_episode` 0.0002.

## Changes since revision 1
1. **Success** now requires status `done` and no unintended field change. Before, the final state alone decided success.
2. **Write results** now echo their effect, and the contract checks that the echo matches the request and that field types are correct. Without this, a return-value check had nothing to check on writes.
3. **`SILENT_READ`** now corrupts an int field and makes half the values implausible. This gives the contract a fault class it should win, so P2 can go either way.
4. **Primary test:** a task-level permutation test replaces McNemar, because the pairs cluster by task. McNemar is kept as a descriptive statistic. The tables report per-fault results and `delayed_harm`.
5. **H5** is descriptive only.
6. **MDE** is computed at the task level from dev SD, replacing the pair-level approximation.
7. **Runtime fixes made during dev** (no change to design):
   - Fault rng and `fault_slot` are seeded per (task, fault, seed). Before, kinds and slots were identical across tasks.
   - Ollama errors are retried.
   - The token-repeat 500 is handled as a parse failure.
   - Analysis uses only final-mode runs.

## Owner decisions at this checkpoint
1. Approve revision 2 as the pre-registration.
2. **P2:** dev suggests no testA effect is detectable (diff 0.017 vs MDE 0.087).
   - (a) Keep it as a primary. A null result alongside 100% detection is reportable, and Holm costs P1/P3 little. **This is the recommended option.**
   - (b) Demote it to exploratory, with Holm over P1 and P3 only.
3. **SILENT_WRONG and SILENT_PARTIAL in P1:** both are at 0 for every arm, so they dilute P1. The default is to keep them as pre-registered and report per fault.
