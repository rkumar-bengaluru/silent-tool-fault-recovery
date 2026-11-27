# silent-tool-fault-recovery

Code, data and paper source for **"When Should a Tool Agent Recover? Comparing Triggers for Silent Tool
Faults with Recovery Held Fixed"** (Rupak Kumar, AvantaLabs AI, 2026).

[![Preprint DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24590562.svg)](https://doi.org/10.5281/zenodo.24590562)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24586177.svg)](https://doi.org/10.5281/zenodo.24586177)
[![Logs DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24585902.svg)](https://doi.org/10.5281/zenodo.24585902)

- Paper (PDF): [`paper/paper.pdf`](paper/paper.pdf), built from the source in `paper/`;
  preprint on Zenodo: [10.5281/zenodo.24590562](https://doi.org/10.5281/zenodo.24590562)
- Code v1.0.0, archived: [10.5281/zenodo.24586177](https://doi.org/10.5281/zenodo.24586177)
  (all versions: [10.5281/zenodo.24586176](https://doi.org/10.5281/zenodo.24586176))
- Episode logs of the final runs: [10.5281/zenodo.24585902](https://doi.org/10.5281/zenodo.24585902)

A tool-using agent works on a small in-memory record service. Faults are injected into its tool calls.
The worst are *silent* faults: a write is dropped, lands on the wrong field, or is applied only in part,
while the tool still returns the echo a correct write would have returned. The agent and its single
recovery message are held fixed, and only the **trigger** that decides when the agent is told "this step
may be wrong" varies:

| trigger | fires when |
|---|---|
| `never`, `always` | never / after every tool call (the bounds) |
| `on_error` | the tool returned an error |
| `on_uncertainty` | the agent's stated confidence is below a threshold |
| `on_contract` | the result breaks the tool's return-value contract (schema, types, value domains, echo) |
| `on_readback` | an independent re-read of the world state differs from the write's expected effect |
| `readback_contract` | either of the two above |

Each flag is answered with **recover** (continue with a notice) or **abstain** (stop, counted as a
failure). Success is graded on the final world state, never by a language model. The pre-registered
analysis (P1–P3 on `qwen3:8b`) and all hypotheses are in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Layout

```
stfr/                  the experiment (package; CLI `stfr`)
  svc.py               record service and tools
  faults.py            fault injector (clean, loud, silent_drop, silent_wrong, silent_partial, silent_read, delayed)
  triggers.py          the seven triggers; contracts.py holds the return-value contracts
  agent.py             agent loop, system prompt and NOTICE
  tasks.py             task templates, splits, sealing and fault slots
  grade.py, stats.py   grading; permutation test, Holm, task-cluster bootstrap
  run.py               CLI: pilot, dev, final, aggregate, assets
  paper_assets.py      results.json -> LaTeX macros, tables and figure data
  llm.py               Ollama and Claude clients with a response cache and a spend cap
  configs/             dev configs and the three frozen configs used for the paper
  data/                dev, testA and testB task splits; SEALED.json holds their hashes
tests/                 unit tests (pytest)
results/results.json   every number in the paper, aggregated from the final runs
paper/                 LaTeX source and the built paper.pdf; paper/generated/ is written by `stfr assets`
PREREGISTRATION.md     the analysis plan, fixed before the test split was run
```

## Install

Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
```

## Reproduce the paper's numbers

**1. Tables and figure from `results.json`** (no models needed):

```
uv run stfr assets            # rewrites paper/generated/ from results/results.json
cd paper && latexmk -pdf main.tex
```

`assets` refuses to run if the confirmatory tests block of `results.json` does not match the frozen hash.

**2. `results.json` from the episode logs.** The logs of the three final runs (about 70 MB: every step,
tool call, flag and grade of every episode) are archived on Zenodo at
[10.5281/zenodo.24585902](https://doi.org/10.5281/zenodo.24585902) (`stfr-episode-logs-v1.0.0.zip`, 2.1 MB). Unzip it into
`results/runs/`, then:

```
uv run stfr aggregate --results results \
  results/runs/final-claude-haiku-5-5-20261123T122505323442Z \
  results/runs/final-qwen3_4b-q4_K_M-20261124T011458069993Z \
  results/runs/final-qwen3_8b-q4_K_M-20261123T150409505835Z
```

This rebuilds `results/results.json` byte for byte.

**3. Re-run the experiment.** The local models run through [Ollama](https://ollama.com)
(`ollama pull qwen3:8b-q4_K_M` and `qwen3:4b-q4_K_M`). The Claude run needs `ANTHROPIC_API_KEY`.

```
uv run stfr final --config stfr/configs/frozen_qwen3_8b.json
uv run stfr final --config stfr/configs/frozen_qwen3_4b.json
uv run stfr final --config stfr/configs/frozen_claude.json
```

| model | episodes | time (paper run) | cost |
|---|---|---|---|
| `qwen3:8b-q4_K_M` | 6840 | about 10 h | local |
| `qwen3:4b-q4_K_M` | 6840 | about 5 h | local |
| `claude-haiku-5-5` | 3420 | about 3 h | about $0.91 |

A final run refuses to start if `stfr/` has uncommitted changes, if its seeds overlap the dev seeds, or
if it already ran once (`results/final_<split>_<model>.lock`). Local sampling is not guaranteed to be
bit-exact across machines and Ollama versions, so a re-run may give different episodes; the paper's
episodes are the Zenodo logs. Use `stfr dev --config stfr/configs/dev.json` for development runs.

## Splits

- `dev` (30 tasks): used for every design choice.
- `testA` (60 tasks): the only test split behind the paper's results, run once per model with the
  frozen configs.
- `testB` (60 tasks): **never run and not used for any result in the paper.** It is released for
  future work. Like testA it is sealed: it loads only in `final` mode and only if its hash matches
  `stfr/data/SEALED.json`.

## Citing

Please cite the paper:

```bibtex
@misc{kumar2026recover,
  title     = {When Should a Tool Agent Recover? Comparing Triggers for Silent Tool Faults with Recovery Held Fixed},
  author    = {Kumar, Rupak},
  year      = {2026},
  publisher = {Zenodo},
  note      = {Preprint},
  doi       = {10.5281/zenodo.24590562},
  url       = {https://doi.org/10.5281/zenodo.24590562}
}
```

[`CITATION.cff`](CITATION.cff) has the same entry plus the code and logs DOIs. The arXiv identifier will
be added once the paper is posted there.

## Licence

Code: [MIT](LICENSE). Task splits, `results/`, the episode logs and the paper source in `paper/`:
[CC BY 4.0](LICENSE-DATA).

The code and paper were written with AI assistance (Claude Code); see the AI-assistance statement in
the paper.
