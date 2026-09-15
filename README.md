# Paper replication agent

Reproduces a technical paper's headline results with an **attributable verdict**, for two tracks:
quant-finance (cross-sectional anomalies, ML return prediction) and CS/ML (checkpoint evaluation,
train-and-eval). The deliverable is not the number; it is the report that says which number was
obtained, under which conventions, whether it could have leaked, and what remains unexplained.

- `DESIGN.md` — the design, derived from first principles; §9 is the MVP cut this code implements,
  §10 the implementation status.
- `DESIGN_REVIEW.md` — the adversarial review (11 roles, 57 items) that produced §9.

## Results so far (2026-09-15, CPU-only laptop)

| Paper | Track | Kind of test | Grade | What was reproduced | Cost |
|---|---|---|---|---|---|
| Jegadeesh–Titman 1993 momentum (OSAP Mom12m as ground truth) | finance | re-implementation, live builder | **A** | headline 1.32 vs 1.31 %/mo Match; t-stat 4.76 vs 3.74 Match; shuffle + future-perturbation pass | $0.93 |
| LeCun 1998 MLP-300 on MNIST, 3 seeds | CS | re-implementation | C | runs cleanly; 8.5% vs 4.7% error; convention grid 1.7–9.8% shows the modern setup lands elsewhere | $0 (prebuilt) |
| Vaswani 2017 Transformer | CS | re-implementation, reduced scale | C | mechanism: attention 99.4% vs no-attention 9.9% token accuracy on a copy task; WMT BLEU not comparable at this scale | $4.72 |
| 12 further CS papers (ResNet, LayerNorm, BatchNorm, GAN, DQN, PPO, World Models, Lottery Ticket, LoRA, DPO, Toy Models of Superposition, Transformer Circuits, ROME) | CS | reduced scale | in progress | specs extracted for all 14 (`papers/batch/*/spec.yaml`), builds running | $32.90 for all 14 specs |

**Reading a grade.** The letter is a fixed function of four axes (data fidelity, procedure fidelity,
result, integrity). A: headline claims Match within pre-registered tolerances, leakage tests pass,
open data. B: Consistent with attributed gaps. C: mechanics verified, numbers not (the usual outcome
for a paper whose numbers need GPUs when you have a CPU). F: a leakage test failed or the pipeline
did not run. `REPORT.md` always states the *kind of test* (reproduction / re-implementation /
conceptual replication / mechanics only) and the compute tier, so a C on a reduced-scale run is
not mistaken for a failed method.

## Layout

```
replicator/
  schema.py            spec.yaml / runs/*.json / report.json contracts; the freeze (principle 0.7)
  llm.py               Anthropic SDK wrapper: structured or schema-guided JSON over the PDF, 5-min call cap, cost cap, call log
  kb/                  conventions KB (2 finance + 2 CS families) + OSAP SignalDoc.csv (331 papers)
  spec/intake.py       PDF or arXiv id or HTML -> page images + text, companion links, version
  spec/extract.py      extractor -> referee -> per-claim page re-read
  triage.py            data probe, two-axis tiers, derived tolerances, budget, FREEZE, dry-run
  data/                content-addressed cache, pandera contracts, point-in-time lineage, Table-1 checkpoint,
                       adapters (Kenneth French library, Hugging Face datasets, torchvision, local file)
  templates/anomaly.py finance family: load -> signal -> sort -> portfolios -> evaluate (+ pre-formed portfolios)
  templates/cs_eval.py CS contract: reproduce.sh writes metrics.json per seed; process-group timeout
  build/               bounded tool loop (model cannot end the stage; smoke gate runs at budget expiry),
                       subprocess sandbox, author-code blacklist monitor
  verify/              derived tolerance (2·SE + precision), one Match rule, shuffle / future-perturbation /
                       available_at leakage tests, convention grid (sensitivity, not search)
  orchestrator.py      spec -> setup -> build+run -> verify -> report; budgets; report always produced
  report.py            grade vector -> letter; REPORT.md + report.json
  cli.py               `replicate`, `--dry-run`, `eval-spec`
evals/spec_eval.py     spec-extraction eval against SignalDoc fields
evals/batch.py         resumable multi-paper sweep (spec stage parallel, build/run sequential)
papers/                hand replications = benchmark entries; papers/batch/<slug>/spec.yaml = extracted specs
tests/                 unit + end-to-end (hermetic after first data fetch)
```

## Run

```bash
uv sync
uv run pytest -q

# credentials (extraction and the builder are model calls)
export ANTHROPIC_API_KEY='sk-ant-api03-...'        # the secret from Console -> API Keys -> Create Key

# finance, hermetic, no model calls (prebuilt pipeline = what the builder is expected to write)
uv run python -m replicator.cli replicate papers/momentum_french/spec.yaml --out runs/momentum --batch --prebuilt papers/momentum_french

# finance, live builder (the model writes pipeline.py from the frozen spec)
uv run python -m replicator.cli replicate papers/momentum_french/spec.yaml --out runs/mom_live --batch

# CS, three seeds, convention grid
PYTHON=$PWD/.venv/bin/python uv run python -m replicator.cli replicate papers/mnist_mlp/spec.yaml --out runs/mnist --batch --prebuilt papers/mnist_mlp

# from a PDF or an arXiv id: extraction + triage, stop at the checkpoint
uv run python -m replicator.cli replicate 1706.03762 --track cs --family checkpoint_eval --dry-run
uv run python -m replicator.cli replicate paper.pdf --track finance --family cross_sectional_anomaly --osap Mom12m --dry-run

# the 14-paper sweep (resumable; runs/batch/SUMMARY.md is rewritten after every paper)
uv run python -m evals.batch            # all
uv run python -m evals.batch ppo gan    # subset
```

Interactive mode shows one checkpoint (headline claims with page images, ambiguities by
sensitivity with their source, tier, substitutions) after triage; `--batch` auto-approves;
`--dry-run` stops there. Every run writes `REPORT.md`, `report.json`, `spec.yaml` (frozen),
`runs/*.json` (one per execution), `llm/llm_calls.jsonl` (stage, tokens, cost per call), and
`work/` (the generated code and the sandbox command log).

## What a run costs and how long it takes

Measured on the papers above with Claude Opus 5 (extraction, builder) and Claude Sonnet 5
(claim re-reads):

| Stage | Typical | Notes |
|---|---|---|
| Spec (extract + referee + re-read) | $1–3.5, 5–10 min | scales with paper length; ROME (62 claims) was $3.53 |
| Build (live builder) | $1–5, 3–40 min | finance template: 11 turns, 3 min; CS from scratch: up to the 40-min budget |
| Run + verify | $0 | CPU time only; per-run timeout 15–25 min in the sweep |

## What the sweep taught (fixes now in the code)

- A model call can stall for 15–20 minutes and produce nothing. Calls are capped at 5 minutes with
  one retry; a stalled turn counts as an empty turn, not a lost stage.
- The builder will polish instead of testing. The prompt demands an early smoke call, and the
  orchestrator runs the smoke gate itself when the budget expires, so partial work is judged.
- Killing a shell wrapper does not kill the training script it started. Reproduce runs live in
  their own process group and the group is killed on timeout.
- At reduced scale, every method variant is the same run relabeled. Runs are capped to the
  headline procedures (at most three).
- A copy-task BLEU is not comparable to a WMT BLEU. Compute-tier-2 comparisons are reported as
  values, marked not comparable, rather than as Match or Mismatch.
- The builder is a real reviewer: on the momentum paper it noticed the rolling K-month average
  inflates a plain t-stat and computed the standard error from the non-overlapping cohort series
  instead. The hand-built benchmark entry was corrected to match.

## What is built and what is not

Built and tested: contracts and freeze; derived tolerances; the Match rule; the three leakage
tests (the future-perturbation test catches feature look-ahead the shuffle test cannot);
convention grid; French-library, Hugging Face and torchvision adapters with cache and schema
checks; anomaly template (pre-formed-portfolio and stock-level sorts); CS reproduce.sh contract
with seed aggregation; orchestrator with budgets, blacklist downgrade, and report-always; grade
vector; hand replications as benchmark entries; SignalDoc spec-eval scorer; the batch sweep.

Not in the MVP (see DESIGN.md §9): Tiingo/WRDS adapters, contamination scan, block bootstrap for
overlapping returns, Hamilton DAG caching, Docker sandbox, era images, the Extend flag.

## Data attribution

`replicator/kb/SignalDoc.csv` is the signal documentation table from Open Source Asset Pricing
(Chen & Zimmermann), https://github.com/OpenSourceAP/CrossSection, used as the conventions-KB seed
and the spec-extraction ground truth. Kenneth French library data, Hugging Face datasets, and
torchvision datasets are fetched at run time into `data_cache/`.
