# Paper replication agent

Reproduces a technical paper's headline results, with an attributable verdict, for two tracks:
quant-finance (cross-sectional anomalies, ML return prediction) and CS/ML (checkpoint
evaluation, train-and-eval). Design: `DESIGN.md` (§9 is the MVP cut this code implements);
review history: `DESIGN_REVIEW.md`.

## Layout

```
replicator/
  schema.py            spec.yaml / runs/*.json / report.json contracts; the freeze (principle 0.7)
  llm.py               Anthropic SDK wrapper: structured calls over the PDF, cost cap, call log
  kb/                  conventions KB (two finance + two CS families) + OSAP SignalDoc.csv (331 papers)
  spec/intake.py       PDF -> page images + text, companion links, arXiv version
  spec/extract.py      extractor -> referee -> per-claim page re-read
  triage.py            data probe, two-axis tiers, derived tolerances, budget, FREEZE, dry-run
  data/                content-addressed cache, pandera schemas, point-in-time lineage, Table-1 checkpoint,
                       adapters (Kenneth French library, local file); credentials never enter the sandbox
  templates/anomaly.py finance family: load -> signal -> sort -> portfolios -> evaluate (+ pre-formed portfolios)
  templates/cs_eval.py CS contract: reproduce.sh writes metrics.json per seed
  build/               bounded tool loop (model cannot end the stage), subprocess sandbox, blacklist monitor
  verify/              derived tolerance (k*SE + precision), one Match rule, shuffle / future-perturbation /
                       available_at leakage tests, convention grid (sensitivity, not search)
  orchestrator.py      spec -> setup -> build+run -> verify -> report; budgets; report always produced
  report.py            grade vector -> letter; REPORT.md + report.json
  cli.py               `replicate`, `--dry-run`, `eval-spec`
evals/spec_eval.py     spec-extraction eval against SignalDoc fields
papers/                hand replications = benchmark entries (momentum on French data; MNIST MLP)
tests/                 unit + end-to-end (hermetic after first data fetch)
```

## Run

```bash
uv sync
uv run pytest -q

# finance, hermetic, no model calls (prebuilt pipeline = what the builder is expected to write)
uv run python -m replicator.cli replicate papers/momentum_french/spec.yaml --out runs/momentum --batch --prebuilt papers/momentum_french

# CS, three seeds, convention grid over loss/optimizer/epochs/hidden
PYTHON=$PWD/.venv/bin/python uv run python -m replicator.cli replicate papers/mnist_mlp/spec.yaml --out runs/mnist --batch --prebuilt papers/mnist_mlp

# from a PDF (needs Anthropic credentials: ANTHROPIC_API_KEY or `ant auth login`)
uv run python -m replicator.cli replicate paper.pdf --track finance --family cross_sectional_anomaly --osap Mom12m --dry-run
uv run python -m replicator.cli replicate paper.pdf --track cs --family checkpoint_eval
```

Interactive mode shows one checkpoint (headline claims, ambiguities by sensitivity, tier,
substitutions) after triage; `--batch` auto-approves; `--dry-run` stops there.

## What is built and what is not

Built and tested: contracts and freeze; derived tolerances; the Match rule; the three leakage
tests (the future-perturbation test catches feature look-ahead that the shuffle test cannot);
convention grid; French-library adapter with cache and schema checks; anomaly template (both
pre-formed-portfolio and stock-level sorts); CS reproduce.sh contract with seed aggregation;
orchestrator with budgets, blacklist downgrade, and report-always; grade vector; the two
hand replications as benchmark entries; the SignalDoc spec-eval scorer; builder loop mechanics
(exercised with a scripted fake model).

Needs credentials to exercise: the extractor, referee, re-reader, and the live builder loop.
Not in the MVP (see DESIGN.md §9): Tiingo/WRDS adapters, contamination scan, bootstrap for
overlapping returns, Hamilton DAG caching, Docker sandbox, era images.

## Data attribution

`replicator/kb/SignalDoc.csv` is the signal documentation table from Open Source Asset Pricing
(Chen & Zimmermann), https://github.com/OpenSourceAP/CrossSection, used as the conventions-KB seed
and the spec-extraction ground truth. Kenneth French library data is fetched at run time.
