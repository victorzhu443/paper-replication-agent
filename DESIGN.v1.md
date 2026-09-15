# Paper Replication Agent — Design (first principles)

**Goal.** An agent that takes a quant-finance or CS paper and, within 1–2 hours of wall-clock
time, produces a runnable replication whose headline results are compared against the paper's
reported numbers, with an honest, attributable verdict on how closely they match.

This document is organized as: (0) what replication *is*, derived from first principles, and the
design constraints that fall out of it; (1) the pipeline that those constraints force; (2) each
stage fleshed out with alternatives and the reasoning for each; (3) architecture; (4) how to
evaluate the agent; (5) build order; (6) open questions.

---

## 0. First principles

### 0.1 What a paper actually is

A paper is a **lossy compression of an experiment**. The original experiment had:

| Component | Symbol | What it is |
|---|---|---|
| Data | D | The exact rows the authors used, after their filters |
| Procedure | P | Every computation from D to the reported numbers |
| Environment | E | Library versions, numerics, hardware |
| Randomness | R | Seeds, shuffles, initialization |
| Results | Y | The numbers in the tables and figures |

The paper reports Y faithfully and describes D, P, E in prose, with information loss.
It almost never reports R. Replication means: reconstruct D', P', E' from the prose, choose R',
run, get Y', and decide whether Y' ≈ Y.

**Consequence 1 — the error decomposes.** Y' − Y is the sum of:

- (a) data difference, D' ≠ D
- (b) procedure difference, P' ≠ P — from *our misreading* or from *the paper underspecifying*
- (c) randomness, R' ≠ R
- (d) environment, E' ≠ E (float precision, library defaults, BLAS)
- (e) bugs in our implementation
- (f) errors in the paper itself

The agent's job is not just to get Y' close to Y. It is to make **every term attributable**.
A replication that says "we got 0.31 instead of 0.41" is nearly useless. One that says "we got
0.31; the 0.10 gap is explained by using yfinance instead of CRSP, which drops delisted stocks,
and the paper's Table 1 row count is 18% higher than ours" is the actual deliverable.

Every stage of the pipeline exists to isolate one of these terms:

| Term | Isolated by |
|---|---|
| (a) data | Checkpoint our raw data against the paper's descriptive statistics before modeling |
| (b) procedure | A written spec with an explicit list of ambiguities and the default chosen for each |
| (c) randomness | Multiple seeds before declaring a mismatch |
| (d) environment | Pinned, era-appropriate environment with a lockfile |
| (e) bugs | Tests at each node, author code as an oracle where available, invariant checks |
| (f) paper error | What remains after (a)–(e) are ruled out, stated as such |

### 0.2 The time budget changes everything

1–2 hours means the agent **cannot try everything**. So it must:

- Decide early what it will *not* do (triage before any expensive work).
- Never repeat work (cache every download, every pipeline node, every environment build).
- Run small before running big (a 5-minute smoke run that fails saves a 40-minute full run).
- Bound every loop (a diagnose loop with no cap eats the whole budget on one mismatch).
- Parallelize what is independent (seeds, data downloads, environment build vs. data fetch).

### 0.3 What LLMs are good and bad at, and what that forces

Models are reliable at **bounded, verifiable steps** and unreliable at **long chains of implicit
state**. A 90-minute single conversation where the model "remembers" that it decided log returns
in minute 12 is fragile: context gets compacted, attention drifts, and nothing is auditable.

Therefore:

- **State lives on disk**, in structured files, not in the conversation. Every stage reads and
  writes files. A restart resumes from files.
- **Control flow is code**, not a model. A deterministic orchestrator decides which stage runs
  next, with what budget. The model does the work *inside* a stage.
- **Stage boundaries are verification points.** Each stage's output has a schema and a checker.
  Bad output does not flow downstream.
- **Structured outputs everywhere** a model produces something another program consumes.

### 0.4 The dominant failure in finance is a false positive

In ML replication, the usual failure is "we couldn't get the number." In finance, the more
dangerous failure is **getting the number for the wrong reason**: look-ahead bias, survivorship
bias, or a lag error will happily produce a beautiful Sharpe ratio that "matches" the paper.
So verification cannot be only "does Y' ≈ Y". It must include **adversarial checks that would
fail if the pipeline is leaking**, regardless of whether the numbers match.

### 0.5 You cannot improve what you cannot measure

An agent like this will be tuned by trial and error over prompts, tools, and stage logic.
Without a fixed set of papers with known-correct replications, every change is a guess.
**The benchmark is built before the agent**, not after.

### 0.6 Verification is cheaper than generation

Building a correct pipeline is hard; checking that a pipeline's output satisfies known
properties is easy. So the design leans on cheap independent checks: a second model reviewing
a spec, an invariant test on portfolio weights, a shuffle test on labels. Each check is
cheaper than the thing it checks, and each catches a distinct class of error.

---

## 1. The pipeline these principles force

```
paper.pdf ──► [1 Intake] ──► [2 Spec extraction] ──► [3 Feasibility triage]
                                     │                        │
                                     ▼                        ▼
                             spec.yaml (the contract)   plan.yaml (what we will
                                                         actually do, and why it
                                                         differs from the paper)
                                                              │
        ┌─────────────────────────────────────────────────────┘
        ▼
  [4 Environment] ──┐
                    ├──► [6 Implement] ──► [7 Run] ──► [8 Verify]
  [5 Data] ─────────┘                        ▲            │
                                             └── [9 Diagnose loop] (bounded)
                                                          │
                                                          ▼
                                             [10 Extend] ──► [11 Report]
```

Stages 4 and 5 are independent and run in parallel (principle 0.2).
Stages 1–3 are cheap and sequential; they exist to prevent wasting stages 4–9 (principle 0.2).
Stage 8 exists because of 0.1 and 0.4. Stage 9 is bounded because of 0.2.

---

## 1.5 Two tracks, one pipeline

Decision: the agent supports both quant-finance papers and CS/ML papers. The six error terms
in §0.1 are the same for both, so the pipeline and the orchestrator are shared. What differs is
*which term dominates* and therefore where each track spends its budget:

| | Finance track | CS/ML track |
|---|---|---|
| Dominant error term | (a) data: paywalled panels, survivorship, delisting | (c)+(d) randomness and environment: seeds, CUDA/torch versions |
| Binding constraint | Data access | Compute |
| Usual tier | A or B (data substitution) | A or D (reduced scale) |
| Author code | Rare; oracle mode is a bonus | Common; oracle mode is the default path |
| Dominant false positive | Look-ahead / survivorship leakage | Test-set contamination, tuning on test |
| Cheapest checkpoint | Paper's Table 1 descriptive stats | Dataset size and split counts; the paper's own baseline number |
| Success at reduced scale | Not usually needed | "Method beats baseline by the reported margin at matched compute" |

Everything track-specific is a **plugin behind a common interface**, so the orchestrator does
not know which track it is running:

| Plugin | Finance instances | CS/ML instances |
|---|---|---|
| Paper families (§2.2a) | anomaly, predictability, ML-returns, derivatives, LOB, RL, portfolio | classification, seq2seq/LM, RL, GNN, generative, systems (latency/throughput) |
| Data adapters (stage 5) | French library, FRED, yfinance, Kaggle, WRDS (credentials), crypto | Hugging Face datasets, torchvision, Kaggle, Papers With Code links, direct URLs |
| Data checkpoint | Table 1 moments, firm counts per year | split sizes, class balance, vocabulary size, sequence-length histogram |
| Leakage check | look-ahead audit, survivorship count, shuffle test | train/test near-duplicate scan, "tuned on test" detection, shuffle-label test |
| Tolerances | Sharpe ±0.1, t-stat ±0.5, alpha ±20% rel. | within the paper's reported std, or ±1 point if none reported; must beat the baseline by ≥ half the reported margin |
| Hypothesis library (stage 9) | lag, universe, weighting, delisting, returns definition, winsorization | LR schedule, augmentation, eval protocol (crop/tokenizer/beam), pretrained weight version, batch size vs. steps, mixed precision |
| Environment era | pandas/statsmodels era | CUDA + torch/TF/JAX pin; compiled ops; pretrained checkpoint availability |
| Reduced-scale strategy | subsample years | subset of data, fewer epochs, smaller model variant, compare against the paper's learning curves |

**Why one pipeline instead of two products.** The intake, spec, triage, verify, diagnose, and
report stages are identical in structure; only their plugin tables differ. Two codebases would
mean fixing every orchestrator bug twice, and the benchmark harness (§4) would diverge.

**Why the plugin boundary is exactly here.** Each row above is a place where a finance
convention and a CS convention would give a different answer to the same question. Anything not
in this table (budgeting, caching, the DAG, the report format, the grade rubric) is
track-agnostic by construction.

**The hybrid case.** ML-for-finance papers (Gu–Kelly–Xiu style) carry *both* dominant error
terms: paywalled panel data *and* seed/compute variance. They are the hardest papers in scope
and should be in the benchmark from the start, tagged as both tracks, so the two plugin sets
are exercised together.

**CS/ML-specific design ideas** (the rest of §2 leans finance; these fill the gap):

- *Oracle-first by default.* Clone the repo, build the era environment, run the authors' own
  eval script on their released checkpoint before training anything. Why: this distinguishes
  "the released model reproduces" from "training reproduces," which are different claims and
  cost minutes vs. hours.
- *Compute triage from the paper's own numbers.* Papers often report GPU-hours or steps. If
  reported compute exceeds the budget, plan.yaml picks the reduced-scale strategy up front and
  chooses the comparison: the paper's smallest reported configuration, or an early point on
  its learning curve. Why: matching a number the paper also reports at small scale is a real
  test; extrapolating from 5% of training is not.
- *Seeds are the unit of comparison.* Three seeds minimum; report mean and std; a claim is a
  Match if the paper's number is within our CI or our mean is within theirs. Why: (c) is the
  dominant term; one seed cannot say anything.
- *Baseline-relative verification.* Run the paper's baseline with the same pipeline and check
  the *gap*, not just the absolute number. Why: absolute numbers drift with hardware and
  library versions; the paper's contribution is the gap.
- *Eval-protocol extraction as a first-class spec field.* Metric definition, averaging
  (macro/micro), test split identity, decoding parameters, number of runs. Why: this is the
  CS equivalent of the finance universe filter and is where most CS mismatches come from.
- *Contamination scan.* Hash-based near-duplicate check between train and test; check whether
  a pretrained checkpoint's training data overlaps the test set. Why: the CS false positive.
- *Checkpoint provenance.* Record the exact URL and hash of any pretrained weights. Why: "the
  BERT-base checkpoint" has had several versions with different numbers.

---

## 2. Each stage, fleshed out

Each stage below has: **why it exists**, **inputs/outputs**, **design ideas** (with reasoning),
and **failure modes**.

### Stage 1 — Intake

**Why.** The agent can only reconstruct what it can read. PDFs lose table structure and
equations when naively text-extracted, and the paper's most valuable companions (author code,
supplementary material, cited data papers) are outside the PDF.

**Input.** arXiv ID, DOI, PDF, or URL.
**Output.** `paper/paper.md`, `paper/tables/*.csv`, `paper/figures/*.png`, `paper/equations.md`,
`author_code/` (if found), `references/` (cited data/method papers).

**Design ideas.**

- *Multi-modal extraction.* Convert PDF to markdown with a tool that preserves tables
  (Marker, Nougat, or GROBID for structure). Keep figures as images and let the model read the
  plots directly. Why: results are often only in figures; a text-only pipeline is blind to them.
- *Two extraction paths, compared.* Run two PDF extractors and diff the table cells. Where they
  disagree, send the page image to the model. Why: table extraction errors silently corrupt
  claims, and disagreement is a cheap detector.
- *Companion discovery.* Regex the paper for GitHub/GitLab/Zenodo/OSF/SSRN links, "code is
  available at", and dataset DOIs. Also search GitHub for the paper title. Why: author code, when
  it exists, collapses stages 6 and 9 from "reconstruct" to "port and verify".
- *Citation chasing for data.* If the data section says "we follow Fama and French (1993)",
  fetch that paper's data section. Why: filters and definitions are frequently defined by
  reference, and the agent will otherwise invent them.
- *Version awareness.* Note the paper's date and the arXiv version. Why: this drives the
  environment era in stage 4 and the data end-date in stage 5.
- *Paper family classification.* Tag the paper as one of a small set of families (see §2.2a).
  Why: each family has a template pipeline and known checkpoints, which shrinks the search space
  for every later stage.

**Failure modes.** Scanned PDFs with no text layer; tables split across pages; author repo
exists but is empty or diverged from the paper.

### Stage 2 — Spec extraction (the contract)

**Why.** Principle 0.1(b): procedure error comes from misreading and from underspecification.
Both are only detectable if the interpretation is written down *before* code exists. The spec
is the single artifact every later stage reads from and writes back to. It is also the point
of maximum leverage for a human: correcting one line in the spec is cheaper than debugging
the pipeline it produced.

**Input.** `paper/` from stage 1.
**Output.** `spec.yaml` validated against a schema.

**What the spec contains.**

```yaml
paper: {title, arxiv_id, year, family: cross_sectional_anomaly}
claims:                      # every quantitative result worth checking
  - id: T3.r2
    where: "Table 3, row 2"
    metric: sharpe_monthly
    value: 0.41
    sample: {start: 1963-07, end: 2019-12, universe: nyse_amex_nasdaq}
    priority: headline       # headline | secondary | descriptive
    tolerance: {abs: 0.10}   # per-metric, see stage 8
data:
  sources: [crsp_monthly, compustat_annual]
  universe: {exchanges: [NYSE, AMEX, NASDAQ], share_codes: [10, 11], min_price: 5}
  frequency: monthly
  preprocessing: {winsorize: {pct: 1, per: period}, lag_fundamentals_months: 6}
  splits: {train: [1963, 1990], test: [1991, 2019]}
method:
  signal: "..."              # formulas in LaTeX and, where possible, executable pseudocode
  portfolio: {sort: decile, weighting: value, rebalance: monthly, long_short: true}
  model: {type: ols | lgbm | mlp, hyperparameters: {...}}
  evaluation: [sharpe, alpha_ff3, t_stat]
baselines: [...]
ambiguities:                 # everything the paper does NOT say that the code needs
  - id: A1
    question: "Simple or log returns?"
    default: simple
    reason: "Standard in this literature; Table 2 magnitudes consistent with simple"
    sensitivity: low         # how much we expect this to move the headline
descriptive_stats:           # the paper's Table 1, used as the data checkpoint
  - {metric: n_firm_months, value: 2_100_000}
  - {metric: mean_mktcap_musd, value: 1_850}
```

**Design ideas.**

- *One high-effort call over the whole paper with structured output.* Why: the spec needs
  global consistency (the universe in the data section must match the sample in the claims), and
  chunked extraction loses that. Modern context windows fit any paper; use them.
- *Extract-then-critique.* A second call, with a different prompt ("you are a referee; find every
  place this spec contradicts or omits something in the paper"), revises the spec. Why: principle
  0.6, and extraction and criticism are different cognitive modes; a single pass systematically
  under-reports ambiguities.
- *N independent extractions, diffed.* Run the extraction 2–3 times with different seeds or
  effort levels; fields where they disagree are, by construction, ambiguities. Why: this is the
  cheapest possible ambiguity detector and needs no human.
- *Claims tied to their evidence.* Every claim stores the table/figure it came from and the
  page image. Why: verification (stage 8) and the report must be auditable back to the paper.
- *Equations to executable form.* Convert key formulas to SymPy or plain Python and unit-test
  them on toy inputs in the spec itself. Why: a formula transcribed in LaTeX can be misread
  in stage 6; a formula that already runs cannot.
- *Ambiguity defaults from a learned library.* Maintain a knowledge base of conventional
  defaults per paper family ("Fama-French universe: NYSE/AMEX/NASDAQ, share codes 10/11, lag
  accounting data 6 months"). Why: most ambiguities are conventions, and conventions are stable
  across papers; learning them once turns a guess into a citation.
- *Sensitivity tagging.* Each ambiguity gets a predicted sensitivity (low/medium/high). Why:
  the diagnose loop (stage 9) should test high-sensitivity ambiguities first.
- *Spec diff against author code.* If author code exists, have the model read it and produce a
  second spec, then diff the two. Why: disagreement between paper and code is one of the most
  common findings in replication studies, and the code is usually what produced the numbers.
- *Human checkpoint here, if anywhere.* Show the spec, especially the ambiguity list, to the
  user before stage 4. Why: it is the highest-leverage, lowest-cost place for a correction.

**Failure modes.** Over-confident extraction with no ambiguities listed; claims extracted with
wrong units (annualized vs monthly); tolerances set naively.

#### 2.2a Paper families (a design idea that cuts across every stage)

The method space of quant papers is not open-ended. A small taxonomy covers most of it, and each
family comes with a template task graph, known checkpoints, and a usual-suspects list:

| Family | Typical pipeline | Cheap checkpoint | Usual suspect |
|---|---|---|---|
| Cross-sectional anomaly / factor | signal → sort → portfolios → alpha regression | Table 1 stats; decile 1 vs 10 raw return | lag alignment, weighting, universe |
| Time-series predictability | predictor → predictive regression → OOS R² | in-sample R² and t-stat | overlapping returns SE, OOS window |
| ML return prediction (Gu–Kelly–Xiu style) | features → rolling train → OOS R² → portfolios | feature count, OOS R² of OLS baseline | rolling window, target scaling |
| Derivatives / vol models | market data → calibration → pricing error | ATM implied vol fit | day-count, rate curve |
| Microstructure / LOB | tick data → features → classifier | class balance, horizon | label definition, sampling |
| RL / trading agents | env → policy → backtest | random-policy baseline | reward leakage, cost model |
| Portfolio optimization | returns → covariance → weights → OOS | equal-weight baseline | shrinkage, rebalance |
| CS/ML (non-finance) | dataset → model → metric | dataset size, baseline number | preprocessing, eval protocol |

Why this matters: a family template turns "write a pipeline from scratch" into "fill in a
template," which is faster, more reliable, and easier to verify. The family also chooses the
default tolerances and the diagnose-loop ordering.

### Stage 3 — Feasibility triage

**Why.** Principle 0.2. The 1–2 hour budget is only achievable if the agent decides, in minutes,
what is possible and what "success" means for *this* paper. Without triage, the agent discovers
at minute 50 that the data is paywalled and the run is wasted.

**Input.** `spec.yaml`.
**Output.** `plan.yaml`: target claims, data substitutions, per-stage budgets, success criteria.

**Tiers.**

| Tier | Data | Compute | What we attempt | What "success" means |
|---|---|---|---|---|
| A | Open (French library, FRED, yfinance, Kaggle, crypto exchanges, public GitHub data) | Minutes, CPU | Exact replication | Headline claims Match within tolerance |
| B | Paywalled but substitutable (CRSP → yfinance/Polygon/Sharadar; Compustat → public fundamentals) | Minutes to ~30 min | Approximate replication | Same sign and order of magnitude; gaps attributed |
| C | Proprietary/unavailable (prop tick data, private LOB) | Any | Replicate on synthetic/proxy data | Pipeline mechanics verified, no claim on numbers |
| D | Any | GPU-days | Reduced scale (subset, fewer epochs) | Trend verified, e.g. "model beats baseline" |

**Design ideas.**

- *Data availability probe.* For every source in the spec, actually attempt a tiny fetch (one
  ticker, one month) before committing. Why: "yfinance has this" is a belief; a 200 OK is a fact.
- *Compute estimation from the spec.* Estimate rows × features × epochs and compare to the
  budget. Why: a model that "usually trains in 20 minutes" is a guess; an arithmetic estimate is
  a plan.
- *Claim budgeting.* Rank claims by priority × feasibility and pick the subset that fits. Why:
  replicating the headline number plus two robustness rows is worth more than half-finishing all
  twelve rows of Table 5.
- *Substitution registry.* A table of known data substitutions with their known consequences
  ("yfinance lacks delisting returns → long-short spreads biased upward by ~X bps/month in
  small caps"). Why: this is what turns a Tier B gap from "unexplained" into "attributed."
- *Explicit success criteria written down before running.* Why: otherwise the agent (or the
  human) will rationalize whatever comes out. Pre-registration for replication.
- *Budget allocation as a table in plan.yaml.* E.g. env 5 min, data 15, implement 30, run 20,
  verify 5, diagnose 20, report 5. Why: the orchestrator enforces this; the model does not
  get to decide to spend 60 minutes on data.

**Failure modes.** Over-optimistic tier assignment; substitution with unknown consequences.

### Stage 4 — Environment

**Why.** Principle 0.1(d). Library defaults change (pandas resampling, sklearn solvers, PyTorch
initialization) and a 2018 paper's code often does not run on 2026 libraries. An unreproducible
environment makes the whole replication unreproducible.

**Input.** `spec.yaml`, `author_code/`.
**Output.** A container image or venv, a lockfile, a smoke-test log.

**Design ideas.**

- *Era-pinned base images.* Pre-built images per year (py3.7+pandas0.25+torch1.4 for "2019",
  etc.). Why: building from scratch is slow and flaky; a pre-built image starts in seconds and
  matches the paper's likely environment.
- *Dependency inference from three sources.* Author requirements file > imports in author code
  > method names in the paper ("we use LightGBM"). Why: each is a fallback for the previous.
- *Build in parallel with data (stage 5).* Why: they are independent, and both are I/O-bound.
- *Smoke test the environment, not just the install.* Import everything, run one matrix
  multiply on GPU, load one parquet. Why: an install that succeeds but crashes on first use
  costs a stage.
- *Image cache keyed by lockfile hash.* Why: repeated replications of similar papers should
  reuse the image.
- *Network policy.* Allow only known data hosts and package indexes. Why: sandboxing, and it
  makes the data lineage auditable.

**Failure modes.** GPU driver mismatch; a dependency only available for old Python; author
code that needs a compiled extension.

### Stage 5 — Data

**Why.** Principle 0.1(a): data difference is the largest error term in finance replications
and the one most often left unexamined. Principle 0.4: leakage enters here.

**Input.** `spec.yaml`, `plan.yaml`.
**Output.** Parquet/DuckDB cache, `data_manifest.json` (source, params, hash, row counts),
`data_checkpoint.md` comparing our descriptive stats to the paper's.

**Design ideas.**

- *Source adapters with a common interface.* `fetch(source, params) -> DataFrame`, one adapter
  per source (French library, FRED, yfinance, Kaggle, WRDS if credentials given, a crypto
  exchange, a local file). Why: the agent should call a typed tool, not write scraping code from
  scratch each time; typed tools are cacheable, rate-limited, and auditable.
- *Content-addressed cache.* Key = hash(source, params). Why: retries and diagnose loops must
  never re-download; and the hash is part of the reproducibility record.
- *Schema contracts.* Each adapter declares columns, dtypes, and invariants (dates monotone, no
  duplicate keys, prices > 0). Validated on every load. Why: silent schema drift (a column
  renamed by a provider) is otherwise found in stage 9 at ten times the cost.
- *Descriptive-stats checkpoint against the paper's Table 1.* Row counts, date range, means,
  standard deviations, number of firms per year. Why: this is the cheapest and most decisive
  test of D' ≈ D. If it fails, stop and diagnose data before touching the model.
- *Look-ahead audit as a mechanical check.* Every feature carries an `available_at` timestamp;
  the join to returns asserts `available_at < decision_time`. Why: principle 0.4. The single
  most common way a finance replication produces a fake match.
- *Point-in-time discipline.* Where the source supports it, fetch as-of data; where it does
  not, apply the paper's stated lag (typically 6 months for annual fundamentals). Why:
  restatements and late filings are look-ahead bias in disguise.
- *Synthetic data generator for Tier C.* Given Table 1 statistics, generate a dataset with
  matching moments and the paper's stated structure. Why: lets the pipeline be built and
  mechanically verified even when real data is unobtainable, and gives the user a working
  pipeline to point at real data later.
- *Data quirks knowledge base.* Known issues per source ("yfinance adjusts for splits but the
  adjustment changes retroactively"; "French library returns are in percent"). Why: these are
  learned once and otherwise cost a diagnose iteration every time.
- *Survivorship check.* Count securities that disappear during the sample. If zero, the source
  is survivorship-biased and the report must say so. Why: same as look-ahead; a fake match.

**Failure modes.** Rate limits; provider changes; percent vs decimal; timezone off-by-one on
daily data; missing delisting returns.

### Stage 6 — Implement

**Why.** Principle 0.1(e). Bugs are the term the agent has the most control over, and the
one where structure (small nodes, tests, an oracle) has the highest payoff.

**Input.** `spec.yaml`, `plan.yaml`, data cache, environment.
**Output.** `pipeline/` as a DAG of nodes, each with a test; `run.py` entrypoint.

**Design ideas.**

- *Pipeline as a DAG of pure nodes.* `load → features → signal → portfolios → evaluate → tables`.
  Each node is a function `(inputs, config) -> outputs`, with outputs cached by hash of inputs
  and config. Why: the diagnose loop changes one config value and re-runs only what depends on
  it; this is the difference between a 30-second and a 20-minute iteration.
- *Family template as the starting point.* Instantiate the template for the paper's family and
  have the model fill in the paper-specific nodes. Why: §2.2a; less freedom, fewer bugs.
- *Author code as an oracle, run first.* If it exists, run it as-is in the era environment. Three
  outcomes: it reproduces the paper (then port node by node, checking intermediates); it runs
  but does not reproduce (a finding); it does not run (fall back to reconstruction, but keep the
  code as a reference for ambiguities). Why: an oracle turns reconstruction into verification.
- *Golden-fixture tests per node.* Tiny hand-checkable inputs with known outputs. Why: catches
  the off-by-one-period and sign errors that dominate finance bugs.
- *Invariant tests.* Portfolio weights sum to one; long and short legs have equal absolute
  weight; no NaN in returns after the first period; deciles have roughly equal counts; feature
  timestamps precede return timestamps. Why: principle 0.6; invariants are cheap and catch
  whole classes of bugs.
- *Two independent implementations of the critical formula.* Have the model write the signal
  computation twice (vectorized pandas and a naive loop) and assert equality on a sample. Why:
  N-version programming; the two are unlikely to share the same bug.
- *Executable equations from the spec.* The SymPy/Python formulas from stage 2 are imported,
  not re-derived. Why: one source of truth.
- *Boring libraries by default.* pandas, numpy, statsmodels, scikit-learn, PyTorch. Why: the
  model's prior knowledge of their APIs is accurate, so fewer hallucinated arguments.
- *Config-driven everything.* Every choice from the ambiguity list is a config key with the
  default from the spec. Why: stage 9 flips config keys, not code.

**Failure modes.** Off-by-one lags; per-period vs full-sample standardization; groupby
alignment bugs; silent NaN propagation.

### Stage 7 — Run

**Why.** Principle 0.2. Running is where the wall clock goes; it must be staged and observable.

**Design ideas.**

- *Smoke config first.* One year, one seed, one epoch. Must finish in under five minutes. Why:
  most pipeline bugs surface on any input; find them cheaply.
- *Progressive scaling.* Smoke → 10% → full, or the largest config the remaining budget allows.
  Why: gives a usable partial result if the budget runs out.
- *Seeds in parallel.* For any stochastic method, run 3 seeds concurrently. Why: principle
  0.1(c); one seed cannot distinguish variance from mismatch.
- *Checkpointing.* Long training saves state; a budget kill still leaves a usable model. Why:
  wall-clock kill is a hard constraint; do not lose everything at minute 119.
- *Structured run log.* Config hash, data hash, seed, environment hash, wall time, every metric,
  at every node. Why: the report and the diagnose loop both read this; and it makes any run
  reproducible by hash.
- *Resource watchdog.* Kill on memory or time overrun with a clean error, not an OOM. Why: a
  clean error is diagnosable.

### Stage 8 — Verify

**Why.** Principles 0.1 and 0.4. Verification must both compare numbers *and* try to break the
result.

**Input.** `results.json`, `spec.yaml`.
**Output.** `verification.json`: per claim, outcome plus explanation.

**Outcomes per claim.**

| Outcome | Definition |
|---|---|
| Match | Within the claim's tolerance |
| Consistent | Same sign and order of magnitude; the gap is attributable to a recorded substitution or scale reduction |
| Mismatch | Different conclusion, or gap unattributable |
| Untested | Skipped per plan.yaml, with the reason |

**Design ideas.**

- *Tolerances per metric type, set in the spec, before running.* Sharpe ±0.1, t-stat ±0.5,
  alpha ±20% relative, accuracy ±1 point, R² ±0.5 points. Why: pre-registration; otherwise
  tolerances get chosen to fit the result.
- *Statistical, not just numeric, comparison.* Bootstrap a confidence interval on our metric and
  ask whether the paper's value lies inside it. Why: a Sharpe over 30 years has a standard
  error; "0.38 vs 0.41" is not a mismatch if the CI is ±0.15.
- *Figure comparison.* Re-plot our result in the paper's format and overlay on the paper's
  figure image; have the model judge. Why: many claims are only in figures, and shape agreement
  (cumulative return curves, decile monotonicity) is informative even when numbers are off.
- *Shuffle test.* Re-run with labels/returns shuffled within period. If the result survives,
  there is leakage. Why: principle 0.4; a fake match detector that works even when the number
  matches.
- *Decile monotonicity check.* For sort-based papers, verify returns increase across deciles.
  Why: it is the structural claim behind the headline number and is more robust than the number.
- *Transaction cost and turnover sensitivity.* Report the result with 0, 10, 25 bps costs. Why:
  many anomalies vanish; the report should show where.
- *Subperiod stability.* Split the sample in halves. Why: a result driven by one decade is a
  different finding than a stable one.
- *Independent reviewer call.* A separate model call reviews the pipeline code against the
  spec, looking specifically for lag and leakage errors, without seeing the results. Why:
  principle 0.6, and blinding the reviewer to the numbers removes confirmation bias.

**Failure modes.** Tolerances too loose; comparing annualized to monthly; declaring Match on a
leaking pipeline.

### Stage 9 — Diagnose loop (bounded)

**Why.** Principle 0.1: attribution. Principle 0.2: bounded. When a headline claim is Mismatch,
the job is to find which error term explains it, in a fixed number of iterations.

**Design ideas.**

- *Hypothesis library per family, ranked by prior probability × cheapness.* For cross-sectional
  papers: (1) lag alignment, (2) universe filters, (3) equal vs value weighting, (4) delisting
  returns, (5) simple vs log returns, (6) winsorization per-period vs pooled, (7) rebalance
  frequency, (8) seed variance. Why: most mismatches have one of a dozen causes; test them in
  order rather than reasoning from scratch.
- *Ambiguity flips first.* The spec's high-sensitivity ambiguities are the first hypotheses.
  Why: they were flagged as uncertain for a reason.
- *Intermediate checkpoints, not just the final number.* Compare every intermediate the paper
  reports (Table 1, decile counts, univariate sorts) to ours; the first divergence localizes the
  bug. Why: bisection over the pipeline is far cheaper than re-reading the code.
- *One config change per iteration, re-run only affected nodes.* Why: the DAG cache from stage 6
  makes each iteration cheap; changing two things at once destroys attribution.
- *Hard caps.* E.g. 3 iterations or 20 minutes, whichever first. Then stop and report the best
  attribution so far. Why: principle 0.2; an unbounded loop is the most common way agentic
  systems blow their budget.
- *Record every hypothesis and result, including the failed ones.* Why: they go in the report
  as "things we ruled out," which is real information.

### Stage 10 — Extend (only with leftover budget)

**Why.** Once the pipeline exists, extensions are nearly free and are often the most valuable
output for a practitioner.

**Design ideas.**

- Out-of-sample extension past the paper's end date (the single most requested thing).
- Cost-adjusted results; alternate universes (large-cap only); parameter sensitivity grid.
- Alternate data source (if Tier A and B sources both exist, run both and compare).
- Why these and not others: each is a config change on the existing DAG, so each costs minutes.

### Stage 11 — Report

**Why.** The deliverable is the attribution, not the number. The report is how a reader who was
not present judges the replication.

**Structure of `REPORT.md`.**

1. Tier and every deviation from the paper, first, in one table.
2. Claims table: paper value | ours | CI | outcome | attribution note.
3. Adversarial checks: shuffle test, look-ahead audit, survivorship, cost sensitivity.
4. Ambiguities and the defaults chosen; which ones were tested in stage 9.
5. Diagnose history: hypotheses tried, ruled out, remaining.
6. Extensions.
7. Reproducibility block: command, data hashes, environment lockfile, run log hashes.
8. Confidence grade for the replication as a whole, with the rule that produced it.

**Design ideas.**

- *Grade rubric fixed in advance.* E.g. A: all headline claims Match, adversarial checks pass.
  B: Consistent with attributed gaps. C: mechanics verified, numbers not. F: leakage detected or
  pipeline does not run. Why: the grade must mean the same thing across papers.
- *Machine-readable twin.* `report.json` alongside the markdown, for the benchmark harness.
- *Everything links back.* Each claim links to the page image; each number links to the run log.

---

## 3. Architecture

### 3.1 Components

```
┌──────────────────────────────────────────────────────────────┐
│ Orchestrator (deterministic Python)                          │
│  - stage sequence, per-stage budgets, retries, kill switch   │
│  - reads/writes: spec.yaml, plan.yaml, results.json, logs    │
│  - parallelizes independent stages (env ∥ data; seeds)       │
└───────┬──────────────────┬──────────────────┬────────────────┘
        │                  │                  │
   ┌────▼─────┐      ┌─────▼──────┐     ┌─────▼──────┐
   │ Extractor │      │ Builder    │     │ Verifier   │
   │ (1 call,  │      │ (agentic   │     │ (agentic + │
   │ structured│      │  loop w/   │     │  reviewer  │
   │ output)   │      │  tools)    │     │  call)     │
   └───────────┘      └─────┬──────┘     └────────────┘
                            │
                   ┌────────▼─────────┐
                   │ Sandbox (Docker) │
                   │ repo + data cache│
                   │ net: allowlist   │
                   │ cpu/gpu/time cap │
                   └──────────────────┘
        Tools exposed to the model inside the sandbox:
        bash, read/write/edit, fetch_data(source, params),
        run_node(name, config), compare_claim(id), ask_user(q)
```

**Why the orchestrator is code, not a model.** Principle 0.3. Budgets, retries, and stage
order must be enforced, not suggested. And when something goes wrong at 2 a.m., you debug
Python, not a transcript.

**Why some actions are dedicated tools rather than bash.** `fetch_data`, `run_node`, and
`compare_claim` are typed tools so the harness can cache them, log them, rate-limit them, and
render them. `bash` stays for breadth. Rule: promote to a tool anything you need to gate,
cache, audit, or parallelize.

**Why state is on disk.** Principle 0.3. Resumability, auditability, and it lets the
benchmark harness (§4) inspect every intermediate.

### 3.2 Model usage

- **Spec extraction**: one call, PDF as a document block, structured output against the spec
  schema, highest effort. The most important call in the system; spend on it.
- **Builder loop**: agentic loop with tools; adaptive thinking; context editing to clear stale
  tool results as the loop runs long. Sub-tasks that are reading-heavy (scan an author repo,
  read data docs) go to a cheaper subagent so the main loop's context stays lean.
- **Reviewer**: separate call, blinded to results, reads code against spec.
- **Judge (benchmark only)**: grades a replication against a rubric.

**Why separate calls with separate prompts** rather than one long session: principle 0.3 and
0.6. Extraction, building, and criticism are different jobs; a single context doing all three
drifts and cannot be blinded.

### 3.3 Engine choice

Two viable paths for the builder loop:

1. **Claude Code / Claude Agent SDK as the engine**, with one skill per stage and the
   orchestrator invoking sessions. Fastest to a working MVP because file tools, bash, subagents,
   and context management are built in.
2. **A custom tool-runner loop** on the Anthropic SDK. More control over gating, caching, and
   cost; more code to own.

Recommendation: start with 1 for the MVP; move stages to 2 when they need tighter control.
Why: the hard problems here are the spec, the data, and verification, not the loop. Do not
spend the first month on the loop.

### 3.4 Human-in-the-loop points

Exactly two, both optional and both time-boxed:

- After stage 2: approve/correct the spec (highest leverage).
- After stage 3: approve the tier and substitutions (prevents a wasted run).

Why not more: each pause costs wall clock; why not fewer: these two decisions are where a
human's five minutes saves the agent an hour.

### 3.5 Cross-run memory

A knowledge base that grows across replications:

- Ambiguity defaults per family (stage 2).
- Data source quirks and substitution consequences (stages 3, 5).
- Hypothesis library with observed hit rates (stage 9).
- Environment images per era (stage 4).

Why: most of what makes a human replicator fast is accumulated convention knowledge. The agent
should accumulate it too, in files, reviewed by a human before promotion.

### 3.6 Observability and cost

- Trace every model call: stage, tokens, cost, wall time. Per-paper cost is a benchmark metric.
- Prompt caching on the static system prompts and the paper document (it is reread by every
  stage). Why: the paper is the largest stable prefix; caching it is the single biggest cost win.

---

## 4. Evaluating the agent (build this first)

**Why first.** Principle 0.5.

- **Ground truth set.** 10–20 papers with public, trusted replications.
  Finance: Open Source Asset Pricing (Chen & Zimmermann) publishes code and results for 200+
  anomaly signals; Jensen, Kelly & Pedersen's factor replication has public code; Kenneth
  French's library covers the classic factor papers.
  CS/ML: PaperBench (OpenAI) and CORE-Bench are existing paper-replication benchmarks with
  rubrics to borrow.
- **Stratify by tier and family.** Why: an agent that aces Tier A momentum papers and fails
  everything else should look like that in the numbers.
- **Metrics per paper.** Headline-claim Match rate; Consistent rate; wall time; cost; diagnose
  iterations; adversarial checks passed; and **human agreement with the agent's verdict** (the
  agent saying "Match" when a human says "leakage" is the worst outcome and must be counted).
- **Rubric grading.** Per paper, a checklist of things a correct replication must contain
  (correct universe filter, correct lag, correct weighting, ...), graded by a model judge with
  human spot checks. Why: the final number can be right by accident; the rubric checks the path.
- **Stage-level evals.** Spec extraction alone (against hand-written specs); data checkpoint
  alone; verify alone (given a known pipeline, does it produce the right outcomes?). Why:
  end-to-end evals are slow and confound causes; stage evals tell you *which* stage to fix.
- **Regression discipline.** Every change to a prompt, tool, or stage runs the set.

---

## 5. Build order

1. **Three replications by hand**, Tier A, one per track plus the hybrid: a momentum or
   factor paper on French library data (finance); a small classification or LM paper with a
   public repo and a released checkpoint (CS); an ML-on-returns paper with open data (hybrid). Record every step and decision. Why: this produces the spec schema, the family
   templates, the hypothesis library, and the first three benchmark entries from reality
   instead of imagination.
2. **Spec extraction alone.** PDF → `spec.yaml`, evaluated against the three hand specs. Why:
   highest leverage; independently useful; and it validates the schema before anything
   depends on it.
3. **Data adapters + checkpoint + look-ahead audit** for Tier A sources. Why: the data stage
   is the biggest error term and is fully testable without an agent.
4. **Verify stage** against the hand-built pipelines. Why: it is needed before the builder loop
   exists, so the loop has something to aim at.
5. **Builder loop** for one paper end to end with the time budget enforced.
6. **Benchmark harness** over three papers; grow to ten.
7. Then: Tier B substitutions, author-code oracle mode, diagnose library, extension stage,
   cross-run memory.

---

## 6. Known hard problems

- **Data access is the wall.** Most finance papers use CRSP/Compustat. Without WRDS, the agent
  is doing Tier B at best. Either the user supplies WRDS credentials or the product is
  explicitly "approximate replication on open data." Decide this early; it shapes everything.
- **Papers underspecify.** Replication studies routinely find that unspecified choices swing
  results. The ambiguity list is the product, not a failure.
- **1–2 hours is a hard constraint for ML papers.** Multi-hour training is Tier D. Say so.
- **"Match" needs pre-registered tolerances.** Otherwise the agent grades itself generously.
- **Leakage produces false successes.** The shuffle test and look-ahead audit are not optional.
- **Model-graded evals can be fooled by the same errors that fool the model.** Keep human spot
  checks in the benchmark loop.

---

## 7. Open questions

1. ~~Finance-first or CS-first?~~ **Decided: both tracks, one pipeline, track-specific plugins
   (§1.5).** Build-order consequence: the three hand replications in §5 should be one finance,
   one CS, one hybrid ML-for-finance paper.
2. WRDS/CRSP access: yes, no, or "the user brings credentials"?
3. Operator-in-the-loop (you correcting the spec mid-run) or fully unattended? The first is
   far easier to make good and is the right MVP.
4. Which three papers for the hand replications?
5. Engine: Claude Code sessions per stage (fast MVP) vs. custom loop (control)?
