# Paper Replication Agent — Design (first principles)

**Goal.** An agent that takes a quant-finance or CS paper and produces a runnable replication
whose headline results are compared against the paper's reported numbers, with an honest,
attributable verdict on how closely they match. Users: a quant firm reproducing research before
acting on it, and an ML research lab reproducing a paper before building on it. The
productivity claim is 20–30 researcher-hours replaced by 1–2 hours of turnaround and about 15
minutes of human attention; that is where the time budget comes from (§0.2), and it has to
hold at production quality, not demo quality (§3.8). It is one product and one orchestrator
with several model roles, not one long conversation and not a society of agents (§3.7).

*Revised 2026-09-14 after an adversarial review. The attacks and their dispositions are in
`DESIGN_REVIEW.md`; the pre-review text is `DESIGN.v1.md`.*

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
- (g) a fragile finding: the paper's number is real on its data but sensitive to defensible
  conventions, so any re-implementation lands somewhere else (this is not (b); the paper is
  not wrong, it is one point in a range)

and one term that sits *upstream* of all of them:

- (0) claim error, Y_read ≠ Y: we mis-transcribed the target (wrong row, wrong units,
  annualized vs monthly, v1 vs v3 of the arXiv paper). Every later stage compares against
  Y_read, so an error here is invisible to everything downstream.

The agent's job is not just to get Y' close to Y. It is to make **every term attributable**.
A replication that says "we got 0.31 instead of 0.41" is nearly useless. One that says "we got
0.31; the 0.10 gap is explained by using yfinance instead of CRSP, which drops delisted stocks,
and the paper's Table 1 row count is 18% higher than ours" is the actual deliverable.

**Consequence 1a — attribution is not free.** One run gives one observed gap and seven unknown
terms; the sum is not identifiable. A term can be attributed only by (i) an intervention that
changes that term alone and moves the gap, or (ii) independent evidence (author code, an
intermediate the paper reports, a documented data-source difference). What is left after the
attributed terms is **unexplained**, not "paper error": (f) is claimed only with positive
evidence, such as an internal inconsistency in the paper's own tables or author code that
disagrees with the paper's text.

Every stage of the pipeline exists to isolate one of these terms:

| Term | Isolated by |
|---|---|
| (0) claim error | Every claim value re-read from the page image by an independent call; units and precision recorded |
| (a) data | Checkpoint our raw data against the paper's descriptive statistics before modeling |
| (b) procedure | A written spec with an explicit list of ambiguities and the default chosen for each |
| (c) randomness | Multiple seeds before declaring a mismatch |
| (d) environment | Pinned, era-appropriate environment with a lockfile |
| (e) bugs | Tests at each node, author code as an oracle where available, invariant checks |
| (f) paper error | Positive evidence only: an internal inconsistency in the paper, or author code disagreeing with the text |
| (g) fragility | The convention grid (stage 9): the headline across all defensible defaults, reported as a range |
| (unexplained) | What remains; reported as such, never folded into (f) |

### 0.2 The time budget changes everything

**Where the number comes from.** The target is a researcher's 20–30 hours turned into a result
they can read in the same working session. 1–2 hours of turnaround is the point at which the
product changes how work is done rather than merely speeding it up. Corollaries: cost is not
the binding constraint in production (a $100 run replacing $2,500 of researcher time is cheap;
cost binds in the benchmark, where hundreds of runs happen); a correct report at 2.5 hours
beats a wrong one at 1.5; so the cap is a target with a configurable hard kill (default 4 h),
not a cliff at minute 120. The 1–2 h target still forces everything below, because the design
that hits it is the design that also hits 4 h reliably.

1–2 hours means the agent **cannot try everything**. So it must:

- Decide early what it will *not* do (triage before any expensive work).
- Never repeat work (cache every download, every pipeline node, every environment build).
- Run small before running big (a 5-minute smoke run that fails saves a 40-minute full run).
- Bound every loop (a diagnose loop with no cap eats the whole budget on one mismatch).
- Parallelize what is independent (seeds, data downloads, environment build vs. data fetch).

Three things the budget must also say, or it is not a budget:

- **Cost is a co-constraint.** An hour of agentic loop with high-effort calls is tens of
  dollars; a 20-paper benchmark run on every prompt change is thousands. Every stage has a
  token budget next to its minute budget, and per-paper cost is reported with the result.
- **Verification is inside the budget.** Extra seeds, the convention grid, the leakage tests,
  and the bootstrap are runs; they are not free because they are "checks." The budget table
  in plan.yaml allocates them explicitly, and a plan that cannot afford them says which it
  dropped.
- **Two modes, one pipeline.** *Interactive* (an operator is present; 1–2 h; the two human
  checkpoints are live) and *batch* (unattended; the checkpoints auto-approve and record the
  question; more seeds and a fuller grid; a cost cap rather than a wall-clock cap). The
  benchmark runs in batch mode, so batch mode exists from day one.

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

### 0.7 A replication is a hypothesis test run by an interested party

The agent generates the result and grades it. Every incentive points toward "Match." So the
design must control the agent's own false-positive rate the way a pre-registered study does,
and pre-registration is a *commitment* property (cannot be undone), not a *timing* property
(decided early):

- **Freeze before running.** The orchestrator hashes `spec.claims` and `plan.success_criteria`
  before stage 7 and refuses any later edit to tolerances or target claims. A changed
  tolerance is a new run, reported as such.
- **Tolerances are derived, not chosen.** A tolerance is k standard errors of the statistic on
  the paper's sample length (Sharpe: SE ≈ sqrt((1 + SR²/2)/T)), widened by the paper's
  reporting precision. Nobody, human or model, picks a number.
- **Search is not attribution.** Trying conventions until the number matches is p-hacking.
  Stage 9 runs the convention grid as a *sensitivity analysis* and reports the whole range;
  a configuration is called "the paper's" only with evidence independent of the match.
- **Leakage tests are mechanical.** A model reviewer shares the generator's blind spots. The
  checks that matter (future-perturbation, `available_at` audit, survivorship count) are code.
- **The benchmark's north star is calibration**, not Match rate: when the agent says Match,
  how often does a human agree; and how often does it say Match on a pipeline a human calls
  leaking. Match rate rewards loose tolerances; calibration punishes them.

### 0.8 Missing information is filled from conventions or artifacts, never from reasoning

What the paper does not say cannot be inferred by thinking harder. It can only be filled from
(i) the literature's conventions for that paper family, (ii) author code or data, or (iii) an
explicit guess labeled as one. So the conventions knowledge base (§3.5) is not a nice-to-have
that accumulates later; it is the core asset, seeded by hand from the replication literature
(Chen–Zimmermann's signal documentation, Jensen–Kelly–Pedersen's appendix, Hou–Xue–Zhang)
before the first agent run. Build order step 0.

The same principle says what this document actually describes: three separable products.
A **spec extractor** (paper → auditable claims and ambiguities), a **replication engine**
(spec + data → pipeline with leakage guarantees), and a **verdict engine** (results vs. claims
→ calibrated outcome). Each is useful alone; the pipeline is their composition, and the MVP
can ship them in that order.

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
| Tolerances | derived from the statistic's SE on the paper's sample (§0.7, stage 8) | derived from the paper's seed std, or a binomial SE on the test set if none is reported; the method–baseline gap must exceed the pooled seed SE |
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

- *Multi-modal extraction, with a clear division of labor.* The spec call (stage 2) reads the
  PDF directly as a document block. The markdown conversion (Marker, Nougat, or GROBID for
  structure) exists for the *other* consumers: grep-able text for companion discovery,
  per-page images for claim evidence, table CSVs for the descriptive-stats checkpoint, and
  figures as images so the model can read plots. Why: results are often only in figures, a
  text-only pipeline is blind to them, and the earlier draft had two sources of truth
  (paper.md for stage 1, the PDF for stage 2) without saying which the spec is checked
  against. The PDF is the source; paper.md is derived.
- *Which version.* An arXiv ID names several versions and an SSRN paper has revisions; the
  numbers change between them. Intake records the exact version and every claim cites it.
  Why: term (0) in §0.1.
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
    evidence: paper/pages/p14.png   # the page image the value was read from
    metric: sharpe
    units: {period: monthly, annualized: false, scale: decimal}
    value: 0.41
    reported_precision: 0.005       # half the last printed digit
    n_periods: 678                  # drives the derived tolerance (principle 0.7)
    sample: {start: 1963-07, end: 2019-12}
    method_variant: vw_decile       # claims differ by variant; the method block is not global
    priority: headline       # headline | secondary | descriptive
    # no tolerance here: it is derived by code into plan.yaml and frozen before stage 7
data:
  sources: [crsp_monthly, compustat_annual]
  universe: {exchanges: [NYSE, AMEX, NASDAQ], share_codes: [10, 11], min_price: 5}
  frequency: monthly
  preprocessing: {winsorize: {pct: 1, per: period}, lag_fundamentals_months: 6}
  splits: {train: [1963, 1990], test: [1991, 2019]}
method:
  signal: "..."              # formulas in LaTeX and, where possible, executable pseudocode
  model: {type: ols | lgbm | mlp, hyperparameters: {...}}
  evaluation: [sharpe, alpha_ff3, t_stat]
  variants:                  # one paper, several tables, several procedures
    vw_decile: {sort: decile, weighting: value, rebalance: monthly, long_short: true}
    ew_decile: {sort: decile, weighting: equal, rebalance: monthly, long_short: true}
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
  effort levels; fields where they disagree are candidate ambiguities. Why: cheap and needs no
  human. Caveat: disagreement measures *model* instability, not *paper* ambiguity. It misses
  exactly the case that matters most, a convention the model is confidently and consistently
  wrong about. So it supplements the family's known-ambiguity checklist (§0.8); it does not
  replace it.
- *Claim values re-read from the page image.* A separate, cheap call is given only the page
  image and the claim's `where`, and asked for the value and units. Disagreement with the
  extracted claim blocks the spec. Why: term (0) in §0.1; a wrong target makes every later
  stage worthless, and this is the only stage that can catch it.
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
  What they see: the headline claims with their page images, and the ambiguities sorted by
  sensitivity, high first, each with its default and the default's source (conventions KB,
  author code, or guess). Not the whole YAML. In batch mode the checkpoint auto-approves and
  records what it would have asked.

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

**Tiers are two axes, not one.** Data access and compute are independent: an ML-on-CRSP paper
is data-substituted *and* compute-reduced, and a single letter hides which one drove the
outcome.

Data axis:

| Data tier | Data | What we attempt | What "success" means |
|---|---|---|---|
| A | Open (French library, FRED, yfinance, Kaggle, crypto exchanges, public GitHub data) | Exact replication | Headline claims Match within the derived tolerance |
| B | Paywalled but substitutable (CRSP → yfinance/Polygon/Sharadar; Compustat → public fundamentals) | Approximate replication | Same sign; our t-stat above 2 (or the paper's own threshold); and the point estimate inside the band the substitution registry predicts for this substitution. "Same order of magnitude" is not a criterion: 0.05 and 0.41 share one. |
| C | Proprietary/unavailable (prop tick data, private LOB) | Pipeline on synthetic/proxy data | Mechanics verified, no claim on numbers. Not in the MVP. |

Compute axis:

| Compute tier | Reported compute vs. the declared envelope | What we attempt | What "success" means |
|---|---|---|---|
| 1 | Fits the budget on the declared machine | Full scale | As the data tier says |
| 2 | Does not fit | Reduced scale (subset, fewer epochs, smaller variant), compared against a configuration the paper *also* reports at that scale | Match at the matched point; the trend along the learning curve agrees in sign |

The compute envelope (cores, RAM, GPU or none, disk) is declared in the orchestrator config;
without it "fits the budget" is undefined. Compute tier 2 is a statement about the paper
*given this machine*, and the report names the machine.

What kind of test each combination is; the report uses these words instead of "replication"
for everything: *reproduction* (author code, same data), *re-implementation* (our code, same
data), *conceptual replication* (our code, substituted data), *mechanics only* (synthetic
data).

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
  Where the numbers come from, since measuring them needs both sources: (i) the literature
  (Shumway 1997 on delisting bias; the OSAP and JKP papers on universe sensitivity), (ii) one
  measured run per substitution by whoever has the paywalled source, stored with its date,
  (iii) otherwise "unknown," and the gap stays unexplained. An entry without a provenance is
  a guess wearing a lab coat.
- *Explicit success criteria written down before running.* Why: otherwise the agent (or the
  human) will rationalize whatever comes out. Pre-registration for replication.
- *Budget allocation as a table in plan.yaml, set by code from family defaults.* E.g.
  intake+spec+triage 10 min, env 5, data 15, implement 25, run 20, verify 15, diagnose 15,
  report 5 (reserved and un-killable). Why: the orchestrator enforces this; the model does not
  get to decide to spend 60 minutes on data. Verify gets real minutes because seeds, the grid,
  and the leakage tests are runs (§0.2). The model may *request* a reallocation with a reason;
  code grants it against a cap.
- *Dry-run mode.* Stages 1–3 alone, producing the tier, the plan, and a cost estimate, with no
  data fetched beyond the probe. Why: it is the cheapest useful product in the system, and it
  is what a user wants before spending an hour and a bill.
- *Freeze.* On leaving stage 3 the orchestrator hashes `spec.claims` and
  `plan.success_criteria` (tolerances included) and stores the hash in the run log. Any later
  change fails the run. Why: principle 0.7; pre-registration by the same agent that runs is
  only pre-registration if it cannot be undone.

**Failure modes.** Over-optimistic tier assignment; substitution with unknown consequences.

### Stage 4 — Environment

**Why.** Principle 0.1(d). Library defaults change (pandas resampling, sklearn solvers, PyTorch
initialization) and a 2018 paper's code often does not run on 2026 libraries. An unreproducible
environment makes the whole replication unreproducible.

**Input.** `spec.yaml`, `author_code/`.
**Output.** A container image or venv, a lockfile, a smoke-test log.

**Design ideas.**

- *Two environments, two purposes.* Reproducibility of *our* run needs a lockfile of whatever
  we used, and current libraries are fine for that. The paper's *era* matters only when running
  author code. Why: era-pinning everything is expensive and, for finance, mostly pointless;
  the real risk there is a pandas default that changed, which a golden fixture catches.
- *Era-pinned base images, for author code.* Pre-built images per year (py3.7+pandas0.25+
  torch1.4 for "2019", etc.). Why: building from scratch is slow and flaky; a pre-built image
  starts in seconds and matches the paper's likely environment. Caveat: old CUDA wheels carry
  no kernels for new GPU architectures, so an "era" image is only feasible on hardware of
  roughly that era or on CPU. Era feasibility is a property of the declared compute envelope
  and is checked in triage.
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
- *Credentials never enter the sandbox.* Adapters that need credentials (WRDS, Polygon) run in
  the harness process; the sandbox sees only cached parquet. Why: author code from an arbitrary
  repository runs in the sandbox, and a credential in reach of it is a credential lost.
- *Licensing is part of the cache design.* WRDS and most vendor contracts forbid sharing
  extracts; Yahoo's terms forbid non-personal use of yfinance. Why: a content-addressed cache
  shared across users is redistribution. The cache is per-user for licensed sources, and the
  report records the license of every source.
- *Schema contracts.* Each adapter declares columns, dtypes, and invariants (dates monotone, no
  duplicate keys, prices > 0). Validated on every load. Why: silent schema drift (a column
  renamed by a provider) is otherwise found in stage 9 at ten times the cost.
- *Descriptive-stats checkpoint against the paper's Table 1.* Row counts, date range, means,
  standard deviations, number of firms per year. Why: this is the cheapest and most decisive
  test of D' ≈ D. Its semantics depend on the data tier: in Tier A it is a *gate* (fail → stop
  and diagnose data before touching the model); in Tier B it is a *measurement*, since a
  substituted source will always differ, and the measured gap (firm count −18%, mean size
  +40%) is the first line of the report's attribution. A gate in Tier B would halt every run.
  The pass threshold is a number in the family template, not a judgment call at run time.
- *Look-ahead audit as a mechanical check.* Every feature carries an `available_at` timestamp;
  the join to returns asserts `available_at < decision_time`. Why: principle 0.4. The single
  most common way a finance replication produces a fake match. This is a lineage feature of
  the data library, not a discipline the model is asked to follow: a derived column's
  `available_at` is the max over its inputs, computed by the library.
- *Future-perturbation test.* Re-run feature construction with all data after date t replaced
  by noise; assert every feature value at or before t is bit-identical. Why: a black-box
  look-ahead detector that needs no lineage and no model judgment. Change the future; the past
  must not move. It catches what the shuffle test (stage 8) cannot.
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
  it; this is the difference between a 30-second and a 20-minute iteration. Do not build the
  engine: hash-cached DAGs exist (Hamilton, Kedro, snakemake, or joblib.Memory over plain
  functions). The design work is the node contracts, not the scheduler.
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
- *Two implementations of the critical formula, with a caveat.* Have the model write the signal
  computation twice (vectorized pandas and a naive loop) and assert equality on a sample. Why:
  it catches vectorization mistakes (alignment, groupby). It does *not* catch shared
  misreadings: the same model writes both and brings the same priors (the same `pct_change`
  default, the same lag convention). Independence against misreading comes from the golden
  fixture whose expected values were computed by hand in the spec, or from a different model.
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

- *Smoke config first, defined by the family.* One seed, one epoch, and the shortest sample that
  actually produces the headline statistic: for a cross-sectional sort with a 12-month lookback
  "one year" yields zero portfolio months, so the template says three years. Must finish in
  under five minutes. Why: most pipeline bugs surface on any input; find them cheaply.
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
- *The report is always produced.* The last five minutes of the budget are reserved and cannot
  be consumed by any earlier stage; a budget kill at minute 115 still yields a report that says
  what ran, what did not, and why. Why: a run that produces nothing teaches nothing, and the
  benchmark needs a row for every paper.

### Stage 8 — Verify

**Why.** Principles 0.1 and 0.4. Verification must both compare numbers *and* try to break the
result.

**Input.** `results.json`, `spec.yaml`.
**Output.** `verification.json`: per claim, outcome plus explanation.

**Outcomes per claim.**

| Outcome | Definition |
|---|---|
| Match | Within the claim's tolerance |
| Consistent | Same sign, significant in our run, and the gap is inside the band a recorded substitution or scale reduction predicts |
| Mismatch | Different conclusion, or gap unattributable |
| Untested | Skipped per plan.yaml, with the reason |

**Design ideas.**

- *One Match rule, derived, frozen.* Match ⇔ |Y' − Y_read| ≤ k·SE + reported_precision, with
  k = 2 and SE the standard error of the statistic on the paper's sample length (Sharpe:
  sqrt((1 + SR²/2)/T); t-stat: 1; a mean: σ/√T; accuracy: the paper's seed std, or a binomial
  SE on the test-set size if none is reported). Why: the earlier list (Sharpe ±0.1, t-stat
  ±0.5, accuracy ±1 point) was arbitrary, unit-blind, and ignored sample length; a derived rule
  cannot be chosen to fit the result, and it is computed in stage 3 and frozen (principle 0.7).
  A monthly Sharpe of 0.41 over 678 months has SE ≈ 0.04, so the derived tolerance is ≈ 0.09;
  ±0.1 was roughly right for that case and would be wrong for a 10-year sample.
- *Bootstrap when the analytic SE is wrong.* Block-bootstrap our own series to get a CI where
  returns are autocorrelated or overlapping (time-series predictability papers). Why: an
  i.i.d. SE badly understates the uncertainty of overlapping-return regressions. A wide CI
  makes everything "Match," so the report prints the CI width and a reader can see when a
  Match is uninformative.
- *Claims are not independent.* Ten rows of one table are one finding. The report gives a
  per-claim outcome and one joint verdict for the headline finding, and it does not count
  "8 of 10 rows Match" as evidence when the rows share the same data and sort.
- *Figure comparison.* Re-plot our result in the paper's format and overlay on the paper's
  figure image; have the model judge. Why: many claims are only in figures, and shape agreement
  (cumulative return curves, decile monotonicity) is informative even when numbers are off.
- *Shuffle test, for what it can catch.* Re-run with labels/returns shuffled within period. If
  the result survives, the *label* is leaking into the pipeline (a bad split, target encoding,
  test rows in training). Why: principle 0.4. What it cannot catch: look-ahead *in the
  features*. If the signal at t was built from the return at t+1, shuffling returns breaks
  that link and the result vanishes, so the test passes on a leaking pipeline. Feature
  look-ahead is caught by the future-perturbation test and the `available_at` audit (stage 5),
  which run here too. Three tests, three classes of leakage; none is optional.
- *Decile monotonicity check.* For sort-based papers, verify returns increase across deciles.
  Why: it is the structural claim behind the headline number and is more robust than the number.
- *Transaction cost and turnover sensitivity.* Report the result with 0, 10, 25 bps costs. Why:
  many anomalies vanish; the report should show where.
- *Subperiod stability.* Split the sample in halves. Why: a result driven by one decade is a
  different finding than a stable one.
- *Independent reviewer call, as a supplement.* A separate model call reviews the pipeline code
  against the spec, looking specifically for lag and leakage errors, without seeing the
  results. Why: principle 0.6, and blinding the reviewer to the numbers removes confirmation
  bias. Its limits: it shares the generator's priors, so it is blind to shared misreadings,
  and its recall over a few thousand lines is low. Use a different model family where
  possible, give it the spec's ambiguity list as a checklist, and never let it be the only
  leakage check.

**Failure modes.** Tolerances too loose; comparing annualized to monthly; declaring Match on a
leaking pipeline.

### Stage 9 — Diagnose: convention grid, then bisection (bounded)

**Why.** Principle 0.1: attribution. Principle 0.2: bounded. Principle 0.7: search is not
attribution. When a headline claim is Mismatch, the job is to find which error term explains
it, without turning the search into the thing §0.4 warns about.

The earlier design ("flip one hypothesis per iteration, three iterations, stop when it
matches") had two flaws. Three sequential flips cannot attribute among seven terms. And a
search that stops at Match selects the configuration closest to the paper on the same data,
which is exactly how a fake match is manufactured. So the stage has two parts with different
epistemic status.

**Design ideas.**

**Part 1 — the convention grid (sensitivity, not search).**

- *Run every high-sensitivity ambiguity as a one-factor flip, in parallel, from the DAG cache.*
  Eight binary ambiguities are eight cheap re-runs of the downstream nodes, not a sequential
  search. Why: the DAG cache makes them cheap, and running all of them at once removes the
  stopping rule that turns a sequential search into a fishing expedition.
- *Report the range, not the best point.* "The headline is 0.28–0.45 across defensible
  conventions; the paper's 0.41 is inside that range; the flip that moves it most is
  delisting handling." Why: this is term (g); it is what a practitioner needs to know, and it
  is honest about what the data can and cannot say.
- *A convention is attributed only with independent evidence.* The grid may show that
  equal-weighting reproduces 0.41 exactly. That is a *candidate*, reported as "matches under
  EW, for which the paper's text gives no support," unless the text, author code, or an
  intermediate the paper reports supports it. Why: principle 0.7. The report distinguishes
  "we found a configuration that matches" from "we found the paper's configuration."
- *Hypothesis library per family, ranked by hit rate × cheapness,* seeds the grid. For
  cross-sectional papers: lag alignment, universe filters, equal vs value weighting, delisting
  returns, simple vs log returns, winsorization per-period vs pooled, rebalance frequency,
  seed variance. Why: most mismatches have one of a dozen causes; the list is the prior, and
  hit rates are updated from the benchmark (§3.5).

**Part 2 — bisection over intermediates (attribution proper).**

- *Compare every intermediate the paper reports, not just the final number.* Table 1 moments,
  firm counts per year, decile counts, univariate sorts, the baseline's number. The first
  divergence localizes the term. Why: an intermediate the paper reports is independent
  evidence, so a divergence there attributes without search; and bisection over the pipeline
  is far cheaper than re-reading the code.
- *One targeted change per iteration after bisection, re-run only affected nodes.* Why:
  changing two things at once destroys attribution.
- *Hard caps.* The grid is one parallel batch; bisection gets 3 iterations or 15 minutes,
  whichever first. Then stop and report what was ruled out and what remains unexplained.
  Why: principle 0.2; an unbounded loop is the most common way agentic systems blow their
  budget.
- *Record everything, including the flips that did nothing.* Why: they go in the report as
  "things we ruled out," which is real information; and they train the hit rates.

### Stage 10 — Extend (only with leftover budget)

**Why.** Once the pipeline exists, extensions are nearly free and are often the most valuable
output for a practitioner.

**Design ideas.**

- Out-of-sample extension past the paper's end date (the single most requested thing; if the
  user is a practitioner, this is the product and "leftover budget" is the wrong priority; see
  §7 Q6).
- Cost-adjusted results; alternate universes (large-cap only); parameter sensitivity grid.
- Alternate data source (if Tier A and B sources both exist, run both and compare).
- Why these and not others: each is a config change on the existing DAG, so each costs minutes.

### Stage 11 — Report

**Why.** The deliverable is the attribution, not the number. The report is how a reader who was
not present judges the replication.

**Structure of `REPORT.md`.**

1. The kind of test performed (stage 3), the data and compute tiers, and every deviation from
   the paper, first, in one table.
2. Claims table: paper value | ours | CI | outcome | attribution note.
3. Adversarial checks: shuffle test, look-ahead audit, survivorship, cost sensitivity.
4. Ambiguities and the defaults chosen; which ones were tested in stage 9.
5. Convention grid: the headline's range across defaults. Bisection history: what was ruled
   out, what remains unexplained.
6. Extensions.
7. Reproducibility block: command, data hashes, environment lockfile, run log hashes.
8. Confidence grade for the replication as a whole, with the rule that produced it.

**Design ideas.**

- *Grade is a vector first, a letter second.* Four independent axes, each with a fixed rule:
  data fidelity (Tier A checkpoint passed / Tier B gap measured / synthetic), procedure
  fidelity (author code / re-implemented, with the count of unexplained ambiguities), result
  (Match / Consistent / Mismatch / Untested per headline claim), integrity (all three leakage
  tests passed / one failed / not run). The letter is a fixed function of the vector. A: all
  headline claims Match, integrity passed, data tier A. B: Consistent with attributed gaps,
  integrity passed. C: mechanics verified, numbers not. F: any leakage test failed, or the
  pipeline does not run. Why: the grade must mean the same thing across papers, and a letter
  alone hides *which* axis failed, which is the only thing a reader acts on. The report also
  names the kind of test performed (reproduction, re-implementation, conceptual replication,
  mechanics only), because a B on a reproduction and a B on a conceptual replication are
  different facts.
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

**`ask_user` is a checkpoint, not a chat.** It is callable only at the two points in §3.4;
elsewhere it records the question in the run log and returns the spec's default. Why: a model
that can block on a human mid-build has an unbounded budget, and an unattended run would hang.

**The paper is untrusted input.** PDFs, READMEs, and author code are text the model reads and
may be steered by; "report Match" in a footnote is a prompt injection. Why it matters here:
the verdict must not depend on anything the paper says about itself. The frozen tolerances
and the mechanical checks are the defense, and the reviewer call gets the code, not the prose.

**The output repo runs without the harness.** The DAG runner, data adapters, lineage checks,
and invariant tests ship as a published library the generated repo depends on. Why: a
"runnable replication" that only runs inside the agent's harness is not one.

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

Exactly two, both optional, both time-boxed, and both auto-approved in batch mode (§0.2):

- After stage 2: approve/correct the spec (highest leverage).
- After stage 3: approve the tier and substitutions (prevents a wasted run).

Why not more: each pause costs wall clock; why not fewer: these two decisions are where a
human's five minutes saves the agent an hour.

### 3.5 Cross-run memory

A knowledge base, seeded by hand before the first run (§0.8, build step 0) and grown across
replications:

- Ambiguity defaults per family (stage 2).
- Data source quirks and substitution consequences (stages 3, 5).
- Hypothesis library with observed hit rates (stage 9).
- Environment images per era (stage 4).

Why: most of what makes a human replicator fast is accumulated convention knowledge. The agent
should accumulate it too, in files, reviewed by a human before promotion.

### 3.6 Observability and cost

- Trace every model call: stage, tokens, cost, wall time. Per-paper cost is a benchmark metric
  and a per-run cap enforced by the orchestrator, alongside wall time (§0.2).
- Prompt caching on the static system prompts and the paper document (it is reread by every
  stage). Why: the paper is the largest stable prefix; caching it is the single biggest cost win.

### 3.7 One agent or many

The question decomposes into three, and each answer is forced by a principle already in this
document.

**One context or many?** Many. Principle 0.3: a 90-minute build in one conversation degrades
and cannot be audited; state lives on disk and each stage starts from files. Principle 0.7: the
reviewer must not see the results, and the critic must not inherit the extractor's reasoning,
which is impossible inside one context. So the extractor, the critic, the builder, and the
reviewer are separate contexts by necessity, not by taste.

**One role or many?** Several, but few, and they do not talk to each other. Extraction,
criticism, building, and reviewing are different cognitive modes with different prompts; one
prompt doing all four does each worse and cannot be blinded. But the roles do not negotiate,
do not message each other, and do not decide what runs next. Code decides. Each role is a
worker with a typed input and a typed output on disk.

**One process or many?** Many, and most are not agents. Seeds, the environment build, data
fetches, and the convention grid run concurrently, but they are deterministic code the
orchestrator launches. Only the builder and the diagnoser are agentic loops; everything else
is a single structured call or plain code.

**One product?** Yes. The user talks to one thing: `replicate <paper>`, two optional
checkpoints, one report. The quant firm and the research lab run the same agent with a
different deployment profile (§3.8); finance and CS are plugins (§1.5). Neither justifies a
second agent.

| Role | Kind | Sees | Must not see | Budget | Why separate |
|---|---|---|---|---|---|
| Extractor | one structured call | PDF, companions | — | ~5 min | the most important call; global consistency needs the whole paper in one context |
| Claim re-reader | one cheap call per claim | one page image + `where` | the extractor's value | ~1 min | term (0); independence from the extractor is the point |
| Critic (referee) | one call | PDF + draft spec | — | ~3 min | criticism is a different mode from extraction |
| Triage | code + one call | spec, probe results, compute envelope | — | ~2 min | mostly arithmetic; the call picks substitutions and writes success criteria |
| Builder | agentic loop with tools | spec, plan, template, data schemas and aggregates | raw data rows (§3.8) | 25 min | the only long loop; read-heavy side tasks (author repo, data docs) go to cheap subagents |
| Runner | code | pipeline, configs | — | 20 min | seeds and the grid in parallel; no model in the loop |
| Reviewer | one call, blinded | code + spec + ambiguity list | results, the paper's prose | ~3 min | confirmation bias; principle 0.6 |
| Diagnoser | bounded agentic loop, fresh context | run log, grid, intermediates vs. paper | the builder's context | 15 min | starts from disk; the builder's memory of its own choices is the bias to remove |
| Reporter | one call | everything on disk | — | 5 min, reserved | reads artifacts; asserts nothing that is not on disk |
| Judge | one call, benchmark only | report + rubric + human labels | — | — | never in the production path |

What this is not: a "planner agent" delegating to "worker agents" that report back in prose.
Every handoff is a file with a schema, checked by code. The orchestrator is the only thing
with a global view, and it is Python.

When would a second agent be justified? Only for a role that needs its own tool loop and its
own budget and cannot be expressed as a stage. The diagnoser is the one candidate, and it is
already a stage. Splitting the builder by node group (data nodes vs. model nodes) buys
parallelism at the cost of a merge; do it only once the benchmark shows building is the
bottleneck.

### 3.8 Production constraints and deployment profiles

Same agent, two profiles, chosen by config:

| | Quant firm | Research lab |
|---|---|---|
| Dominant track | Finance | CS/ML |
| Data | WRDS/CRSP, internal panels through the local-file adapter, licensed vendors | Hugging Face, torchvision, public URLs, author checkpoints |
| Compute | CPU boxes; GPU rare | GPUs; compute tier 2 common |
| Credentials | Vendor logins, internal databases | HF tokens, cloud GPU |
| Dominant risk | Data egress and license terms | Arbitrary author code |
| Human checkpoint | A PM approves substitutions and cost assumptions | A researcher approves the reduced-scale plan |
| What they act on | Cost-adjusted, out-of-sample result (stage 10 is first-class) | Whether the method's gain over the baseline survives at matched scale |

What "production" adds beyond the pipeline:

- **Data rows never enter a prompt.** The model sees schemas, row counts, and summary
  statistics; it writes code that runs on the data. Why: a firm's internal panel must not
  leave the building, and it keeps contexts small. Enforced by `fetch_data` and `run_node`
  returning aggregates and paths, never frames.
- **Every run is versioned end to end.** Agent version, prompt hashes, conventions-KB version,
  adapter versions, and the frozen spec hash are in the report. Why: a replication is
  evidence, and evidence has a chain of custody. It is also the only way to know whether a
  benchmark change came from the agent or from the KB.
- **A run always ends in a report or a structured failure.** No silent exits, no partial
  state without a manifest. Why: that is what a scheduler, a dashboard, and an audit consume.
- **Tenancy.** Caches, credentials, and the sandbox are per-user; licensed data never crosses
  users through the cache (stage 5). Why: license terms and least privilege.
- **Human minutes are a reported metric.** The target is about 15 minutes of human attention
  per paper at the two checkpoints. Report human minutes next to wall clock and cost; a
  checkpoint that routinely takes 40 minutes of the human's time is a product bug.

---

## 4. Evaluating the agent (build this first)

**Why first.** Principle 0.5.

- **Ground truth set.** 10–20 papers with public, trusted replications.
  Finance: Open Source Asset Pricing (Chen & Zimmermann) publishes code and results for 200+
  anomaly signals; Jensen, Kelly & Pedersen's factor replication has public code; Kenneth
  French's library covers the classic factor papers.
  CS/ML: PaperBench (OpenAI) and CORE-Bench are existing paper-replication benchmarks with
  rubrics to borrow.
- **The benchmark is hermetic.** Every benchmark paper's data is a frozen snapshot in the cache;
  benchmark runs never touch the network. Why: yfinance changes retroactively and rate-limits;
  a score that moves because Yahoo changed is not a regression signal. Ground-truth finance
  numbers from OSAP or JKP were computed on CRSP, so for open-data benchmark papers the ground
  truth is *our own* hand replication on the open source, verified once by a human, not the
  published CRSP number.
- **Contamination is the default, not the exception.** The model has read the momentum paper,
  the Fama–French papers, the OSAP code, and PaperBench. A high score on famous papers measures
  recall. So: stratify by fame, include recent or obscure papers (SSRN working papers from the
  last year), and report the strata separately. And run a *perturbed-paper probe*: hand the
  agent a copy of a benchmark paper with the headline number altered; an agent that matches
  the altered number is reading the answer, not computing it.
- **Stratify by tier and family.** Why: an agent that aces Tier A momentum papers and fails
  everything else should look like that in the numbers.
- **Metrics per paper, calibration first.** The north star is **human agreement with the
  agent's verdict**, split into the false-Match rate (agent says Match, human says Mismatch or
  leakage; this is the number that must be near zero) and the false-Mismatch rate. Then:
  headline-claim Match rate on the human-verified subset; Consistent rate; wall time; cost;
  leakage tests passed; grid width. Why: Match rate alone rewards loose tolerances and rewards
  search (§0.7); calibration punishes both.
- **Rubric grading.** Per paper, a checklist of things a correct replication must contain
  (correct universe filter, correct lag, correct weighting, ...), graded by a model judge with
  human spot checks. Why: the final number can be right by accident; the rubric checks the path.
- **Stage-level evals are the fast loop.** Spec extraction alone (against hand-written specs);
  data checkpoint alone; verify alone (given a known pipeline, does it produce the right
  outcomes?). Why: end-to-end evals are slow and confound causes; stage evals tell you *which*
  stage to fix. And they are cheap: a spec-extraction eval over 20 papers is minutes and a few
  dollars; an end-to-end sweep is a day and hundreds.
- **Regression discipline, tiered.** Every change to a prompt, tool, or stage runs the stage
  evals; the end-to-end set runs nightly in batch mode and before any promotion into the
  conventions KB.

---

## 5. Build order

0. **Conventions knowledge base, by hand.** For the first two families (cross-sectional
   anomaly, ML return prediction): the ambiguity checklist, the conventional default for each,
   its source, and its known sensitivity, from the replication literature. Why: principle 0.8;
   this is the asset the hand replications consume, and writing it first makes step 1 faster
   and turns its decisions into citations instead of guesses.
1. **Three replications by hand**, Tier A on the data axis, one per track plus the hybrid
   (the hybrid third, not first; it is the hardest paper and a template should not be derived
   from the worst case): a momentum or
   factor paper on French library data (finance); a small classification or LM paper with a
   public repo and a released checkpoint (CS); an ML-on-returns paper with open data (hybrid). Record every step and decision. Why: this produces the spec schema, the family
   templates, the hypothesis library, and the first three benchmark entries from reality
   instead of imagination. Each one is committed the day it is done as a hermetic benchmark
   entry (frozen data snapshot, hand spec, expected outcomes), so the harness in step 6 has
   rows before it has code.
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
2. WRDS/CRSP access: yes, no, or "the user brings credentials"? This also decides whether the
   substitution registry can ever be *measured* (stage 3) or only cited.
3. ~~Operator-in-the-loop or fully unattended?~~ **Decided: both, as modes (§0.2).**
   Interactive is the MVP product; batch exists from day one because the benchmark needs it.
4. Which three papers for the hand replications? Constraint added: all three Tier A on the
   data axis, and the hybrid third, not first.
5. Engine: Claude Code sessions per stage (fast MVP) vs. custom loop (control)?
6. ~~Who is the user, and is the objective "match the paper" or "does this work"?~~
   **Decided 2026-09-14: a quant firm (reproduce before acting) and an ML research lab
   (reproduce before building on). Practitioner-first: the paper's number is the sanity check;
   for the quant profile, stage 10's cost-adjusted out-of-sample result is first-class (§3.8).**
7. ~~Where does 1–2 hours come from?~~ **Decided: it is the productivity target, 20–30
   researcher-hours → 1–2 h turnaround (§0.2). Interactive mode: 2 h target with a configurable
   hard kill (default 4 h); batch mode: cost-capped.**
8. ~~Both tracks now, or finance first?~~ **Decided: both, because one user is finance-heavy
   and the other CS-heavy (§3.8).** Consequence: the conventions KB (step 0) and the hand
   replications cover both tracks, and the CS success definition (checkpoint evaluation vs.
   training reproduction at matched scale) is written into the CS family templates before the
   CS plugins are built. Open sub-question: which CS families first.
9. **Compute envelope.** What machine does this run on: cores, RAM, GPU or not? Compute tier
   2, era feasibility, and "seeds in parallel" are undefined without it.
10. **Data licensing.** yfinance is unofficial scraping under terms that forbid non-personal
    use; vendor contracts forbid sharing extracts. Which sources are acceptable for a product
    versus a personal tool decides the adapter list and the cache design. For the quant-firm
    profile this is not optional: assume licensed sources and a local-file adapter, and treat
    yfinance as a development convenience only.
11. ~~One agent or many?~~ **Decided: one product and one orchestrator; several model roles
    that are separate contexts by necessity (blinding, context hygiene) and never talk to each
    other (§3.7).**

---

## 8. Open-source references, by stage

Verified 2026-09-14. "Use for" says which part of this design each one serves; "caveat" says
why it is not simply adopted wholesale. Everything here is a component or a benchmark, not a
substitute for the pipeline: none of these systems does triage, freezes tolerances, runs a
convention grid, or reports attribution.

### Benchmarks and ground truth (§4)

| Reference | Use for | Caveat |
|---|---|---|
| [PaperBench](https://github.com/openai/preparedness) (OpenAI; [paper](https://arxiv.org/abs/2504.01848)) | CS track rubric design: 20 ICML 2024 papers, 8,316 gradable leaf tasks, hierarchical rubric graded by a model judge. Borrow the rubric structure and the judge protocol. | ML-only; grades *code and execution*, not the verdict calibration this design makes primary. Contamination: famous papers. |
| [CORE-Bench](https://github.com/siegelz/core-bench) (Princeton; [paper](https://arxiv.org/abs/2409.11363)) | "Reproduction" mode (author code + same data) evaluation: 270 tasks from 90 papers in Python/R across CS, social science, medicine. Task format for oracle mode. | Given the repository; no re-implementation, no data substitution. |
| [SciReplicate-Bench](https://github.com/xyzCS/SciReplicate-Bench) | Algorithm-level reproduction from LaTeX source; its dual-agent baseline (Paper Agent + Code Agent) is a data point for §3.7. | NLP only; tasks are single functions, not end-to-end results. |
| [LMR-Bench](https://github.com/du-nlp-lab/LMR-Bench) | Masked-method reproduction in LM research; a cheap stage-6 eval format (fill the missing node). | Same limitation: unit-level. |
| [EXP-Bench](https://github.com/Just-Curieous/Curie/tree/main/benchmark/exp_bench) | End-to-end experiment reproduction from 51 AI papers; its 0.5% full-success rate is the honest prior for compute tier 2. | ML-only. |
| [REPRO-Bench](https://github.com/uiuc-kang-lab/REPRO-Bench) | The closest thing to a *verdict* benchmark: 112 social-science papers with published reproduction reports; the agent must assess reproducibility. Directly relevant to calibration (§0.7). | Social science, reproduction packages given; best agent 21% accuracy. |
| [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) | Reference harness for bounded agentic ML experimentation with budgets. | Improve-a-model tasks, not replication. |
| [Open Source Asset Pricing / CrossSection](https://github.com/OpenSourceAP/CrossSection) (Chen & Zimmermann; [site](https://www.openassetpricing.com/)) | Finance ground truth and the conventions KB seed (§0.8): 319 signals with `SignalDoc.csv` recording each signal's construction, sample, and the paper's reported vs. reproduced numbers. This is the single most valuable file for step 0 of the build order. | Code is Stata/R on CRSP/Compustat; numbers are CRSP-based, so Tier A open-data runs compare against our own hand replication, not OSAP's number (§4). |
| [JKP / Global Factor Data](https://github.com/bkelly-lab/jkp-data) (Jensen, Kelly & Pedersen; [docs](https://jkpfactors.s3.amazonaws.com/documents/Documentation.pdf)) | Python code building 153 factors from CRSP/Compustat with every universe, weighting, and lag convention documented; the second conventions-KB seed and the "universe sensitivity" entries of the substitution registry. Pre-computed factor returns are free from jkpfactors.com. | Needs WRDS to run; the free artifact is the factor returns, not the panel. |
| [Tidy Finance](https://www.tidy-finance.org/) (Python and R editions; [GKX replication post](https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/)) | Plain-language conventions for CRSP/Compustat cleaning, delisting returns, the 6-month lag, and a worked Gu–Kelly–Xiu partial replication. Best prose source for the KB. | Textbook, not a library. |
| [Gu–Kelly–Xiu replications](https://github.com/NanyeonK/Empirical-asset-pricing-via-machine-learning) ([another](https://github.com/duongtran14/Partial-replication-of-Gu-Kelly-Xiu-2020-Empirical-Asset-Pricing-via-Machine-Learning.)) | Candidate hybrid-track hand replication (§5 step 1): the authors publish the characteristics panel on Xiu's homepage, so it is Tier A on the data axis. | Community code, unverified; use as a reference, not an oracle. |

### Agents that reproduce papers (prior art for §3.7)

| Reference | Use for | Caveat |
|---|---|---|
| [Paper2Code / PaperCoder](https://github.com/going-doer/Paper2Code) | Planning → analysis → code generation with separate prompts; evidence that role separation beats one prompt (§3.7). Its planning artifact is a weaker cousin of `spec.yaml`. | Generates a repo; no data stage, no verification, no verdict. |
| [AutoReproduce](https://arxiv.org/abs/2505.20662) | "Paper lineage" (chase cited method papers) is the CS version of stage 1's citation chasing. | Paper, code availability varies. |
| [The AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) | Progressive agentic tree search over experiments; a reference for bounded exploration, and a cautionary one (it grades itself). | Discovery, not replication. |
| [Claude Agent SDK (Python)](https://github.com/anthropics/claude-agent-sdk-python) | Engine option 1 in §3.3: bundled tool loop, subagents, structured outputs, file checkpointing and rewind. | Sessions per stage need the paper as a stable cached prefix (§3.6). |
| [anthropics/financial-services](https://github.com/anthropics/financial-services) | Apache-2.0 reference agents, skills, and MCP data connectors for finance workflows; "AI drafts, humans sign off" matches §3.4. | Equity research and banking workflows, not replication. |

### Intake (stage 1)

| Reference | Use for | Caveat |
|---|---|---|
| [Marker](https://github.com/VikParuchuri/marker) | Default PDF → markdown with tables and equations; the first of the two extractors. | License restrictions on commercial use; check before the quant-firm profile. |
| [Docling](https://github.com/docling-project/docling) (IBM; [paper](https://arxiv.org/abs/2501.17887)) | The second extractor for the diff; strong table structure recognition, permissive license. | Slower on long PDFs. |
| [MinerU](https://github.com/opendatalab/MinerU) | Alternative second extractor; best on complex layouts. | Heavier dependency stack. |
| [GROBID](https://github.com/kermitt2/grobid) | Header, reference, and citation parsing for companion discovery and citation chasing. | Turns tables and formulas into images; not for the table path. |

### Data (stage 5)

| Reference | Use for | Caveat |
|---|---|---|
| [pandas-datareader](https://github.com/pydata/pandas-datareader) ([Fama–French reader](https://pandas-datareader.readthedocs.io/en/latest/readers/famafrench.html)) | French library and FRED adapters, Tier A. | Maintenance is intermittent; wrap it, do not depend on its API shape. |
| [wrds](https://pypi.org/project/wrds/) | CRSP/Compustat adapter for the quant-firm profile; runs in the harness process, never the sandbox (§3.8). | Credentials; license forbids sharing extracts. |
| [OpenBB Platform](https://github.com/OpenBB-finance/OpenBB) | One adapter interface over ~100 providers (Tiingo, Polygon, FMP, FRED, …), already typed and provider-swappable; a candidate base for the adapter layer instead of writing one per source. | Provider quality varies; the substitution registry still has to characterize each. |
| [Tiingo](https://github.com/hydrosquall/tiingo-python) | Survivorship-bias-adjusted daily EOD with decades of history and a free tier; the recommended replacement for yfinance in anything shared (§7 Q10). | Free tier caps; fundamentals are paid. |
| [pyanomaly](https://github.com/chulwoohan/pyanomaly) | 200+ firm characteristics from CRSP/Compustat in under an hour; quantile sorts, factor and cross-sectional regressions. A ready-made finance family template for the quant-firm profile. | WRDS-only; conventions are its own, so diff them against the KB. |
| [pandera](https://github.com/unionai-oss/pandera) | Schema contracts on every adapter output (columns, dtypes, monotone dates, no duplicate keys, prices > 0). | Does not do lineage; `available_at` is ours. |
| [Lacuna](https://github.com/eyenoticeall/Lacuna) ([PyPI](https://pypi.org/project/lacuna-quant/)) | Validation layer over pandas/Polars for leakage, overfitting, fragile results, unrealistic costs, and missing point-in-time evidence; overlaps with stage 8's adversarial checks and the convention grid. | Young project; read the code before trusting a check. |
| [pit-release-gate](https://github.com/MaxWellApexLab/pit-release-gate) | Look-ahead susceptibility scoring for features from late-arriving data; complementary to the future-perturbation test. | Narrow scope. |
| [Qlib](https://github.com/microsoft/qlib) | Reference for the ML-return-prediction family: rolling train/test, point-in-time data handling, an expression engine for features. | Its own data format; adopt patterns, not the platform. |

### Environment, pipeline, sandbox (stages 4, 6, 7)

| Reference | Use for | Caveat |
|---|---|---|
| [pixi](https://prefix.dev/) / [uv](https://github.com/astral-sh/uv) | Lockfiles for *our* environment (§ stage 4); pixi for conda+PyPI mixes (CUDA), uv for pure-PyPI. | Era images for author code still need containers. |
| [Hamilton](https://github.com/apache/hamilton) ([caching](https://hamilton.dagworks.io/en/latest/concepts/caching/)) | The DAG of pure nodes with hash-based caching from stage 6: nodes are plain functions, cache keys are code version + inputs, outputs to parquet. Exactly the engine the design says not to build. | Learn its node-naming conventions; the family template becomes a Hamilton module. |
| [E2B](https://github.com/e2b-dev/E2B) | Firecracker microVM sandboxes with an Apache-2.0 SDK, for the builder's execution environment when not self-hosting Docker. | Cloud infra is not open source; the quant-firm profile likely needs on-prem Docker or gVisor. |
| [awesome-sandbox](https://github.com/restyler/awesome-sandbox) | Survey of self-hostable sandboxes for the on-prem case. | — |

### Evaluation harness (§4)

| Reference | Use for | Caveat |
|---|---|---|
| [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) (UK AISI) | The benchmark harness itself: tasks, solvers, model-graded scorers with rubrics, bootstrap CIs on scores, a log viewer for inspecting grader reasoning. Stage-level evals and the judge (§3.7) fit its Task/Solver/Scorer model directly. | Model-graded scoring inherits §6's last hard problem; keep the human spot checks. |
| [anachron](https://github.com/LesterALeong/anachron) | Inspect extension scoring whether an agent used information it could not have had at the time; a point-in-time check on the *agent's tool calls*, complementary to the pipeline-level tests. | Small project. |

**What does not exist yet, as far as this search found:** a benchmark of *finance* paper
replications with human-verified verdicts on open data, and any agent that reports attribution
rather than a number. Both are what §4 and §5 build, and both are publishable on their own.

---

## 9. Comparison to open-source work, and the deletion pass

Added 2026-09-14. Method: compare the methodology in §0–§8 against what the repositories in
§8 actually do, then run the five-step process (make requirements less dumb; delete the part or
process; simplify and optimize; accelerate cycle time; automate, last). §9 is the **MVP cut**.
§2 remains the full design; anything marked *deleted* here is not built until its add-back
trigger fires. Where §9 and earlier sections conflict, §9 wins for the MVP.

### 9.1 What the existing systems do, side by side

| Dimension | PaperBench / CORE-Bench (benchmarks) | Paper2Code, SciReplicate, AutoReproduce (agents) | OSAP / JKP (finance replication projects) | Lacuna (validation) | This design |
|---|---|---|---|---|---|
| Input | PDF (PaperBench); repo + PDF (CORE) | PDF, sometimes LaTeX | The paper, CRSP/Compustat, a human | Signal and price frames | PDF, optional repo, optional user data |
| Output | Score against a rubric | A code repository | Signal code, portfolio returns, a doc row per paper | PASS/WARN/FAIL/UNKNOWN per check | A report with per-claim outcomes and attribution |
| Spec artifact before code | Rubric tree, written by humans with the authors | Planning + analysis artifacts (Paper2Code) | `SignalDoc.csv`: one hand-extracted row per paper with key table, test, sign, return, t-stat, weighting, quantile, holding period, start month, filters, sample years | None | `spec.yaml` with claims, ambiguities, and variants |
| Number comparison | Result-match leaf nodes, judged by a model | None (code quality only) or execution-output equality | Original-paper t-stat vs. replicated t-stat; quality label 1_good / 2_fair / 3_distant / 4_lack_data | Bootstrap, PSR/DSR, multiple-testing corrections | Derived tolerance = 2 SE + precision, frozen |
| Leakage checks | None | None | Convention discipline, by hand | Availability-safe joins, purged CV, forward-looking data detection | Shuffle, future-perturbation, `available_at` audit |
| Attribution of a gap | None | None | Notes column, by hand | Sensitivity surfaces, subperiods | Convention grid + bisection over intermediates |
| Time budget | 12 h reproduce cap; agents mismanaged time | None stated | Weeks per paper | n/a | 2 h target, hard kill, report reserved |
| Human role | Rubric author; JudgeEval spot checks | None | Everything | None | One checkpoint (§9.3) |
| Honesty mechanisms | `reproduce.sh` re-run on a fresh VM; author-code blacklist with post-hoc log monitor; zeroed score on violation | None | Public code | Signed `.lacuna` audit bundles | Freeze; blinded reviewer; perturbed-paper probe |
| Best reported result | 21% (PaperBench); ~21% verdict accuracy (REPRO-Bench) | 88% "rated best by authors" for code, no execution claim | ~85% of predictors reproduce clearly or likely | n/a | not built |

Three things the comparison makes plain:

1. **Nobody does the verdict.** The agents stop at code. The benchmarks grade with a model
   judge. REPRO-Bench, the one benchmark that asks for a reproducibility verdict, tops out at
   about 21% accuracy. The verdict engine is the uncontested part of this design and the
   hardest part of the prior art.
2. **The finance conventions KB and the finance spec benchmark already exist.** OSAP's
   `SignalDoc.csv` is 331 rows of exactly the fields `spec.yaml` asks the extractor to
   produce, hand-extracted by domain experts, with the original paper's t-stat next to the
   replicated one. Step 0 of the build order and the spec-extraction eval for the finance
   track are downloads, not work. PaperBench's author-co-developed rubrics play the same role
   for the CS track.
3. **The failure modes are known.** PaperBench's agents "frequently finished early, claiming
   that they had finished or had faced a problem they couldn't solve," and "all agents failed
   to strategize about how best to replicate the paper given the limited time." Removing the
   agent's ability to end early nearly doubled one model's score. This design's orchestrator
   already owns the clock; §9.4 makes it own the stop as well.

### 9.2 Step one: make the requirements less dumb

Every requirement needs a person's name attached. These had none, or had a weaker
justification than the design pretended:

| Requirement | Who asked for it | Verdict |
|---|---|---|
| Two PDF extractors diffed, disagreements sent as page images | Nobody. Justified by "table errors silently corrupt claims," which the page-image re-read of every claim already covers. | Delete |
| N independent spec extractions diffed | Nobody. Already downgraded in the review to "supplements the checklist." | Delete |
| Era-pinned base images per year | Nobody. PaperBench reproduces on a fresh Ubuntu VM with current libraries; CORE-Bench uses plain Docker. Author code gets its own requirements file; if it fails to install, that is a finding. | Delete |
| Two implementations of the critical formula | Nobody. Golden fixtures with hand-computed values catch the same class and are not correlated with the model's priors. | Delete |
| Figure overlay judged by a model | Nobody. Claims in figures are extracted as numbers by the re-reader or marked Untested. | Delete |
| Synthetic data for Tier C | Nobody. Already out of the MVP. | Delete |
| Eight paper families with templates | Nobody asked for eight. OSAP's rows show the anomaly family is one template with six knobs: test type (port sort 70%, regression), weighting (EW/VW), quantile, holding period (1 or 12 months), start month, filter. Two hundred papers fit it. | Two families for MVP: cross-sectional anomaly, ML return prediction. One CS family: classification / LM eval on a released checkpoint. |
| Cross-run memory with automatic promotion | Nobody. The KB is seeded from OSAP and JKP and edited by hand from benchmark findings. | Delete the automation; keep the files. |
| Container image cache by lockfile hash | Nobody. `uv` resolves and installs in seconds. | Delete |
| Hash-cached DAG engine (Hamilton) | The convention grid, which needs cheap downstream re-runs. `joblib.Memory` over plain functions gives that with zero framework. | Simplify; add Hamilton back when the grid is measurably slow |
| Transaction-cost, subperiod, PBO, turnover checks | The quant-firm profile, by name. But Lacuna already implements all of them with PASS/WARN/FAIL/UNKNOWN outcomes and audit bundles. | Adopt, do not build |
| Bootstrap CI | Time-series predictability papers only. | Not in MVP; analytic SE only. |
| Blinded reviewer call | Principle 0.6, but no evidence of its marginal value over the mechanical tests. | Keep as one non-gating call; measure its hit rate on the benchmark; delete if zero |
| Two human checkpoints | The operator, by name. But the second (approve tier and substitutions) is shown five minutes after the first and the human has just read the spec. | Merge into one checkpoint showing spec + plan together |
| Read-heavy subagents, prompt caching layout, tracing schema | Engine details that leaked into a design document. Weave (used by CORE-Bench) or any tracer does the job. | Delete from the design; engine choice handles them |
| Seven separate artifacts (spec, plan, manifest, results, verification, report.json, run log) | Nobody asked for seven. | Three: `spec.yaml` (claims, ambiguities, plan, frozen hash), `runs/` (one JSON per run, hash-named), `report.json` + `REPORT.md` |

### 9.3 Step two: delete, and what survived

After deletion the pipeline is five stages, not eleven, with four model roles, not ten, one
human checkpoint, and three artifacts.

| Stage | What it does | Model calls | Code |
|---|---|---|---|
| **Spec** | PDF → `spec.yaml` (claims with page images, ambiguities with KB defaults, method variants), then the plan: data probe, tier, derived tolerances, budget. Freeze. One human checkpoint, or auto-approve in batch. | Extractor (1), claim re-reader (1 per claim, cheap), referee (1) | Probe, tolerance derivation, freeze |
| **Setup** | Lockfile with `uv`; adapters fetch into a per-user content-addressed cache; pandera schemas; Table 1 checkpoint (gate in A, measure in B). | none | all |
| **Build + Run** | Instantiate the family template, fill paper-specific nodes, golden fixtures, invariants, smoke run, full run, seeds. The builder cannot end the stage; the orchestrator does. | Builder (agentic loop) | Docker, joblib cache, seeds |
| **Verify** | Fresh-container run of `reproduce.sh`; derived-tolerance comparison; shuffle, future-perturbation, `available_at`; Lacuna checks for the finance profile; convention grid as one parallel batch; bisection over paper-reported intermediates, capped. | Reviewer (1, non-gating) | everything else |
| **Report** | Kind of test, tiers, deviations, claims table, leakage results, grid range, unexplained remainder, grade vector, reproducibility block. Reserved five minutes. | Reporter (1) | assembles from `runs/` |

"Extend" is a flag that adds out-of-sample years and cost levels to the grid. It is not a
stage.

**What was deleted and why it is safe.** The two-extractor diff, the N-extraction diff, and the
figure overlay all defended against claim error; the page-image re-read defends against it
more directly. Era images and the image cache defended against environment error; a lockfile
of *our* environment plus "author code installs its own requirements or fails visibly" covers
the same ground for a fraction of the maintenance. The eight families and the DAG engine were
optimizations of a thing that did not yet exist. The cost, subperiod, and overfitting checks
exist in Lacuna with an audit format better than the one this document sketched.

### 9.4 Add-backs, with the reason each earned its way in

If nothing comes back, not enough was deleted. These four come back, three of them from
evidence in the repositories rather than from this document's reasoning:

1. **`reproduce.sh` re-run in a fresh container, after the builder is done, is the only
   source of reported numbers.** From PaperBench. Why: it prevents hard-coded results and
   numbers copied from the agent's session, and it is the mechanical form of "the output runs
   without the harness." Cost: one container start. Earlier draft: none.
2. **The builder cannot declare itself finished.** From PaperBench's IterativeAgent result:
   removing the ability to end early took one model from 13% to 24%. Why: "finished early
   claiming done" is the dominant agent failure, and the orchestrator already owns the clock,
   so owning the stop is free. The builder works piecemeal through the template's node list;
   the stage ends when the nodes pass or the budget does.
3. **An author-code blacklist with a post-hoc log monitor**, in re-implementation mode. From
   PaperBench. Why: the report says which kind of test was run, and "re-implementation" is a
   lie if the builder read the author's repository. The monitor scans the sandbox's network
   and file logs; a hit downgrades the kind of test to "reproduction" rather than zeroing the
   run, since for this product the information is still useful.
4. **OSAP's `SignalDoc.csv` as the finance spec-extraction eval and the KB seed, and
   PaperBench's rubrics as the CS one.** Why: step 0 and the fastest eval loop in §4 are
   already built by people with more domain knowledge than the model. The extractor is scored
   on 331 finance papers for key table, test type, sign, weighting, quantile, holding period,
   start month, filter, sample years, and the reported t-stat, in minutes, before any pipeline
   code exists. The outcome labels also come from there: OSAP's 1_good / 2_fair / 3_distant /
   4_lack_data map onto Match / Consistent / Mismatch / Untested, and their "1_clear / 2_likely
   / indirect" predictability labels are the claim-priority field.

Two things were considered and not added back:

- *Lacuna's UNKNOWN outcome* for a verify check that cannot be established. Considered because
  it is honest. Not added: "Untested" with a reason already covers it, and two words for one
  state is a bug.
- *REPRO-Bench's 112 instances as a calibration eval for the verdict engine.* Considered
  because it is the only verdict benchmark. Not added yet: social-science reproduction
  packages exercise none of the finance or CS plugins. Revisit when the verdict engine is
  isolated enough to run on a foreign package.

### 9.5 Step three: simplify what remains

- **Compare t-stats first.** OSAP and JKP both compare the original and replicated t-stat,
  which already has unit standard error. The derived-tolerance rule reduces to "within about
  2 of the paper's t-stat" for any claim reported with one, and the Sharpe and alpha rules are
  the same rule after a unit conversion. One rule, three units.
- **The anomaly template is six knobs.** Test type, weighting, quantile, holding period, start
  month, filter. The convention grid for the MVP is those six plus delisting handling and the
  return definition. Eight flips, one batch.
- **The spec is the plan.** Tolerances, tier, budget, and the frozen hash live in `spec.yaml`;
  there is no `plan.yaml`.
- **One human checkpoint**, showing headline claims with page images, the top ambiguities with
  their KB source, the tier, and the substitutions. Approve, edit, or stop. Stopping here *is*
  the dry-run mode.

### 9.6 Step four: accelerate the cycle

The cycle that matters is "change a prompt, know whether it got better." After the cut:

| Loop | Ground truth | Cost per iteration | Available |
|---|---|---|---|
| Spec extraction, finance | `SignalDoc.csv`, 331 rows | minutes, a few dollars | today |
| Spec extraction, CS | PaperBench rubric trees, 20 papers | minutes | today |
| Verify on a known pipeline | The three hand replications | seconds | after build step 1 |
| Leakage tests | Deliberately leaked fixtures (a signal built from t+1) | seconds | after build step 3 |
| End to end | Hermetic benchmark entries | hours, hundreds of dollars | nightly, after step 5 |

The first two rows are why the build order changes: the extractor is the first thing built
*and* the first thing measured, against ground truth nobody on this project has to write.

### 9.7 Step five: automate, last

The builder loop is the automation. It is already last in the build order, after the spec,
the data adapters, the leakage tests, and the verify stage each work and are measured on their
own. Automating the loop before those exist would automate a process that is not yet known
to work, which is the error the order exists to prevent.

### 9.8 Revised build order (supersedes §5 for the MVP)

0. Download `SignalDoc.csv` and the JKP documentation; write the anomaly-family KB from them.
   Download PaperBench rubrics; write the CS checkpoint-eval KB. Days, not weeks.
1. Spec extractor, scored against both. Iterate until the finance field accuracy is high
   enough that a human checkpoint is a review, not a rewrite.
2. Two hand replications: one OSAP anomaly on open data (French library plus Tiingo), one
   PaperBench paper with a released checkpoint. The hybrid Gu–Kelly–Xiu paper is third,
   after the loop exists.
3. Adapters, pandera schemas, Table 1 checkpoint, the three leakage tests, Lacuna wiring.
   All testable without an agent.
4. Verify and Report on the hand pipelines, including `reproduce.sh` in a fresh container.
5. Builder loop for the anomaly family with the orchestrator owning the stop and the clock.
6. Benchmark harness in Inspect over the hand entries; grow toward ten; perturbed-paper probe.
7. Then, in order of measured need: Hamilton if the grid is slow, the bootstrap for overlapping
   returns, more families, oracle mode, Tier B substitution registry, era images if a CS paper
   ever actually needs one.

---

## 10. Implementation status (updated 2026-09-15)

The §9 MVP cut is implemented in `replicator/`; `README.md` has layout, commands, and the
results table. Repository: https://github.com/victorzhu443/paper-replication-agent.

| Piece | Status | Where |
|---|---|---|
| Contracts, freeze, run records, report schema | built, tested | `replicator/schema.py` |
| Extractor → referee → page re-read; strict structured outputs for small schemas, schema-guided JSON with lenient parse and repair rounds for the full spec | built, exercised live on 15 papers | `replicator/spec/`, `replicator/llm.py` |
| Conventions KB (2 finance + 2 CS families) seeded from OSAP SignalDoc | built | `replicator/kb/` |
| Triage: probe, two-axis tiers, derived tolerances, budget, freeze, dry-run; author code not tiered as data; missing sources make claims Untested rather than sinking the paper | built, tested | `replicator/triage.py` |
| Data: content-addressed cache, pandera contracts, `available_at` lineage, Table-1 checkpoint; adapters for French library, Hugging Face datasets (Hub probe), torchvision, local file | built, tested live | `replicator/data/` |
| Anomaly template (pre-formed portfolios and stock-level sorts; cohort-based inference for overlapping holding periods), CS `reproduce.sh` contract with process-group timeouts | built, tested | `replicator/templates/` |
| Builder: bounded tool loop, orchestrator-owned stop, smoke gate on budget expiry, 5-minute call cap, subprocess sandbox, blacklist monitor | built; exercised live (momentum: 11 turns, $0.93, grade A) | `replicator/build/` |
| Verify: one Match rule, shuffle + future-perturbation + `available_at` tests, convention grid, reduced-scale comparisons marked not comparable | built, tested | `replicator/verify/`, `orchestrator.py` |
| Orchestrator with budgets, report-always, grade vector, variant cap | built, tested end to end | `replicator/orchestrator.py`, `report.py` |
| Spec-extraction eval against SignalDoc; 14-paper resumable sweep | built | `evals/` |
| Hand replication 1: Jegadeesh–Titman momentum on French deciles | grade A, live builder and prebuilt | `papers/momentum_french/` |
| Hand replication 2: LeCun 1998 MLP-300 on MNIST, 3 seeds | grade C; claim value flagged as recalled | `papers/mnist_mlp/` |
| 14-paper CS sweep (Transformer, ResNet, LayerNorm, BatchNorm, GAN, DQN, PPO, World Models, Lottery Ticket, LoRA, DPO, Superposition, Circuits, ROME) | specs extracted for all ($32.90); Transformer grade C at reduced scale; rest in progress | `papers/batch/` |

Observed on the sweep and folded back into the design: model-call stalls must fail fast (§0.2's
budget is only enforceable if a single call cannot consume it); the builder must be made to test
early (§9.4 add-back 2 is necessary but not sufficient, the orchestrator also runs the gate on
expiry); reduced-scale results are values, not verdicts, unless the paper reports the same
configuration at that scale (§2 stage 3, compute axis).

Deferred per §9: Tiingo/WRDS adapters, contamination scan, block bootstrap, Hamilton, Docker
sandbox, era images, Extend flag.
