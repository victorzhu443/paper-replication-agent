# Research replication agent

Reproduces a technical paper's headline results with an **attributable verdict**, for two tracks:
quant-finance (cross-sectional anomalies, ML return prediction) and CS/ML (checkpoint evaluation,
train-and-eval). The deliverable is not the number; it is the report that says which number was
obtained, under which conventions, whether it could have leaked, and what remains unexplained.

- `DESIGN.md` — the design, derived from first principles; §9 is the MVP cut this code implements,
  §10 the implementation status.
- `DESIGN_REVIEW.md` — the adversarial review (11 roles, 57 items) that produced §9.

## Results so far (2026-09-17, CPU-only laptop)

Two papers reproduced to their published claims by agent-written code, one per track: Jegadeesh–Titman
momentum (finance, A) and Batch Normalization at the paper's own MNIST configuration (CS, A). The 14
CS papers at reduced scale reproduced their mechanisms (C); at paper scale on this CPU, Superposition
matches 5 of 7 exact geometry values, Lottery Ticket does not fit the compute budget, and the rest
need a GPU.

<!-- RESULTS:BEGIN -->
| Paper | Track | Kind of test | Tiers | Grade | Headline claims | Leakage | Cost |
|---|---|---|---|---|---|---|---|
| Jegadeesh–Titman 1993 momentum (OSAP Mom12m as ground truth) | finance | re-implementation, live builder | A/1 | **A** | 1.32 vs 1.31 %/mo Match; t-stat 4.76 vs 3.74 Match | shuffle P, future-perturbation P | $0.93 |
| LeCun 1998 MLP-300 on MNIST, 3 seeds | CS | re-implementation | A/1 | C | 8.5% vs 4.7% error; convention grid 1.7–9.8% | shuffle P | $0 |
| Vaswani 2017 Transformer | CS | re implementation | A/2 | C | attention 99.4% vs no-attention 9.9% token accuracy on a copy task; WMT BLEU not comparable · c_t2_big_ende_bleu Mismatch; c_t2_big_enfr_bleu Mismatch | shuffle P | $4.72 |
| He 2015 ResNet | CS | re implementation | A/2 | C | build passed; full CIFAR runs exceeded the CPU time cap (fixed by the SCALE probe for later papers) | — | $4.90 |
| Ba 2016 LayerNorm | CS | re implementation | A/2 | C | MNIST MLP baseline 98.4%; LN vs baseline comparison ran | shuffle n/a | $10.73 |
| Ioffe 2015 BatchNorm | CS | re implementation | A/2 | C | reduced scale: BN 96.5% vs no-BN 91.7% at 10k steps; at paper scale (below) grade A | shuffle n/a | $2.76 |
| Goodfellow 2014 GAN | CS | re implementation | A/2 | C | MNIST GAN trains; shuffle test passes | shuffle P | $2.74 |
| Mnih 2013 DQN | RL | conceptual replication | B/2 | C | CartPole DQN mean return 226, best episodes 500 | shuffle n/a | $2.35 |
| Schulman 2017 PPO | RL | conceptual replication | B/2 | C | CartPole 500/500 on 3 seeds; clipping beats no-clip on 2 of 3 · t1_clip_eps02 Consistent; t1_clip_eps01 Consistent | shuffle n/a | $1.82 |
| Ha 2018 World Models | RL | conceptual replication | B/2 | C | VAE loss 3169→0.7; MDN-RNN and CMA-ES controller beat random policy (t=23) | shuffle P | $17.15 |
| Frankle 2019 Lottery Ticket | CS | re implementation | A/2 | C | winning tickets beat unpruned by 0.25–0.3 pts; early-stop speedup 2–3× | — | $4.75 |
| Hu 2021 LoRA | CS | re implementation | A/2 | C | RoBERTa-base on SST-2 subset: LoRA 91.97% vs full FT 92.66% with 0.3M trainable params (paper's claim) | shuffle n/a | $5.75 |
| Rafailov 2023 DPO | CS | re implementation | A/2 | C | DPO loss from Eq. 7: 98.5% held-out preference accuracy, 100% win rate vs reference, reward margin grows | shuffle P | $4.40 |
| Elhage 2022 Toy Models of Superposition | CS | re implementation | A/2 | C | n=20, m=5: dense regime 5 features, sparse (S=0.99) 12.9 features in 5 dims; linear never superposes; antipodal pair 0.498 (paper 1/2) | shuffle n/a | $4.21 |
| Elhage 2021 Transformer Circuits | CS | re implementation | A/2 | C | induction heads emerge in the 2-layer model: second-half loss 0.018 vs 3.3 first half (gap 3.28); 1-layer gap only 0.92 | shuffle n/a | $2.59 |
| Meng 2022 ROME | CS | re implementation | A/2 | C | GPT-2 causal tracing: MLP restoration at the last subject token carries the effect (18.9 vs 0.77 for attention), peaking at an early layer | shuffle n/a | $9.80 |

Sweep total: $78.70 in model calls for 14 papers. Reports: `papers/batch/<slug>/REPORT.md`. Updated 2026-09-17 17:35.

### Paper-scale runs on this machine (compute tier 1 attempt)

| Paper | Grade | Headline claims | Leakage | Cost |
|---|---|---|---|---|
| Ioffe 2015 BatchNorm | **A** | mnist_bn_gt_nobn_final Match (0.00883 vs 0); mnist_bn_faster_to_baseline_acc Match (1.08e+04 vs 5e+04); mnist_bn_activation_stability Match (0.264 vs 0) | shuffle P | $1.99 |
| Frankle 2019 Lottery Ticket | not graded | all Untested · failed: RuntimeError: build did not pass smoke: {'passed': False, 'p | — | $17.04 |
| Elhage 2022 Toy Models of Superposition | C | c_dstar_sticky_half Mismatch (0.505 vs 0.5); c_dstar_dense_one Match (1 vs 1); c_adversarial_vulnerability_3x Mismatch (1.46 vs 3); c_dim_tetrahedron Match (0.75 vs 0.75) | shuffle n/a | $6.90 |
<!-- RESULTS:END -->

**Reading a grade.** The letter is a fixed function of four axes (data fidelity, procedure fidelity,
result, integrity). A: headline claims Match within pre-registered tolerances, leakage tests pass,
open data. B: Consistent with attributed gaps. C: mechanics verified, numbers not (the usual outcome
for a paper whose numbers need GPUs when you have a CPU). F: a leakage test failed or the pipeline
did not run. Not graded: the harness or the network failed before the paper was tested. `REPORT.md` always states the *kind of test* (reproduction / re-implementation /
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
# on a laptop, keep it awake: subprocess timeouts use monotonic time and do not advance in sleep
caffeinate -i uv run python -m evals.batch
uv run python -m evals.batch            # all
uv run python -m evals.batch ppo gan    # subset
```

Interactive mode shows one checkpoint (headline claims with page images, ambiguities by
sensitivity with their source, tier, substitutions) after triage; `--batch` auto-approves;
`--dry-run` stops there. Every run writes `REPORT.md`, `report.json`, `spec.yaml` (frozen),
`runs/*.json` (one per execution), `llm/llm_calls.jsonl` (stage, tokens, cost per call), and
`work/` (the generated code and the sandbox command log).

## Running at paper scale on a GPU

The CPU sweep grades every CS paper C because their numbers are not comparable at reduced
scale. On a GPU the agent runs the papers' own configurations (`REPLICATOR_GPU=1`; per-paper
plans in `evals/batch.py` GPU_HINTS) and the verdicts become real Match/Mismatch.

```bash
# on a fresh Ubuntu GPU box (Lambda, RunPod, Vast, AWS g5/g6, or your own NVIDIA machine)
git clone https://github.com/victorzhu443/research-replication-agent && cd research-replication-agent
bash scripts/gpu_bootstrap.sh            # drivers check, uv, CUDA torch, RL extras, tests
export ANTHROPIC_API_KEY='sk-ant-api03-...'
bash scripts/gpu_run.sh                  # all 14 in tmux; or name papers: bash scripts/gpu_run.sh batchnorm lora
```

Rough single-GPU budget (24 GB card): about 70 GPU-hours for the 14 papers, so on the order of
$50–100 of GPU time plus ~$60 of model calls. Put `data_cache/` on a persistent volume so
datasets survive instance restarts. Full WMT14 Transformer training and Atari-scale DQN remain
multi-GPU or multi-day jobs; their plans compare against the paper's smaller reported points.

## What a run costs and how long it takes

Measured on the papers above with Claude Opus 5 (extraction, builder) and Claude Sonnet 5
(claim re-reads):

| Stage | Typical | Notes |
|---|---|---|
| Spec (extract + referee + re-read) | $1–3.5, 5–10 min | scales with paper length; ROME (62 claims) was $3.53 |
| Build (live builder) | $1–5, 3–40 min | finance template: 11 turns, 3 min; CS from scratch: up to the 40-min budget |
| Run + verify | $0 | CPU time only; per-run timeout 15–25 min in the sweep |

## What the sweep taught (fixes now in the code)

- **A verifier must check its own preconditions or abstain.** Six papers were graded F for
  leakage that did not exist because the shuffle test trusted properties of generated code it
  had never checked. The smoke gate now runs the script with shuffled labels and at SCALE=0.1
  and refuses the build until it acknowledges the shuffle, reports a no-skill reference, and
  echoes its scale; at verify time a missing precondition means "not judged", never F.
  `tests/test_cs_verify.py` has a fixture for every false-verdict mode the sweep produced.

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
