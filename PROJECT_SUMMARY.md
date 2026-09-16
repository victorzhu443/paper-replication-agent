# Paper Replication Agent — project summary for a resume

Repository: https://github.com/victorzhu443/paper-replication-agent
Built September 12–15, 2026. Python 3.12, Anthropic Claude API (Opus 5 + Sonnet 5), pandas,
statsmodels, PyTorch, pandera, pydantic, Hugging Face datasets/transformers/peft, gymnasium.
About 3,400 lines of Python across 30 modules, 14 tests, 36 commits.

---

## One-paragraph description

An autonomous agent that takes a quantitative-finance or machine-learning paper (PDF or arXiv
id) and produces a runnable replication with an **attributable verdict**: which headline numbers
were reproduced, under which unstated conventions, whether the result could have come from data
leakage, and what remains unexplained. Designed from first principles for a quant firm reproducing
research before acting on it and for an ML lab reproducing a paper before building on it. The
design was stress-tested with an 11-role adversarial review and a Musk-style deletion pass before
implementation, then validated on 16 papers.

## Resume bullets (pick and trim)

- Designed and built an end-to-end LLM agent that reproduces research papers: PDF → structured
  replication contract → data acquisition → code generation in a sandbox → execution → statistical
  verification → graded report; 5 stages, 4 model roles, deterministic Python orchestrator.
- Reproduced Jegadeesh–Titman (1993) momentum to published numbers (1.32 vs 1.31 %/month,
  t-stat within tolerance) with the agent writing the pipeline itself in 11 turns for $0.93;
  the agent's re-implementation found and corrected an overlapping-returns inference error in the
  hand-written benchmark.
- Ran a 14-paper CS/ML sweep (Transformer, ResNet, BatchNorm, LayerNorm, GAN, DQN, PPO, World
  Models, Lottery Ticket, LoRA, DPO, Toy Models of Superposition, Transformer Circuits, ROME)
  end to end on a CPU laptop for about $65 in model calls; extracted structured specs for all 14
  (24–62 claims each) and reproduced the mechanism-level claims at reduced scale.
- Built the verification layer: pre-registered tolerances derived from standard errors and frozen
  by hash before any run; three mechanical leakage tests (label shuffle, future-perturbation for
  feature look-ahead, point-in-time lineage audit); a convention grid that reports the result's
  range across unstated conventions instead of searching for a match.
- Seeded a conventions knowledge base from Open Source Asset Pricing's 331-paper SignalDoc
  table and used it as ground truth for a spec-extraction evaluation.
- Engineered for production use: content-addressed data cache, schema contracts on every adapter,
  credentials isolated from the code sandbox, no data rows in prompts, per-call cost and time
  caps, process-group timeouts, resumable batch runner, report always produced on failure.
- Wrote the design document from first principles (error decomposition of a replication,
  budget-driven pipeline, agent-as-interested-party controls), ran an 11-role adversarial review
  producing 57 tracked findings, and cut the design with a five-step deletion pass (11 stages → 5,
  10 model roles → 4, 7 artifacts → 3) before writing code.

## Results table

| Paper | Track | Kind of test | Grade | What was reproduced | Cost |
|---|---|---|---|---|---|
| Jegadeesh–Titman 1993 momentum | finance | re-implementation (agent-written) | **A** | headline 1.32 vs 1.31 %/mo; t-stat 4.76 vs 3.74; leakage tests pass | $0.93 |
| LeCun 1998 MLP-300 on MNIST | CS | re-implementation | C | runs on 3 seeds; 8.5% vs 4.7% error; grid 1.7–9.8% explains the gap | $0 |
| Vaswani 2017 Transformer | CS | reduced scale | C | attention 99.4% vs no-attention 9.9% on a copy task | $6.18 |
| He 2015 ResNet | CS | reduced scale | C | build passed; CIFAR runs exceeded the CPU time cap | $4.90 |
| Goodfellow 2014 GAN | CS | reduced scale | C | MNIST GAN trains; shuffle test passes | $2.74 |
| Frankle 2019 Lottery Ticket | CS | paper scale (LeNet/MNIST) | C | winning tickets beat unpruned by 0.25–0.3 pts; early-stop speedup 2–3× | $4.75 |
| Ioffe 2015 BatchNorm | CS | paper scale (MNIST) | re-verifying | BN 96.5% vs no-BN 91.7% at 10k steps, both seeds | $2.76 |
| Ba 2016 LayerNorm | CS | reduced scale | re-verifying | MNIST MLP baseline 98.4% | $10.73 |
| Mnih 2013 DQN | RL | reduced scale (CartPole) | re-verifying | mean return 226, best episodes 500 | $2.35 |
| Schulman 2017 PPO | RL | paper-class task (CartPole) | re-verifying | 500/500 on 3 seeds; clip beats no-clip 2 of 3 | $1.82 |
| Ha 2018 World Models | RL | reduced scale | re-verifying | VAE loss 3169→0.7; controller beats random, t=23 | $17.15 |
| LoRA, DPO, Superposition, Circuits, ROME | CS | reduced scale | in progress | specs extracted | $13.40 |

"Re-verifying" rows were graded F by a shuffle-test bug (the test did not confirm the generated
script had actually shuffled labels, and used the wrong null for RL returns); the bug is fixed and
those papers are being re-scored without new model calls.

## How to read the grades

A: headline claims match within pre-registered tolerances, leakage tests pass, open data.
B: consistent with attributed gaps. C: mechanics verified, numbers not comparable (the normal
outcome when a paper's numbers need GPUs and the machine is a CPU). F: a leakage test failed or
the pipeline did not run. Every report states the kind of test (reproduction / re-implementation /
conceptual replication / mechanics only) and the compute tier, so a C is never mistaken for a
failed method.

## Architecture in five lines

1. **Spec**: one high-effort call reads the PDF into a contract (claims with page images, units,
   precision; ambiguities with defaults and their source); a referee call revises it; a cheap call
   re-reads every headline value from its page image. Triage probes data, assigns data and compute
   tiers, derives tolerances, and freezes the contract by hash.
2. **Setup**: lockfile, cached data through typed adapters with schema checks and point-in-time
   lineage, Table-1 checkpoint.
3. **Build + run**: a bounded tool loop writes the pipeline from a family template; the
   orchestrator, not the model, ends the stage (smoke gate, timed 10%-scale probe); runs are
   sandboxed with process-group timeouts.
4. **Verify**: one Match rule (2·SE + printed precision), three leakage tests, convention grid.
5. **Report**: grade vector → letter, always produced, with a reproducibility block (spec hash,
   data hashes, lockfile hash, every run's record, every model call's cost).

## Things worth saying in an interview

- Why the agent grades itself honestly: tolerances are derived, not chosen, and frozen before
  any run; searching conventions until the number matches is reported as a range, not a match.
- Why the shuffle test is not enough: a signal built from next month's return passes a label
  shuffle; the future-perturbation test (change the future, the past must not move) catches it.
- What the sweep taught: model calls can stall for 20 minutes; builders polish instead of testing;
  killing a shell does not kill its training process; a copy-task BLEU is not a WMT BLEU. Each
  became a mechanism in the orchestrator.
- The agent as reviewer: on the momentum paper it noticed that averaging overlapping K-month
  returns inflates a plain t-stat and computed the standard error from the non-overlapping cohort
  series instead, which the hand-built benchmark then adopted.
