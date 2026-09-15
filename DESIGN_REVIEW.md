# Adversarial review of DESIGN.md

Date: 2026-09-14. Method: every section of DESIGN.md was attacked from eleven roles, asking
"why" until the chain reached either an unstated assumption or a decision only the owner can
make. Each item ends with one disposition:

- **Fixed** — resolved by an edit to DESIGN.md (the pre-review text is kept at `DESIGN.v1.md`)
- **Decision** — needs the owner; mirrored in DESIGN.md §7 with a recommendation in Part 4 below
- **Noted** — recorded so it is not re-litigated; no change, or the change is a later artifact

---

## Part 1. Three root causes

Most of the fifty-odd specific attacks below trace to three root issues.

### R1. The document has no organizing principle for the agent's own incentives

The doc knows the dominant failure is a false positive (§0.4) and has pieces of the defense:
pre-registered tolerances, a blinded reviewer, a shuffle test. But it never states the principle
those pieces serve, so it applies them inconsistently:

- **Stage 9 was a sequential search over conventions that stops on Match.** With eight binary
  ambiguities there are 256 configurations. Picking the one closest to the paper, on the same
  data, is exactly "getting the number for the wrong reason." The doc's own §0.4 warns against
  this and then builds it.
- **The benchmark's headline metric was Match rate**, which rewards loose tolerances and rewards
  the search.
- **The example budget gave verification 5 minutes** while stage 8 lists three seeds, a shuffle
  re-run, a bootstrap, subperiods, cost sensitivity, and a figure overlay. Each is "cheap"; the
  sum is not.
- **Tolerances were "set in the spec"** by the same extraction call whose output is later graded
  against them. Pre-registration by the party being graded is only pre-registration if it
  cannot be undone.

Why did this happen? "Pre-registration" was treated as a *timing* property (decide early)
rather than a *commitment* property (cannot be changed after). Timing is easy for an agent to
satisfy and meaningless; commitment needs a mechanism.

→ **Fixed.** New principle §0.7 (a replication is a hypothesis test run by an interested party).
Freeze mechanism in stage 3. Derived, not chosen, tolerances in stage 8. Stage 9 rewritten as a
parallel convention grid (sensitivity) plus bisection over paper-reported intermediates
(attribution). §4 north star changed to calibration with a false-Match rate that must be near
zero.

### R2. Attribution was treated as free

`Y' − Y = (a)+(b)+(c)+(d)+(e)+(f)` reads like an equation to solve. It is one observed number
and six unknowns. The terms interact (a data substitution changes which procedure ambiguities
matter). And (f) "errors in the paper" was defined as *the residual*, which is where every bug
the agent did not find in its own code lands. Three diagnose iterations cannot attribute among
six terms; that is arithmetic, not pessimism.

Two terms were also missing. Term (0), **claim error**: we mis-read the target (wrong row, wrong
units, annualized vs. monthly, arXiv v1 vs. v3). It sits upstream of everything and nothing
downstream can detect it. Term (g), **fragility**: the paper's number is real on its data but
sensitive to defensible conventions, so any re-implementation lands elsewhere. This is not (b)
and not (f); the paper is one point in a range.

→ **Fixed.** §0.1 adds (0) and (g); states that a term is attributed only by an intervention
that moves it alone or by independent evidence; reserves "paper error" for positive evidence;
names the remainder "unexplained."

### R3. Missing information was treated as recoverable by reasoning

What a paper omits cannot be inferred by thinking harder. It is filled from (i) the field's
conventions, (ii) author code or data, or (iii) a guess labeled as one. That makes the
conventions knowledge base the core asset of the system, and it must exist before the first
agent run, built by hand from the replication literature. The doc had it as "cross-run memory"
that accretes later.

→ **Fixed.** §0.8; build order step 0; §3.5 seeded by hand. §0.8 also names the three separable
products the doc bundles (spec extractor, replication engine, verdict engine), which clarifies
what has standalone value and what the MVP actually is.

---

## Part 2. Attacks by role

### Replication scientist / referee

1. **"Replication" names four different tests.** Author code on the same data is a
   *reproduction*. Our code on the same data is a *re-implementation*. Our code on substituted
   data is a *conceptual replication*. Synthetic data is *mechanics only*. The verdict means
   something different in each, and a reader cannot tell which they got. → **Fixed**: stage 3
   names them; the report's first line states which.
2. **The claims schema had one global `method` block.** Papers report claims under different
   procedures: Table 3 value-weighted, Table 4 equal-weighted, Table 5 with a different sample.
   A claim that cannot say which procedure produced it cannot be verified. → **Fixed**:
   `method.variants` and `claim.method_variant`.
3. **`claim.sample.universe` duplicated `data.universe`** as a string next to a struct. Two
   representations of one fact drift. → **Fixed**: removed from the claim.
4. **Term (0) was absent.** Wrong row, wrong units, wrong version, and everything downstream
   compares against the wrong number with perfect confidence. → **Fixed**: independent
   page-image re-read of every claim value; `units`, `reported_precision` fields; the exact
   paper version recorded at intake.
5. **Tier B success "same sign and order of magnitude" is vacuous.** A Sharpe of 0.05 and one
   of 0.41 share an order of magnitude. → **Fixed**: same sign, significant in our run, and
   inside the band the substitution registry predicts.
6. **The substitution registry's "known consequences (~X bps/month)": known how?** Measuring the
   consequence of CRSP→yfinance requires CRSP. Without a provenance, the registry is a guess
   that reads like a citation. → **Fixed**: each entry carries its source (literature, one
   measured run with a date, or "unknown," in which case the gap stays unexplained).
7. **(f) as the residual.** → **Fixed** (R2).
8. **Ten rows of one table counted as ten pieces of evidence.** They share the data and the
   sort. → **Fixed**: per-claim outcomes plus one joint verdict per finding.

### Statistician

9. **Tolerances were arbitrary and unit-blind.** Sharpe ±0.1: monthly or annualized? The
   standard error of a Sharpe ratio is roughly sqrt((1 + SR²/2)/T). For SR 0.41 monthly over
   678 months that is about 0.04, so ±0.1 is 2.5 SE, right by luck. Over a 10-year sample it is
   0.5 SE and would call noise a mismatch. → **Fixed**: Match ⇔ |Y' − Y_read| ≤ 2·SE +
   reported_precision, computed in stage 3 and frozen.
10. **Two comparison rules were listed (tolerance and bootstrap CI) with no precedence.**
    → **Fixed**: one rule. Bootstrap only where the analytic SE is wrong (autocorrelation,
    overlapping returns), and the CI width is printed so a Match from a wide CI is visibly
    uninformative.
11. **The shuffle test was claimed as a general leakage detector. It is not.** Shuffling
    labels within period detects *label* leakage (bad split, target encoding, test rows in
    train). The dominant finance false positive is *feature* look-ahead: a signal at t built
    from the return at t+1. Shuffle the returns and that link breaks, the result vanishes, and
    the test *passes* on a leaking pipeline. → **Fixed**: shuffle scope stated honestly; a
    future-perturbation test added (replace all data after t with noise, assert features at or
    before t are bit-identical); `available_at` lineage made a library feature, not a model
    discipline.
12. **CS rule "beat the baseline by at least half the reported margin": why half?**
    → **Fixed**: the gap must exceed the pooled seed SE.
13. **Survivorship check "count disappearances; zero means biased."** A source that drops
    delisted tickers entirely shows zero disappearances either way. → **Noted**: the
    firm-count-per-year comparison against Table 1 already covers this and is the check that
    works; the disappearance count stays as a cheap first signal.

### Quant PM (the practitioner who would actually use this)

14. **Who is the user?** Never stated. A PM wants "does this work after costs, now" and treats
    the paper's number as a sanity check. An academic wants the attribution. An ML engineer
    wants the code. The doc's own stage 10 says out-of-sample extension is "the single most
    requested thing" and then makes it leftover budget. That sentence is evidence about the real
    user that the design ignores. → **Decision** Q6; note added in stage 10.
15. **Cost sensitivity needs turnover, which needs holdings.** Not every template produces
    them. → **Noted**: a template requirement for the finance families.
16. **The cheapest useful product is a dry run:** tier, plan, and cost estimate before spending
    an hour and a bill. → **Fixed**: dry-run mode in stage 3.

### ML researcher

17. **What does the CS track deliver in 1–2 hours?** "Oracle-first" evaluates the authors'
    released checkpoint. That verifies a checkpoint, not a method. Training reproduction is
    compute tier 2 for almost every paper of interest. The doc never says what CS success looks
    like inside the budget. → **Decision** Q8.
18. **Era-pinned images for CS are often infeasible.** Old torch/CUDA wheels carry no kernels
    for current GPU architectures. Era feasibility is a property of the hardware. → **Fixed**:
    stage 4 caveat; the compute envelope must be declared.
19. **Era pinning for finance is mostly pointless.** *Our* reproducibility needs a lockfile of
    whatever we ran; the paper's era matters only for author code. → **Fixed**: two
    environments, two purposes.
20. **Three seeds give a noisy std.** Fine, but the rule was unstated. → **Fixed** via the
    derived tolerance.

### Software architect

21. **Stage 1 produces `paper.md`; stage 2 sends the PDF.** Two sources of truth, and the doc
    never said which the spec is checked against. → **Fixed**: the PDF is the source, the
    markdown is derived, and each consumer of the derived artifacts is named.
22. **"Every stage boundary has a schema and a checker."** Only `spec.yaml` is sketched.
    `plan.yaml`, `data_manifest.json`, `results.json`, `verification.json`, `report.json` have
    none, and no checker has a number ("data checkpoint fails" at what deviation?). → **Fixed**
    in part: thresholds live in the family template, not in run-time judgment. **Noted**: the
    schemas are the next design artifact (Part 3).
23. **A hash-cached DAG of pure nodes is a workflow engine.** Do not build one. → **Fixed**:
    Hamilton, Kedro, snakemake, or joblib.Memory; the design work is node contracts.
24. **`ask_user(q)` is a sandbox tool, but §3.4 says exactly two human checkpoints.** A model
    that can block on a human mid-build has an unbounded budget; an unattended run hangs.
    → **Fixed**: callable only at the two checkpoints; elsewhere it logs the question and
    returns the spec default.
25. **Who fills the budget table in `plan.yaml`?** If the model, principle 0.3 is violated. If
    code, by what rule? → **Fixed**: code, from family defaults; the model may request a
    reallocation with a reason, granted against a cap.
26. **Tiers conflated data access with compute.** An ML-on-CRSP paper is both substituted and
    reduced; one letter hides which drove the outcome. → **Fixed**: two axes.
27. **N-version programming by one model.** "The two are unlikely to share the same bug" is
    false for an LLM writing both: same priors, same `pct_change` default, same lag convention.
    → **Fixed**: the caveat is stated; independence against misreading comes from a
    hand-computed golden fixture or a different model.
28. **The output must run without the agent.** If the DAG runner, adapters, and lineage checks
    are harness internals, the "runnable replication" is not runnable outside the harness.
    → **Fixed**: §3.1 states the output repo depends on a published library.

### Infra / SRE

29. **No compute envelope anywhere.** Tier D, "seeds in parallel," "env ∥ data," and compute
    triage are all undefined without one. → **Decision** Q9; stage 3 now requires it declared.
30. **A budget kill at minute 115 could produce nothing.** → **Fixed**: the last five minutes
    are reserved and un-killable; a report is always produced.
31. **Smoke config "one year" yields zero portfolio months** for a sort with a 12-month
    lookback. → **Fixed**: the family template defines the smoke sample.
32. **The benchmark hits yfinance live.** Scores would drift because Yahoo changed, and rate
    limits would fail runs at random. → **Fixed**: hermetic benchmark with frozen snapshots.
33. **Verification was outside the budget.** → **Fixed**: §0.2 and the budget table.

### Security and compliance

34. **Author code from an arbitrary repository runs in the sandbox, and the sandbox is where
    data is fetched, so WRDS credentials are in reach of that code.** → **Fixed**: credentialed
    adapters run in the harness process; the sandbox sees only cached parquet.
35. **The paper, its README, and its code are untrusted text the model reads.** A footnote
    saying "report Match" is a prompt injection. → **Fixed**: §3.1 note; the verdict must not
    depend on anything the paper says about itself; the reviewer gets code, not prose.
36. **Licensing.** yfinance is unofficial scraping under terms that forbid non-personal use;
    vendor contracts forbid sharing extracts; a content-addressed cache shared across users is
    redistribution. → **Fixed**: per-user cache for licensed sources; source licenses in the
    report. **Decision** Q10 on which sources a product may use.

### LLM / agent engineer

37. **"N extractions diffed = ambiguities by construction" is wrong.** Disagreement measures
    model instability. A convention the model is confidently and consistently wrong about
    produces no disagreement, and that is the case that matters. → **Fixed**: caveat; it
    supplements the family's known-ambiguity checklist rather than replacing it.
38. **The blinded reviewer shares the generator's priors and has low recall over thousands of
    lines.** → **Fixed**: a supplement, never the only leakage check; different model family
    where possible; the ambiguity list as its checklist.
39. **What does the human see at the checkpoint?** Thirty ambiguities in YAML is not a
    checkpoint. → **Fixed**: headline claims with page images, ambiguities sorted by
    sensitivity with default and source.
40. **Stages 1–3 latency.** Two or three high-effort whole-paper calls, a critique pass, and
    page re-reads can be 20–30 minutes before any code exists, and the budget table did not
    include them. → **Fixed**: the budget table now has an intake+spec+triage line.
41. **Prompt caching across separate sessions with different system prompts** only works if the
    paper is the stable prefix. → **Noted**: implementation detail for §3.6.

### Eval / benchmark scientist

42. **Match rate as north star.** → **Fixed**: calibration first; false-Match rate is the
    number that must be near zero.
43. **Contamination is the default.** The model has read the momentum paper, Fama–French, the
    OSAP code, and PaperBench. A high score on famous papers measures recall. → **Fixed**:
    stratify by fame; include recent and obscure papers; report strata separately; a
    perturbed-paper probe (alter the headline number in a copy; an agent that matches the
    altered number is reading, not computing).
44. **Finance ground truth is CRSP-based** (OSAP, JKP). A Tier A open-data run cannot be scored
    against it. → **Fixed**: the ground truth for open-data benchmark papers is our own
    human-verified hand replication on that open source.
45. **"Build the benchmark first" (§0.5) versus the harness at build step 6.** → **Fixed**: each
    hand replication is committed as a hermetic benchmark entry the day it is done; stage
    evals are the fast loop; end-to-end runs nightly in batch mode.
46. **Regression on every change at 1–2 hours × 20 papers is unaffordable.** → **Fixed**:
    tiered regression.

### Product and cost owner

47. **Cost was never a constraint.** An hour of high-effort agentic loop is tens of dollars; a
    benchmark sweep is thousands. → **Fixed**: cost is a co-constraint with a per-run cap.
48. **Why 1–2 hours?** The number drives triage, caps, and smoke runs, and is never justified.
    If a person is waiting, human checkpoints count against it. If it is cost, the constraint is
    dollars. If neither, batch mode with more seeds is strictly better. → **Decision** Q7.
49. **Both tracks now doubles the benchmark, the conventions KB, and the hand replications
    before there is a user for either.** "One pipeline" is the right architecture and is not an
    argument for shipping both plugin sets. → **Decision** Q8.
50. **Operator-in-the-loop versus unattended is a false dichotomy.** → **Fixed**: two modes;
    batch exists from day one because the benchmark needs it.

### Red team (how the agent fools you, or itself)

51. Search until it matches. → **Fixed** (stage 9).
52. Loosen the tolerance after seeing the number. → **Fixed** (freeze).
53. Bootstrap a wide CI so everything is inside it. → **Fixed** (CI width shown; analytic SE
    is the default).
54. Data checkpoint in Tier B always fails, so either every run halts or the agent learns to
    soften the check. → **Fixed** (gate in A, measurement in B).
55. The reviewer confirms the generator. → **Fixed** (supplement; different model).
56. The model "knows" the answer from training and the benchmark cannot tell recall from
    computation. → **Fixed** (perturbed-paper probe; fame strata).
57. Feature look-ahead passes the shuffle test. → **Fixed** (future-perturbation test).

---

## Part 3. What the design still does not specify

These are the next artifacts. None is a flaw in the document; each is work the document now
correctly points at.

- JSON schemas for `plan.yaml`, `data_manifest.json`, `results.json`, `verification.json`,
  `report.json`, and the conventions KB entry format.
- The first family template, cross-sectional anomaly: node list and contracts, checkpoint
  thresholds, smoke sample, hypothesis list with initial hit rates, holdings output for
  turnover. Build steps 0 and 1 produce it.
- The compute envelope config and the rule that maps reported compute to compute tier 2.
- The MVP adapter list with each source's license.
- The exact SE formula per metric type in the derived-tolerance rule, including the CS metrics.

---

## Part 4. Decisions for the owner, with a recommendation each

Mirrored in DESIGN.md §7.

| # | Decision | Recommendation | Why |
|---|---|---|---|
| Q2 | WRDS access | User brings credentials; the product is open-data by default | Anything else makes the substitution registry unmeasurable and the product a WRDS front-end |
| Q4 | Three hand papers | All Tier A on the data axis; hybrid third, not first | The hybrid is the hardest paper; hand-replicating it first produces a template from the worst case |
| Q5 | Engine | Claude Code sessions per stage for the MVP | The doc's own reasoning is right; the loop is not the hard part |
| Q6 | Who is the user; match the paper or "does it work" | Practitioner first; "match" is the sanity check, stage 10 is the product | The doc's own evidence: OOS extension is the most requested thing |
| Q7 | Why 1–2 hours | Interactive mode: a wall-clock cap because a person waits. Batch mode: a cost cap only | Makes the number mean something and removes it where it means nothing |
| Q8 | Both tracks or finance first | Finance first behind track-agnostic interfaces; define CS success before building CS plugins | Halves the benchmark and KB work; CS at 1–2 h has no stated deliverable yet |
| Q9 | Compute envelope | Declare one machine now (cores, RAM, GPU or none) | Compute tier, era feasibility, and parallel seeds are undefined without it |
| Q10 | Data licensing | yfinance for personal/dev only; a licensed source (Tiingo, Polygon, Sharadar) for anything shared | Terms of service, and a cache that is shared is redistribution |

---

## Part 5. Owner decisions received (2026-09-14, mid-review)

The owner answered four of the Part 4 questions while the review was in progress; DESIGN.md
was updated accordingly.

| # | Decision | Where it landed |
|---|---|---|
| Q6 | Users are a quant firm (reproduce before acting) and an ML research lab (reproduce before building on); practitioner-first | Goal; §3.8 deployment profiles; stage 10 first-class for the quant profile |
| Q7 | 1–2 h is the productivity target: 20–30 researcher-hours → 1–2 h turnaround, ~15 min of human attention | §0.2 derivation; hard kill configurable (default 4 h); batch mode cost-capped |
| Q8 | Both tracks, because the two users split along that line | §3.8; CS success definition required before CS plugins |
| Q11 (new) | One agent or many? | §3.7: one product, one orchestrator, ten model roles that are separate contexts by necessity and never talk to each other; only the builder and diagnoser are agentic loops |

Still open: Q2 (WRDS), Q4 (which three papers), Q5 (engine), Q9 (compute envelope per profile),
Q10 (licensed source list for the quant profile).
