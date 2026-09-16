"""Batch replication sweep: spec stage in parallel (API-bound), build/run/verify sequentially
(CPU-bound). Resumable: a paper with spec.yaml skips extraction; one with REPORT.md and no
failure is skipped entirely. Writes runs/batch/SUMMARY.md after every paper.

    uv run python -m evals.batch            # all
    uv run python -m evals.batch gan ppo    # subset
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from replicator.orchestrator import Orchestrator
from replicator.schema import Spec
from replicator.spec.intake import fetch_arxiv, fetch_html_as_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / ("batch_gpu" if __import__("os").environ.get("REPLICATOR_GPU") else "batch")

# slug -> paper. `hint` is the reduced-scale plan the triage/builder receive (compute tier 2, CPU only).
PAPERS: dict[str, dict] = {
    "attention": dict(arxiv="1706.03762", family="train_and_eval", seeds=[0], grid=False, timeout=1500,
        hint="CPU only, compute tier 2. WMT14 is out of reach: train a 2-layer d_model=128 Transformer on a synthetic "
             "copy/reverse task (or a <=20k-pair IWSLT-style subset if a download is fast) for <=15 min; report loss, "
             "token accuracy, and BLEU on the small task, plus a no-attention baseline. Paper-scale BLEU claims are Untested."),
    "resnet": dict(arxiv="1512.03385", family="train_and_eval", seeds=[0], grid=False, timeout=1500,
        hint="CPU only, compute tier 2. Section 4.2 CIFAR-10: ResNet-20 vs plain-20 and, if time allows, 56-layer variants on a "
             "10k-image subset for 3-5 epochs. The checkable claim at this scale is the *direction*: residual nets train "
             "faster/better than plain nets of the same depth, and deeper plain nets degrade. Paper's 8.75% is Untested."),
    "layernorm": dict(arxiv="1607.06450", family="train_and_eval", seeds=[0, 1], grid=False, timeout=900,
        hint="CPU only. Claimed effect: layer normalization speeds convergence. Reproduce on a small model the paper class "
             "covers (MLP on MNIST or a small GRU on a sequence task) with/without LN at matched steps; report train loss "
             "and test accuracy at fixed step counts."),
    "batchnorm": dict(arxiv="1502.03167", family="train_and_eval", seeds=[0, 1], grid=False, timeout=900,
        hint="CPU only. Section 4.1 / Figure 1: 3-layer fully-connected MNIST net (100 units/layer, sigmoid) with vs without "
             "BN; BN reaches higher test accuracy faster. Report accuracy after fixed step counts (e.g. 5k, 10k, 20k steps)."),
    "gan": dict(arxiv="1406.2661", family="train_and_eval", seeds=[0], grid=False, timeout=1200,
        hint="CPU only. MLP GAN on MNIST (<=15 min). Checkable: D loss near log 4 = 1.386 at equilibrium and samples "
             "classified as digits with high confidence by a small classifier trained in the same script; Table 1 Parzen "
             "log-likelihood 225 is Untested unless a Parzen estimate fits in time."),
    "dqn": dict(arxiv="1312.5602", family="train_and_eval", seeds=[0], grid=False, timeout=1500,
        hint="CPU only, compute tier 2. Atari-scale is out of reach: DQN with replay + target net on CartPole-v1 (gymnasium) "
             "for <=150k steps; checkable claims are mechanism-level: learning curve rises well above the random baseline; "
             "report mean return of last 50 episodes and random-policy return. Atari scores are Untested."),
    "ppo": dict(arxiv="1707.06347", family="train_and_eval", seeds=[0, 1, 2], grid=False, timeout=900,
        hint="CPU only. PPO-Clip (eps=0.2, GAE lambda 0.95) on CartPole-v1 for <=200k steps; compare clip vs no-clip. "
             "Checkable: reaches >=475 mean return and the clipped objective beats the unclipped (Table 1 direction)."),
    "worldmodels": dict(arxiv="1803.10122", family="train_and_eval", seeds=[0], grid=False, timeout=1500,
        hint="CPU only, compute tier 2. CarRacing is out of reach: use CartPole-v1 pixels or a tiny grid env. Three stages, "
             "each checked: VAE reconstruction loss falls; MDN-RNN next-latent NLL falls below a constant predictor; CMA-ES "
             "linear controller on (z,h) beats a random policy. Paper CarRacing scores are Untested."),
    "lottery": dict(arxiv="1803.03635", family="train_and_eval", seeds=[0, 1], grid=False, timeout=1500,
        hint="CPU only. LeNet-300-100 on MNIST, iterative magnitude pruning (20% per round) to ~10% weights remaining, "
             "2 epochs per round. Claim: winning tickets reset to the original init match/exceed the unpruned accuracy at "
             "10-20% weights while random-reinit tickets do worse. Report accuracy at each sparsity for both."),
    "lora": dict(arxiv="2106.09685", family="train_and_eval", seeds=[0], grid=False, timeout=1500,
        hint="CPU only. GPT-2 small (124M) via transformers+peft: LoRA r=4 on q,v vs full fine-tune on a 2k-example SST-2 "
             "subset (HF datasets) for <=10 min each; report accuracy and the trainable-parameter fraction. Claim: LoRA is "
             "within ~1 point of full FT with <1% trainable parameters."),
    "dpo": dict(arxiv="2305.18290", family="train_and_eval", seeds=[0], grid=False, timeout=1200,
        hint="CPU only. Implement the DPO loss (Eq. 7) exactly and unit-test it on a hand-computed case. Then a tiny policy "
             "(gpt2 or a small transformer) on a synthetic preference set (e.g. prefer completions with a target attribute); "
             "checkable: implicit reward margin rises and win-rate vs the reference model exceeds 50% and grows with beta."),
    "superposition": dict(arxiv="2209.10652", family="train_and_eval", seeds=[0], grid=False, timeout=900,
        hint="CPU only, exactly reproducible. Section 2 ReLU output toy model: n=20 features, m=5 hidden dims, importance "
             "0.7^i, sparsity sweep S in {0, 0.7, 0.9, 0.97, 0.99}; reproduce the phase change: dense regime represents only "
             "the top features, sparse regime packs many in superposition. Report features represented and W^T W stats."),
    "circuits": dict(url="https://transformer-circuits.pub/2021/framework/index.html", family="train_and_eval", seeds=[0],
        grid=False, timeout=1200,
        hint="CPU only. Attention-only transformers on a synthetic repeated-random-token task: a 2-layer model develops "
             "induction heads (prefix-matching + copying scores high; second-half loss far below first-half), a 1-layer "
             "model cannot do in-context copying. Report those scores and the loss gap."),
    "rome": dict(arxiv="2202.05262", family="checkpoint_eval", seeds=[0], grid=False, timeout=1500,
        hint="CPU only. GPT-2 small/medium from HF. Causal tracing (Sec 3) on ~10 facts: corrupt subject tokens, restore "
             "hidden states per (layer, token); checkable claim: the largest restoration effect at the last subject token is "
             "in early-middle MLP layers. Then one ROME rank-one edit: target-token probability rises after the edit."),
}


GPU = bool(__import__("os").environ.get("REPLICATOR_GPU"))

# Paper-scale plans for a single modern GPU (used when REPLICATOR_GPU=1). Hours are rough.
GPU_HINTS = {
    "attention": ("Full scale is 8 GPUs x 12 h. On one GPU: train the base Transformer on WMT14 EN-DE (HF wmt14 de-en) for as many "
                  "steps as the time budget allows and compare against the paper's learning-curve/dev PPL; report BLEU with sacrebleu on newstest2014.", 6.0),
    "resnet": ("Section 4.2 exactly: ResNet-20/32/44/56 and plain-20/56 on full CIFAR-10, 64k iterations, the paper's schedule; compare Table 6 errors.", 3.0),
    "layernorm": ("Section 6.6 permutation-invariant MNIST at full scale plus one RNN experiment (e.g. the order-embedding or skip-thought is too big; use the MNIST and an attentive-reader-style small LSTM).", 1.0),
    "batchnorm": ("Section 4.1 Figure 1 exactly: 3x100 sigmoid MLP on MNIST, 50k steps, with/without BN.", 0.5),
    "gan": ("MNIST MLP GAN as in the paper; Parzen-window log-likelihood estimate (Table 1: 225 +/- 2).", 1.0),
    "dqn": ("DQN on 2 Atari games (Breakout, Pong) with ale-py, 10M frames each or the time budget; compare Table 1.", 12.0),
    "ppo": ("PPO on MuJoCo is license-free now via gymnasium[mujoco]: HalfCheetah, Hopper, Walker2d, 1M steps, 3 seeds; compare Figure 3 / Table 1.", 4.0),
    "worldmodels": ("CarRacing-v3 (gymnasium[box2d]): 10k rollouts, VAE, MDN-RNN, CMA-ES controller; compare the 906 +/- 21 score.", 24.0),
    "lottery": ("LeNet-300-100 MNIST and Conv-2/4/6 on CIFAR-10 with iterative pruning, 5 trials; compare Figures 3-5.", 3.0),
    "lora": ("RoBERTa-base on full GLUE SST-2, MRPC, CoLA, RTE with LoRA r=8 vs full FT; compare Table 2.", 3.0),
    "dpo": ("GPT-2 large on IMDb sentiment (Section 6.1 setting) with the released preference recipe; reward-KL frontier vs PPO not required.", 4.0),
    "superposition": ("Exact: already paper scale on CPU; rerun with matched_scale true.", 0.2),
    "circuits": ("2-layer attention-only model on a real corpus (openwebtext subset) as in the paper; induction-head scores.", 2.0),
    "rome": ("GPT-2 XL causal tracing on 1000 CounterFact facts and ROME edits with the paper's efficacy/paraphrase/specificity metrics.", 3.0),
}


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (OUT / "batch.log").open("a") as f:
        f.write(line + "\n")


def spec_stage(slug: str) -> str:
    p = PAPERS[slug]
    d = OUT / slug
    d.mkdir(parents=True, exist_ok=True)
    if (d / "spec.yaml").exists():
        return f"{slug}: spec exists"
    try:
        if "arxiv" in p:
            src = fetch_arxiv(p["arxiv"], d / "paper")
        else:
            src = fetch_html_as_text(p["url"], d / "paper", slug)
        orch = Orchestrator(d, batch=True)
        hint, hours = (GPU_HINTS[slug][0], GPU_HINTS[slug][1]) if GPU and slug in GPU_HINTS else (p["hint"], None)
        spec = orch.stage_spec(src, "cs", p["family"], hint=hint, reported_gpu_hours=hours)
        if not GPU:
            spec.plan.compute_tier = 2
        spec.plan.seeds = [0, 1, 2] if GPU else p["seeds"]
        spec.plan.budget.run_timeout_s = int(hours * 3600 * 1.5) if (GPU and hours) else p["timeout"]
        spec.plan.budget.build = 40
        spec.plan.success_criteria = hint + " || " + spec.plan.success_criteria
        spec.plan.frozen_hash = None
        spec.freeze()
        spec.save(d / "spec.yaml")
        cost = orch.llm.cost_usd if orch.llm else 0
        return f"{slug}: spec ok, {len(spec.claims)} claims, {len(spec.ambiguities)} ambiguities, ${cost:.2f}"
    except Exception as e:  # noqa: BLE001
        (d / "spec_error.txt").write_text(traceback.format_exc())
        return f"{slug}: spec FAILED {type(e).__name__}: {str(e)[:200]}"


def replicate(slug: str) -> dict:
    p = PAPERS[slug]
    d = OUT / slug
    if (d / "REPORT.md").exists():
        rep = json.loads((d / "report.json").read_text())
        if not rep.get("failure"):
            return {"slug": slug, "skipped": True, **_row(rep)}
    if not (d / "spec.yaml").exists():
        return {"slug": slug, "grade": "-", "note": "no spec"}
    orch = Orchestrator(d, batch=True)
    spec = Spec.load(d / "spec.yaml")
    spec.plan.budget.build = max(spec.plan.budget.build, 40)
    t0 = time.time()
    rep = orch.replicate(spec, do_grid=p["grid"])
    row = _row(json.loads(rep.model_dump_json()))
    row.update(slug=slug, minutes=round((time.time() - t0) / 60, 1), cost=round(orch.llm.cost_usd if orch.llm else 0, 2))
    return row


def _row(rep: dict) -> dict:
    heads = [c for c in rep.get("claims", []) if c["outcome"] != "Untested"][:3]
    leak = {l["test"]: l["passed"] for l in rep.get("leakage", [])}
    return {"grade": (rep.get("grade") or {}).get("letter", "-"),
            "kind": rep.get("kind_of_test"), "tiers": f"{rep.get('data_tier')}/{rep.get('compute_tier')}",
            "claims": "; ".join(f"{c['claim_id']}={c['outcome']}" for c in heads) or "all Untested",
            "leakage": ", ".join(f"{k}:{'P' if v else ('F' if v is False else '-')}" for k, v in leak.items()),
            "failure": (rep.get("failure") or "")[:80]}


def write_summary(rows: list[dict]) -> None:
    lines = ["# Batch replication summary", "", f"updated {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "| paper | grade | kind | tiers | claims | leakage | min | $ | failure |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['slug']} | {r.get('grade','-')} | {r.get('kind','-')} | {r.get('tiers','-')} | {r.get('claims','-')} | "
                     f"{r.get('leakage','-')} | {r.get('minutes','-')} | {r.get('cost','-')} | {r.get('failure','')} |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def reverify(slug: str) -> dict:
    """Re-run run+verify+report on an existing work dir (no model calls unless the smoke gate
    now fails), e.g. after a verify-stage fix. Deletes the old report first."""
    d = OUT / slug
    for f in ("REPORT.md", "report.json"):
        (d / f).unlink(missing_ok=True)
    for f in (d / "runs").glob("*.json") if (d / "runs").exists() else []:
        f.unlink()
    p = PAPERS[slug]
    orch = Orchestrator(d, batch=True)
    spec = Spec.load(d / "spec.yaml")
    spec.plan.budget.build = max(spec.plan.budget.build, 40)
    t0 = time.time()
    rep = orch.replicate(spec, prebuilt=d / "work", do_grid=p["grid"])
    row = _row(json.loads(rep.model_dump_json()))
    row.update(slug=slug, minutes=round((time.time() - t0) / 60, 1), cost=round(orch.llm.cost_usd if orch.llm else 0, 2))
    return row


def _replicate_in_subprocess(slug: str) -> dict:
    """Fresh interpreter per paper: fixes to the package apply at the next paper, no restart."""
    import subprocess
    out = subprocess.run([sys.executable, "-c",
                          f"import json; from evals.batch import replicate; print('ROW=' + json.dumps(replicate({slug!r}), default=str))"],
                         cwd=ROOT, capture_output=True, text=True)
    for line in out.stdout.splitlines()[::-1]:
        if line.startswith("ROW="):
            return json.loads(line[4:])
    return {"slug": slug, "grade": "-", "failure": (out.stderr[-300:] or "subprocess produced no row").replace("\n", " ")}


def main(slugs: list[str]) -> None:
    global GPU, OUT
    if slugs and slugs[0] == "--matched":
        # paper-scale plan (GPU_HINTS) for the named papers on this machine, compute tier 1, long timeouts
        GPU = True
        OUT = ROOT / "runs" / "batch_matched"
        slugs = slugs[1:]
    if slugs and slugs[0] == "--reverify":
        rows = []
        targets = slugs[1:]
        if targets == ["auto"]:
            # every paper that has a spec and either no report, a recorded failure, or a failed leakage test
            targets = []
            for slug in PAPERS:
                d = OUT / slug
                if not (d / "spec.yaml").exists():
                    continue
                rp = d / "report.json"
                if not rp.exists():
                    targets.append(slug); continue
                r = json.loads(rp.read_text())
                if r.get("failure") or any(l["passed"] is False for l in r.get("leakage", [])):
                    targets.append(slug)
            log(f"reverify auto: {targets}")
        for slug in targets:
            log(f"reverify {slug} ...")
            row = reverify(slug)
            rows.append(row)
            log(f"{slug}: grade {row.get('grade')} claims [{row.get('claims')}] leakage [{row.get('leakage')}] {row.get('failure','')}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    slugs = slugs or list(PAPERS)
    log(f"batch start: {slugs}")
    with ThreadPoolExecutor(max_workers=3) as ex:
        for msg in ex.map(spec_stage, slugs):
            log(msg)
    rows: list[dict] = []
    for slug in slugs:
        log(f"replicate {slug} ...")
        row = _replicate_in_subprocess(slug)
        rows.append(row)
        log(f"{slug}: grade {row.get('grade')} claims [{row.get('claims')}] leakage [{row.get('leakage')}] {row.get('failure','')}")
        write_summary(rows)
    log("batch done")


if __name__ == "__main__":
    main(sys.argv[1:])
