"""Regenerate the results tables in README.md and PROJECT_SUMMARY.md from report.json files, and
copy each batch paper's REPORT.md / report.json into papers/batch/<slug>/ so the repo carries the
evidence. Idempotent: rewrites only the text between the RESULTS markers.

    uv run python -m evals.update_docs
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "runs" / "batch"

TITLES = {
    "attention": ("Vaswani 2017 Transformer", "CS"), "resnet": ("He 2015 ResNet", "CS"),
    "layernorm": ("Ba 2016 LayerNorm", "CS"), "batchnorm": ("Ioffe 2015 BatchNorm", "CS"),
    "gan": ("Goodfellow 2014 GAN", "CS"), "dqn": ("Mnih 2013 DQN", "RL"), "ppo": ("Schulman 2017 PPO", "RL"),
    "worldmodels": ("Ha 2018 World Models", "RL"), "lottery": ("Frankle 2019 Lottery Ticket", "CS"),
    "lora": ("Hu 2021 LoRA", "CS"), "dpo": ("Rafailov 2023 DPO", "CS"),
    "superposition": ("Elhage 2022 Toy Models of Superposition", "CS"),
    "circuits": ("Elhage 2021 Transformer Circuits", "CS"), "rome": ("Meng 2022 ROME", "CS"),
}
# Hand-written one-liners on what the run showed; refined as reports come in.
NOTES = {
    "attention": "attention 99.4% vs no-attention 9.9% token accuracy on a copy task; WMT BLEU not comparable",
    "resnet": "build passed; full CIFAR runs exceeded the CPU time cap (fixed by the SCALE probe for later papers)",
    "gan": "MNIST GAN trains; shuffle test passes",
    "lottery": "winning tickets beat unpruned by 0.25–0.3 pts; early-stop speedup 2–3×",
    "batchnorm": "BN 96.5% vs no-BN 91.7% at 10k steps on both seeds (paper's direction)",
    "layernorm": "MNIST MLP baseline 98.4%; LN vs baseline comparison ran",
    "dqn": "CartPole DQN mean return 226, best episodes 500",
    "ppo": "CartPole 500/500 on 3 seeds; clipping beats no-clip on 2 of 3",
    "worldmodels": "VAE loss 3169→0.7; MDN-RNN and CMA-ES controller beat random policy (t=23)",
    "lora": "RoBERTa-base on SST-2 subset: LoRA 91.97% vs full FT 92.66% with 0.3M trainable params (paper's claim)",
    "dpo": "DPO loss from Eq. 7: 98.5% held-out preference accuracy, 100% win rate vs reference, reward margin grows",
    "superposition": "n=20, m=5: dense regime 5 features, sparse (S=0.99) 12.9 features in 5 dims; linear never superposes; antipodal pair 0.498 (paper 1/2)",
}


def _cost(d: Path) -> float:
    p = d / "llm" / "llm_calls.jsonl"
    return sum(json.loads(l).get("cost_usd", 0) for l in p.open()) if p.exists() else 0.0


def _metrics_note(d: Path) -> str:
    """Fallback note from the latest run record when no hand-written note exists."""
    runs = sorted((d / "runs").glob("*.json")) if (d / "runs").exists() else []
    for f in runs[::-1]:
        r = json.loads(f.read_text())
        if not r.get("error") and r.get("metrics"):
            items = [(k, v) for k, v in r["metrics"].items() if not k.endswith("_std") and not k.startswith("_")][:3]
            return "; ".join(f"{k}={v:.3g}" for k, v in items)
    return "no successful run"


def rows() -> list[str]:
    out = ["| Paper | Track | Kind of test | Tiers | Grade | Headline claims | Leakage | Cost |", "|---|---|---|---|---|---|---|---|",
           "| Jegadeesh–Titman 1993 momentum (OSAP Mom12m as ground truth) | finance | re-implementation, live builder | A/1 | **A** | 1.32 vs 1.31 %/mo Match; t-stat 4.76 vs 3.74 Match | shuffle P, future-perturbation P | $0.93 |",
           "| LeCun 1998 MLP-300 on MNIST, 3 seeds | CS | re-implementation | A/1 | C | 8.5% vs 4.7% error; convention grid 1.7–9.8% | shuffle P | $0 |"]
    total = 0.0
    for slug, (title, track) in TITLES.items():
        d = BATCH / slug
        cost = _cost(d)
        total += cost
        rp = d / "report.json"
        if not rp.exists():
            out.append(f"| {title} | {track} | — | — | in progress | spec extracted | — | ${cost:.2f} |")
            continue
        r = json.loads(rp.read_text())
        g = (r.get("grade") or {}).get("letter", "-")
        claims = [c for c in r.get("claims", []) if c["outcome"] != "Untested"][:2]
        if claims:
            cl = "; ".join(f"{c['claim_id']} {c['outcome']}" for c in claims)
        else:
            cl = NOTES.get(slug) or _metrics_note(d)
        if slug in NOTES and claims:
            cl = NOTES[slug] + " · " + cl
        leak = ", ".join(f"{l['test'].replace('_', '-')} {'P' if l['passed'] else ('F' if l['passed'] is False else 'n/a')}"
                         for l in r.get("leakage", []) if l["test"] != "contamination_scan") or "—"
        fail = f" · failed: {r['failure'][:60]}" if r.get("failure") else ""
        out.append(f"| {title} | {track} | {r.get('kind_of_test','-').replace('_',' ')} | {r.get('data_tier')}/{r.get('compute_tier')} | "
                   f"{'**A**' if g == 'A' else g} | {cl}{fail} | {leak} | ${cost:.2f} |")
    out.append("")
    out.append(f"Sweep total: ${total:.2f} in model calls for {len(TITLES)} papers. Reports: `papers/batch/<slug>/REPORT.md`. "
               f"Updated {time.strftime('%Y-%m-%d %H:%M')}.")
    return out


def splice(path: Path, block: str) -> None:
    s = path.read_text()
    if "<!-- RESULTS:BEGIN -->" not in s:
        raise SystemExit(f"{path} has no RESULTS markers")
    s = re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->",
               "<!-- RESULTS:BEGIN -->\n" + block + "\n<!-- RESULTS:END -->", s, flags=re.S)
    path.write_text(s)


def copy_reports() -> None:
    for slug in TITLES:
        d = BATCH / slug
        dest = ROOT / "papers" / "batch" / slug
        dest.mkdir(parents=True, exist_ok=True)
        for f in ("REPORT.md", "report.json"):
            if (d / f).exists():
                shutil.copy(d / f, dest / f)
        # the generated code is part of the evidence; copy the reproduce contract and scripts
        w = d / "work"
        if w.exists():
            (dest / "work").mkdir(exist_ok=True)
            for f in w.iterdir():
                if f.is_file() and f.suffix in (".py", ".sh", ".md", ".json") and f.name not in ("build_result.json", "setup.json") and f.stat().st_size < 400_000:
                    shutil.copy(f, dest / "work" / f.name)


def main() -> None:
    block = "\n".join(rows())
    splice(ROOT / "README.md", block)
    splice(ROOT / "PROJECT_SUMMARY.md", block)
    copy_reports()
    print(block)


if __name__ == "__main__":
    main()
