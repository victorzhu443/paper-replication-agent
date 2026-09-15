"""Produce specs for a directory of PDFs named <OSAP Acronym>.pdf, then score them against
SignalDoc.csv. Needs Anthropic credentials. Cost: one extractor + one referee call per paper.

    uv run python -m evals.run_spec_eval pdfs/ preds/ --limit 20
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from replicator.llm import LLM
from replicator.spec.extract import apply_kb_defaults, extract_spec, referee_spec
from .spec_eval import score_dir


def main(pdf_dir: Path, pred_dir: Path, limit: int = 20) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(pred_dir / "llm", max_cost_usd=100.0)
    for pdf in sorted(pdf_dir.glob("*.pdf"))[:limit]:
        out = pred_dir / f"{pdf.stem}.yaml"
        if out.exists():
            continue
        spec = extract_spec(llm, pdf, "finance", "cross_sectional_anomaly", paper_hint=f"OSAP acronym: {pdf.stem}")
        spec = referee_spec(llm, pdf, spec)
        spec = apply_kb_defaults(spec)  # no OSAP override here: the eval measures the extractor, not the KB
        spec.save(out)
        print(f"{pdf.stem}: {len(spec.claims)} claims, {len(spec.ambiguities)} ambiguities, ${llm.cost_usd:.2f} so far")
    print(json.dumps(score_dir(pred_dir), indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]), int(a[2]) if len(a) > 2 else 20)
