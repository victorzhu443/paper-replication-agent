"""Descriptive-stats checkpoint against the paper's Table 1. Gate in Tier A; measurement in Tier B."""
from __future__ import annotations

from dataclasses import dataclass

from ..schema import DescriptiveStat


@dataclass
class CheckpointResult:
    passed: bool
    rows: list[dict]
    mode: str  # gate | measure


def compare(paper_stats: list[DescriptiveStat], ours: dict[str, float], data_tier: str,
            rel_tol: float = 0.25) -> CheckpointResult:
    rows, ok = [], True
    for s in paper_stats:
        v = ours.get(s.metric)
        if v is None:
            rows.append({"metric": s.metric, "paper": s.value, "ours": None, "rel_gap": None, "within": None})
            continue
        gap = (v - s.value) / abs(s.value) if s.value else float("inf")
        within = abs(gap) <= rel_tol
        ok &= within
        rows.append({"metric": s.metric, "paper": s.value, "ours": v, "rel_gap": round(gap, 3), "within": within})
    mode = "gate" if data_tier == "A" else "measure"
    return CheckpointResult(passed=ok if mode == "gate" else True, rows=rows, mode=mode)
