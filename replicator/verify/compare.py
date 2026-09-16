"""One Match rule, applied to frozen tolerances. Outcomes per DESIGN.md stage 8."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..schema import Claim, ClaimResult, Outcome, Spec
from .tolerance import derived_tolerance


def series_se(metric: str, series: np.ndarray) -> float | None:
    """Our own SE for metrics without an analytic rule from the paper's numbers."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return None
    if metric in ("mean_return", "alpha", "alpha_ff3", "alpha_ff5", "premium", "spread"):
        return float(x.std(ddof=1) / math.sqrt(len(x)))
    return None


def compare_claim(spec: Spec, c: Claim, ours: Optional[float], our_std: Optional[float] = None,
                  substitution_note: str = "", significant: Optional[bool] = None,
                  tolerance_override: Optional[float] = None) -> ClaimResult:
    """tolerance_override is the verify-time fill from our own SE, recorded in the report and
    the run log; it is never written into the frozen spec."""
    tol = spec.plan.tolerances.get(c.id)
    if tol is None and tolerance_override is not None:
        tol = tolerance_override
    if tol is not None:
        tol = max(tol, 0.005 * abs(c.value))  # numerical floor, see verify.tolerance
    if ours is None or (isinstance(ours, float) and math.isnan(ours)):
        return ClaimResult(claim_id=c.id, paper_value=c.value, outcome=Outcome.untested, tolerance=tol,
                           note="no value produced")
    if tol is None:
        return ClaimResult(claim_id=c.id, paper_value=c.value, our_value=ours, our_std=our_std,
                           outcome=Outcome.untested, note="no derived tolerance; needs n_periods, t-stat or std")
    gap = ours - c.value
    if c.relation != "eq":
        # directional claim: judged as an inequality against the threshold, with the tolerance as slack
        ok = {"gt": gap > 0, "ge": gap >= -tol, "lt": gap < 0, "le": gap <= tol}[c.relation]
        return ClaimResult(claim_id=c.id, paper_value=c.value, our_value=ours, our_std=our_std, tolerance=tol,
                           outcome=Outcome.match if ok else Outcome.mismatch,
                           note=f"directional claim ({c.relation} {c.value:.4g}): ours={ours:.4g}")
    if abs(gap) <= tol:
        return ClaimResult(claim_id=c.id, paper_value=c.value, our_value=ours, our_std=our_std, tolerance=tol,
                           outcome=Outcome.match, note=f"|gap|={abs(gap):.4g} <= tol {tol:.4g}")
    same_sign = (ours > 0) == (c.value > 0)
    if same_sign and spec.plan.data_tier == "B" and substitution_note and (significant is None or significant):
        return ClaimResult(claim_id=c.id, paper_value=c.value, our_value=ours, our_std=our_std, tolerance=tol,
                           outcome=Outcome.consistent,
                           note=f"|gap|={abs(gap):.4g} > tol {tol:.4g}; same sign; substitution: {substitution_note}")
    return ClaimResult(claim_id=c.id, paper_value=c.value, our_value=ours, our_std=our_std, tolerance=tol,
                       outcome=Outcome.mismatch, note=f"|gap|={abs(gap):.4g} > tol {tol:.4g}")


def freeze_tolerances(spec: Spec) -> None:
    """Called once in triage, before any run. Missing SEs are filled later only from our own series,
    and that fill is recorded in the run log, never edited into the frozen spec."""
    for c in spec.claims:
        t = derived_tolerance(c)
        if t is not None:
            spec.plan.tolerances[c.id] = float(t)
