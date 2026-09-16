"""Derived, not chosen (DESIGN.md §0.7). Match <=> |ours - paper| <= k*SE + reported_precision.

SE rules per metric on the paper's sample length T:
  t_stat            : 1
  sharpe (per-period): sqrt((1 + SR^2/2) / T)     (Lo 2002)
  mean_return / alpha: from reported t-stat if given (|value|/t), else needs our std/sqrt(T)
  r2 / oos_r2       : from bootstrap in verify; fallback 0.5 points
  accuracy/f1/top1  : paper's seed std if reported, else binomial sqrt(p(1-p)/n_test)
  bleu/perplexity   : paper's seed std if reported, else 1 point / 2% relative
"""
from __future__ import annotations

import math

from ..schema import Claim

K = 2.0


def standard_error(c: Claim) -> float | None:
    m = c.metric.lower().removesuffix("_pct").removesuffix("_percent")
    T = c.n_periods
    v = c.value
    if m in ("t_stat", "tstat", "t"):
        return 1.0
    if m == "sharpe":
        if not T:
            return None
        sr = v
        # convert annualized Sharpe to per-period for the SE formula if needed
        per = {"monthly": 12, "daily": 252, "weekly": 52, "quarterly": 4, "annual": 1}.get(c.units.period, 1)
        if c.units.annualized:
            sr = v / math.sqrt(per)
        se = math.sqrt((1 + sr * sr / 2) / T)
        return se * math.sqrt(per) if c.units.annualized else se
    if m in ("mean_return", "alpha", "alpha_ff3", "alpha_ff5", "premium", "spread"):
        if c.reported_t_stat:
            return abs(v) / abs(c.reported_t_stat)
        return None  # filled from our own series in verify.compare
    if m in ("accuracy", "f1", "top1", "top5", "auc", "error_rate", "test_error", "error"):
        if c.reported_std:
            return c.reported_std
        if T:  # T = number of test examples
            p = v / 100.0 if v > 1 else v
            se = math.sqrt(max(p * (1 - p), 1e-9) / T)
            return se * 100.0 if v > 1 else se
        return None
    if m in ("dimensions_per_feature", "feature_dimensionality", "ratio", "fraction"):
        return c.reported_std if c.reported_std is not None else (0.0 if c.reported_precision else None)
    if m in ("r2", "oos_r2"):
        return c.reported_std or 0.25  # points; refined by bootstrap when a series is available
    if m in ("bleu", "rouge", "meteor"):
        return c.reported_std or 0.5
    if m in ("perplexity", "loss"):
        return c.reported_std or 0.01 * abs(v)
    return c.reported_std


def derived_tolerance(c: Claim, fallback_se: float | None = None) -> float | None:
    se = standard_error(c)
    if se is None:
        se = fallback_se
    if se is None:
        return None
    # numerical floor: a printed value is never more precise than 0.5% of itself (float and
    # rounding noise); exact theoretical values (1/2, 3/4) still match to this slack
    return max(K * se + c.reported_precision, 0.005 * abs(c.value))
