"""Hand replication #1 (finance, tier A, hermetic): 12-2 momentum decile spread on Kenneth
French's pre-formed prior-return portfolios. This is what the builder is expected to write for a
portfolio-level anomaly paper; kept as the reference implementation and the benchmark entry."""
from __future__ import annotations

import numpy as np
import pandas as pd

from template_anomaly import portfolio_pipeline
from replicator.data.adapters import french


def run(config: dict, seed: int = 0) -> dict:
    cfg = {"french_name": "10_mom", **config}
    # Overlapping K-month holding: inference uses the cohort-series SE by default (template
    # option overlap_inference=cohort), a fix found by the live builder's re-implementation on
    # 2026-09-15; the rolling-series t-stat is kept as t_stat_rolling.
    metrics, inter = portfolio_pipeline(cfg, seed)
    metrics["_intermediates"] = {"n_portfolios": inter["n_portfolios"], "first_date": inter["first_date"], "last_date": inter["last_date"]}
    metrics["_data_hashes"] = inter["data_hashes"]
    metrics["mean_return_pct"] = metrics["mean_return"] * 100
    return metrics


def raw_data() -> pd.DataFrame:
    df, _ = french.fetch({"name": "10_mom", "weighting": "VW"})
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """The 'signal' at the portfolio level is the prior long-short return, lagged one month
    (a portfolio-level analogue of a signal formed from past data only)."""
    df = raw.sort_values("date").copy()
    cols = [c for c in df.columns if c != "date"]
    df["ls_prior"] = (df[cols[-1]] - df[cols[0]]).shift(1)
    df["ls_prior_12"] = (df[cols[-1]] - df[cols[0]]).rolling(11).sum().shift(2)
    return df[["date", "ls_prior", "ls_prior_12"]]
