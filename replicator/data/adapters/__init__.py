"""Typed data adapters with a common interface: fetch(params) -> (DataFrame, cache_key).

Credentialed adapters (WRDS, Polygon) run in the harness process; the sandbox only ever sees
the parquet cache. The model never sees rows: `describe()` returns aggregates for prompts.
"""
from __future__ import annotations

import pandas as pd

from . import french, hf_datasets, localfile, torchvision_ds

ADAPTERS = {
    "french_library": french,
    "local_file": localfile,
    "torchvision": torchvision_ds,
    "hf_datasets": hf_datasets,
}

# canonical source -> ordered substitutes, with the registry's known consequence
SUBSTITUTIONS = {
    "crsp_monthly": [
        ("local_file", "user-supplied CRSP extract: no known consequence"),
        ("french_library", "pre-formed portfolios only: cannot re-sort; universe = CRSP NYSE/AMEX/NASDAQ; delisting included"),
        ("tiingo", "no delisting returns; survivorship-adjusted history; spreads biased upward in small caps (Shumway 1997)"),
    ],
    "compustat_annual": [("local_file", "user-supplied extract: no known consequence")],
    "french_library": [("french_library", "exact source")],
    # open CS datasets: exact sources
    "mnist": [("torchvision", "exact source")],
    "fashion_mnist": [("torchvision", "exact source")],
    "cifar10": [("torchvision", "exact source")],
    "cifar100": [("torchvision", "exact source")],
}


def probe(source: str) -> tuple[bool, str]:
    """A 200 OK is a fact; 'yfinance has this' is a belief (DESIGN.md stage 3)."""
    mod = ADAPTERS.get(source)
    if mod is None:
        return False, f"no adapter for {source}"
    return mod.probe()


def options_for(canonical: str) -> list[tuple[str, str]]:
    """Substitution options for a canonical source, including Hub datasets resolved by name."""
    if canonical in SUBSTITUTIONS:
        return SUBSTITUTIONS[canonical]
    repo = hf_datasets.resolve(canonical)
    if repo:
        return [(f"hf_datasets:{repo}", "exact source")]
    return [(canonical, "direct")]


def describe(df: pd.DataFrame) -> dict:
    """Aggregates only: the model never receives rows."""
    num = df.select_dtypes("number")
    out = {"rows": int(len(df)), "columns": list(df.columns)}
    if "date" in df.columns:
        out["date_min"] = str(pd.Timestamp(df["date"].min()).date())
        out["date_max"] = str(pd.Timestamp(df["date"].max()).date())
    out["means"] = {c: float(num[c].mean()) for c in num.columns[:20]}
    out["stds"] = {c: float(num[c].std()) for c in num.columns[:20]}
    return out
