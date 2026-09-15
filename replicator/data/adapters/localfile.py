"""User-supplied data (a firm's CRSP extract, a lab's dataset). Validated against the same schemas."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schemas import StockPanel


def probe() -> tuple[bool, str]:
    return True, "local"


def fetch(params: dict) -> tuple[pd.DataFrame, str]:
    p = Path(params["path"])
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, parse_dates=["date"])
    if params.get("schema", "stock_panel") == "stock_panel":
        StockPanel.validate(df)
    return df, f"local:{p.name}:{p.stat().st_size}"
