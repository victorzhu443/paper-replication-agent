"""Content-addressed cache. Key = hash(source, params). Nothing is downloaded twice.
Per-user by construction (lives under the run root); licensed sources never cross users."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parents[2] / "data_cache"


def cache_key(source: str, params: dict) -> str:
    return hashlib.sha256(json.dumps({"source": source, "params": params}, sort_keys=True).encode()).hexdigest()[:20]


def cached_fetch(source: str, params: dict, fetch: Callable[[], pd.DataFrame], root: Path = CACHE_ROOT) -> tuple[pd.DataFrame, str]:
    root.mkdir(parents=True, exist_ok=True)
    key = cache_key(source, params)
    p = root / f"{source}__{key}.parquet"
    if p.exists():
        return pd.read_parquet(p), key
    df = fetch()
    df.to_parquet(p)
    (root / f"{source}__{key}.json").write_text(json.dumps({"source": source, "params": params, "rows": len(df),
                                                               "columns": list(df.columns)}, indent=2, default=str))
    return df, key
