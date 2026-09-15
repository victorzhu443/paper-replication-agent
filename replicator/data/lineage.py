"""Point-in-time lineage. Every feature column carries an `available_at` series computed by the
library (max over its inputs), not by model discipline (DESIGN.md stage 5)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PITFrame:
    """A DataFrame indexed by (date, id) or date, with per-column availability timestamps."""
    df: pd.DataFrame
    available_at: dict[str, pd.Series] = field(default_factory=dict)

    def add_feature(self, name: str, values: pd.Series, inputs: list[str], lag: pd.DateOffset | None = None) -> None:
        avail = None
        for col in inputs:
            a = self.available_at[col]
            avail = a if avail is None else pd.concat([avail, a], axis=1).max(axis=1)
        if lag is not None:
            avail = avail + lag
        self.df[name] = values
        self.available_at[name] = avail.reindex(self.df.index)

    def audit(self, feature: str, decision_time: pd.Series) -> tuple[bool, int]:
        """Assert available_at <= decision_time for every row. Returns (passed, n_violations)."""
        a = self.available_at[feature].reindex(decision_time.index)
        viol = int((a > decision_time).sum())
        return viol == 0, viol
