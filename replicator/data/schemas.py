"""Schema contracts validated on every adapter load (pandera)."""
from __future__ import annotations

import pandera.pandas as pa

PortfolioReturns = pa.DataFrameSchema(
    {"date": pa.Column(pa.DateTime, nullable=False)},
    checks=[pa.Check(lambda df: df["date"].is_monotonic_increasing, error="dates monotone"),
            pa.Check(lambda df: not df["date"].duplicated().any(), error="no duplicate dates")],
    strict=False, coerce=True,
)

Factors = pa.DataFrameSchema(
    {"date": pa.Column(pa.DateTime), "mkt_rf": pa.Column(float), "smb": pa.Column(float),
     "hml": pa.Column(float), "rf": pa.Column(float)},
    checks=[pa.Check(lambda df: df["date"].is_monotonic_increasing, error="dates monotone")],
    strict=False, coerce=True,
)

StockPanel = pa.DataFrameSchema(
    {"date": pa.Column(pa.DateTime), "id": pa.Column(str), "ret": pa.Column(float, nullable=True),
     "prc": pa.Column(float, nullable=True, checks=pa.Check.ge(0)), "mktcap": pa.Column(float, nullable=True)},
    checks=[pa.Check(lambda df: not df.duplicated(["date", "id"]).any(), error="unique (date,id)")],
    strict=False, coerce=True,
)
