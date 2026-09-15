"""Spec-extraction eval against OSAP SignalDoc.csv (DESIGN.md §9.6): score a directory of
extracted specs, one per OSAP acronym, on the fields the row documents. Runs in seconds."""
from __future__ import annotations

from pathlib import Path

import yaml

from replicator.kb import signaldoc_rows
from replicator.schema import Spec

FIELDS = ["weighting", "quantile", "holding_months", "start_month", "sample_start_year", "sample_end_year", "sign", "return", "t_stat"]


def truth_for(row: dict) -> dict:
    t: dict = {}
    if row.get("Stock Weight"):
        t["weighting"] = row["Stock Weight"]
    if row.get("LS Quantile"):
        t["quantile"] = float(row["LS Quantile"])
    if row.get("Portfolio Period"):
        t["holding_months"] = int(float(row["Portfolio Period"]))
    if row.get("Start Month"):
        t["start_month"] = int(float(row["Start Month"]))
    if row.get("SampleStartYear"):
        t["sample_start_year"] = int(row["SampleStartYear"])
    if row.get("SampleEndYear"):
        t["sample_end_year"] = int(row["SampleEndYear"])
    if row.get("Sign"):
        t["sign"] = float(row["Sign"])
    if row.get("Return"):
        t["return"] = float(row["Return"])
    if row.get("T-Stat"):
        t["t_stat"] = float(row["T-Stat"])
    return t


def pred_for(spec: Spec) -> dict:
    p: dict = {}
    cfg = spec.default_config()
    head = next((c for c in spec.claims if c.priority == "headline"), None)
    if "weighting" in cfg:
        p["weighting"] = str(cfg["weighting"])
    if "quantile" in cfg:
        p["quantile"] = float(cfg["quantile"])
    if "holding_months" in cfg:
        p["holding_months"] = int(float(cfg["holding_months"]))
    if "start_month" in cfg:
        p["start_month"] = int(float(cfg["start_month"]))
    if head:
        if head.sample_start:
            p["sample_start_year"] = int(str(head.sample_start)[:4])
        if head.sample_end:
            p["sample_end_year"] = int(str(head.sample_end)[:4])
        p["sign"] = 1.0 if head.value > 0 else -1.0
        if head.metric in ("mean_return_pct", "mean_return"):
            p["return"] = head.value if head.units.scale == "percent" else head.value * 100
        if head.reported_t_stat:
            p["t_stat"] = head.reported_t_stat
        t = next((c for c in spec.claims if c.metric == "t_stat"), None)
        if t:
            p["t_stat"] = t.value
    return p


def score(truth: dict, pred: dict) -> dict:
    out = {}
    for f in FIELDS:
        if f not in truth:
            continue
        if f not in pred:
            out[f] = None
            continue
        a, b = truth[f], pred[f]
        if isinstance(a, str):
            out[f] = a.upper() == str(b).upper()
        elif f in ("return", "t_stat"):
            out[f] = abs(a - b) <= 0.02 * max(abs(a), 1e-9) + 0.01
        else:
            out[f] = abs(float(a) - float(b)) < 1e-9
    return out


def score_dir(pred_dir: Path, limit: int = 1000) -> dict:
    rows = {r["Acronym"]: r for r in signaldoc_rows()}
    per_field: dict[str, list] = {f: [] for f in FIELDS}
    n = 0
    for p in sorted(pred_dir.glob("*.yaml"))[:limit]:
        acr = p.stem
        if acr not in rows:
            continue
        spec = Spec.load(p)
        s = score(truth_for(rows[acr]), pred_for(spec))
        for f, v in s.items():
            per_field[f].append(v)
        n += 1
    summary = {f: {"n": len(v), "exact": sum(1 for x in v if x is True), "missing": sum(1 for x in v if x is None)} for f, v in per_field.items() if v}
    return {"papers": n, "fields": summary}
