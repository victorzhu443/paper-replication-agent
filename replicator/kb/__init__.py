"""Conventions knowledge base (DESIGN.md §0.8). Seeded by hand from OSAP SignalDoc.csv and JKP.

Two families for the MVP. Each entry: ambiguity checklist with conventional default, source,
sensitivity; smoke config; hypothesis order; data checkpoint thresholds.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

KB_DIR = Path(__file__).parent

FAMILIES: dict[str, dict[str, Any]] = {
    "cross_sectional_anomaly": {
        "template": "replicator.templates.anomaly",
        # OSAP SignalDoc columns -> the six knobs of the anomaly template (§9.5) + two more
        "ambiguities": [
            {"config_key": "weighting", "question": "Equal- or value-weighted portfolios?",
             "options": ["EW", "VW"], "default": "EW",
             "source": "conventions_kb", "sensitivity": "high",
             "reason": "OSAP: 210 of 242 documented signals use EW in the original paper"},
            {"config_key": "quantile", "question": "Decile, quintile, or tercile sorts?",
             "options": ["0.1", "0.2", "0.333"], "default": "0.1",
             "source": "conventions_kb", "sensitivity": "medium",
             "reason": "OSAP LS Quantile column: deciles are the modal choice"},
            {"config_key": "holding_months", "question": "Holding period / rebalance frequency in months?",
             "options": ["1", "3", "6", "12"], "default": "1",
             "source": "conventions_kb", "sensitivity": "medium",
             "reason": "OSAP Portfolio Period: 1 (121 signals) and 12 (110 signals) dominate"},
            {"config_key": "start_month", "question": "Which month are annual signals updated?",
             "options": ["6", "7", "12"], "default": "6",
             "source": "conventions_kb", "sensitivity": "low",
             "reason": "Fama-French convention: accounting data lagged to June"},
            {"config_key": "price_filter", "question": "Minimum price filter?",
             "options": ["0", "1", "5"], "default": "0",
             "source": "conventions_kb", "sensitivity": "medium",
             "reason": "OSAP applies none by default; $5 filter is common in later papers"},
            {"config_key": "exchanges", "question": "Universe exchanges?",
             "options": ["NYSE", "NYSE_AMEX", "NYSE_AMEX_NASDAQ"], "default": "NYSE_AMEX_NASDAQ",
             "source": "conventions_kb", "sensitivity": "high",
             "reason": "CRSP share codes 10/11 on all three exchanges is the Fama-French universe"},
            {"config_key": "delisting", "question": "Delisting returns included?",
             "options": ["include", "exclude"], "default": "include",
             "source": "conventions_kb", "sensitivity": "high",
             "reason": "Shumway (1997): excluding them biases small-cap long-short spreads upward"},
            {"config_key": "return_def", "question": "Simple or log returns?",
             "options": ["simple", "log"], "default": "simple",
             "source": "conventions_kb", "sensitivity": "low",
             "reason": "Portfolio returns are simple by construction in this literature"},
        ],
        "smoke": {"years": 3},
        "hypothesis_order": ["delisting", "weighting", "exchanges", "holding_months", "quantile", "price_filter", "return_def"],
        "checkpoint": {"n_firms_rel_tol": 0.25, "mean_rel_tol": 0.5},
        "metrics": ["mean_return", "t_stat", "sharpe", "alpha_ff3"],
    },
    "ml_return_prediction": {
        "template": "replicator.templates.anomaly",
        "ambiguities": [
            {"config_key": "target_scaling", "question": "Are returns demeaned/standardized before fitting?",
             "options": ["none", "cross_sectional_rank", "zscore"], "default": "none",
             "source": "conventions_kb", "sensitivity": "high",
             "reason": "GKX 2020 use raw excess returns as target; ranks for features"},
            {"config_key": "rolling_window", "question": "Expanding or rolling training window?",
             "options": ["expanding", "rolling"], "default": "expanding",
             "source": "conventions_kb", "sensitivity": "high", "reason": "GKX 2020 expanding, refit yearly"},
            {"config_key": "weighting", "question": "Equal- or value-weighted portfolios?",
             "options": ["EW", "VW"], "default": "EW", "source": "conventions_kb", "sensitivity": "high",
             "reason": "GKX Table 7 reports both; EW headline"},
        ],
        "smoke": {"years": 3},
        "hypothesis_order": ["rolling_window", "target_scaling", "weighting"],
        "checkpoint": {"n_firms_rel_tol": 0.25, "mean_rel_tol": 0.5},
        "metrics": ["oos_r2", "sharpe", "mean_return", "t_stat"],
    },
    "checkpoint_eval": {
        "template": "replicator.templates.cs_eval",
        "ambiguities": [
            {"config_key": "split", "question": "Which split is the reported number on: validation or test?",
             "options": ["validation", "test"], "default": "test",
             "source": "conventions_kb", "sensitivity": "high",
             "reason": "GLUE-style papers report dev; leaderboard papers report test"},
            {"config_key": "averaging", "question": "Macro or micro averaging for F1/accuracy?",
             "options": ["macro", "micro"], "default": "macro", "source": "conventions_kb", "sensitivity": "medium",
             "reason": "sklearn default differs from many papers; must be stated"},
            {"config_key": "preprocessing", "question": "Eval-time preprocessing (crop/tokenizer/normalization)?",
             "options": ["as_repo", "as_paper"], "default": "as_repo", "source": "conventions_kb", "sensitivity": "high",
             "reason": "the released eval script is the closest thing to the paper's protocol"},
            {"config_key": "checkpoint_version", "question": "Which released checkpoint version?",
             "options": ["latest", "paper_date"], "default": "paper_date", "source": "conventions_kb",
             "sensitivity": "high", "reason": "checkpoints get silently re-uploaded"},
        ],
        "smoke": {"n_examples": 256},
        "hypothesis_order": ["split", "preprocessing", "checkpoint_version", "averaging"],
        "checkpoint": {"split_size_rel_tol": 0.02},
        "metrics": ["accuracy", "f1", "bleu", "perplexity", "top1"],
    },
    "train_and_eval": {
        "template": "replicator.templates.cs_eval",
        "ambiguities": [
            {"config_key": "lr_schedule", "question": "Learning-rate schedule?",
             "options": ["constant", "cosine", "step", "as_paper"], "default": "as_paper",
             "source": "paper_text", "sensitivity": "high", "reason": "the most common unstated training detail"},
            {"config_key": "augmentation", "question": "Train-time augmentation?",
             "options": ["none", "standard", "as_paper"], "default": "as_paper", "source": "paper_text",
             "sensitivity": "high", "reason": "often described by name only"},
            {"config_key": "batch_vs_steps", "question": "Fixed epochs or fixed steps when batch size changes?",
             "options": ["epochs", "steps"], "default": "steps", "source": "conventions_kb",
             "sensitivity": "medium", "reason": "reduced-scale runs must hold steps, not epochs"},
            {"config_key": "mixed_precision", "question": "Mixed precision at train time?",
             "options": ["fp32", "amp"], "default": "fp32", "source": "conventions_kb", "sensitivity": "low",
             "reason": "small effect on final metric; large on speed"},
            {"config_key": "split", "question": "Validation or test for the reported number?",
             "options": ["validation", "test"], "default": "test", "source": "conventions_kb", "sensitivity": "high",
             "reason": "as checkpoint_eval"},
        ],
        "smoke": {"epochs": 1, "n_examples": 2048},
        "hypothesis_order": ["split", "lr_schedule", "augmentation", "batch_vs_steps", "mixed_precision"],
        "checkpoint": {"split_size_rel_tol": 0.02},
        "metrics": ["accuracy", "f1", "loss", "top1"],
    },
}


def family_kb(family: str) -> dict[str, Any]:
    return FAMILIES[family]


def signaldoc_rows() -> list[dict[str, str]]:
    """OSAP SignalDoc.csv: 331 hand-extracted spec rows. Ground truth for the spec-extraction eval."""
    p = KB_DIR / "SignalDoc.csv"
    with p.open(encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def signaldoc_defaults(acronym: str) -> dict[str, Any] | None:
    """If the paper is one OSAP documents, return the documented conventions as ambiguity defaults."""
    for r in signaldoc_rows():
        if r["Acronym"].lower() == acronym.lower():
            out: dict[str, Any] = {}
            if r.get("Stock Weight"):
                out["weighting"] = r["Stock Weight"]
            if r.get("LS Quantile"):
                out["quantile"] = r["LS Quantile"]
            if r.get("Portfolio Period"):
                out["holding_months"] = str(int(float(r["Portfolio Period"])))
            if r.get("Start Month"):
                out["start_month"] = str(int(float(r["Start Month"])))
            out["_osap"] = {k: r[k] for k in ("Key Table in OP", "Test in OP", "Sign", "Return", "T-Stat",
                                                "SampleStartYear", "SampleEndYear", "Signal Rep Quality", "Filter")}
            return out
    return None
