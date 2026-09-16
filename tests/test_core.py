import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replicator.schema import Claim, Spec, FrozenSpecViolation, Units, Ambiguity, Outcome
from replicator.verify import tolerance, compare, leakage, grid
from replicator.templates import anomaly
from replicator.data.lineage import PITFrame
from evals.spec_eval import truth_for, pred_for, score
from replicator.kb import signaldoc_rows

ROOT = Path(__file__).resolve().parents[1]


def test_sharpe_se_and_tolerance():
    c = Claim(id="x", where="T", metric="sharpe", units=Units(period="monthly"), value=0.41, reported_precision=0.005, n_periods=678)
    se = tolerance.standard_error(c)
    assert abs(se - math.sqrt((1 + 0.41**2 / 2) / 678)) < 1e-12
    assert abs(tolerance.derived_tolerance(c) - (2 * se + 0.005)) < 1e-12
    # a 10-year sample gets a wider tolerance, not the same ±0.1
    c10 = c.model_copy(update={"n_periods": 120})
    assert tolerance.derived_tolerance(c10) > tolerance.derived_tolerance(c)


def test_freeze_blocks_tolerance_edits():
    spec = Spec.load(ROOT / "papers/momentum_french/spec.yaml")
    compare.freeze_tolerances(spec)
    spec.freeze()
    spec.assert_frozen()
    spec.plan.tolerances["T1A.K3.J12.ret"] = 99.0
    with pytest.raises(FrozenSpecViolation):
        spec.assert_frozen()


def test_match_rule_outcomes():
    spec = Spec.load(ROOT / "papers/momentum_french/spec.yaml")
    compare.freeze_tolerances(spec)
    c = spec.claims[0]
    tol = spec.plan.tolerances[c.id]
    assert abs(tol - (2 * 1.31 / 3.74 + 0.005)) < 1e-9
    assert compare.compare_claim(spec, c, 1.31 + tol * 0.9).outcome == Outcome.match
    assert compare.compare_claim(spec, c, 1.31 + tol * 1.1).outcome == Outcome.mismatch
    spec.plan.data_tier = "B"
    r = compare.compare_claim(spec, c, 0.5, substitution_note="no delisting", significant=True)
    assert r.outcome == Outcome.consistent
    assert compare.compare_claim(spec, c, None).outcome == Outcome.untested


def _synthetic_panel(n_ids=200, n_months=120, leak=False, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    rows = []
    for i in range(n_ids):
        r = rng.normal(0.01, 0.08, n_months)
        for t, d in enumerate(dates):
            rows.append((d, f"s{i}", r[t], 10.0 + rng.normal(), rng.lognormal(5, 1)))
    df = pd.DataFrame(rows, columns=["date", "id", "ret", "prc", "mktcap"])
    return df


def _features_clean(raw):
    return anomaly.momentum_signal(raw)[["date", "id", "signal"]]


def _features_leaky(raw):
    p = raw.sort_values(["id", "date"]).copy()
    p["signal"] = p.groupby("id")["ret"].shift(-1)  # uses next month's return: look-ahead
    return p[["date", "id", "signal"]]


def test_future_perturbation_detects_lookahead():
    raw = _synthetic_panel()
    assert leakage.future_perturbation_test(_features_clean, raw).passed is True
    assert leakage.future_perturbation_test(_features_leaky, raw).passed is False


def test_shuffle_test_on_leaky_vs_clean_pipeline():
    raw = _synthetic_panel(n_ids=300, n_months=120, seed=1)

    def run_clean(cfg, seed):
        panel = anomaly.momentum_signal(raw)
        return anomaly.stock_pipeline(panel, None, cfg, seed)

    def run_leaky(cfg, seed):
        p = raw.sort_values(["id", "date"]).copy()
        p["ret_fwd"] = p.groupby("id")["ret"].shift(-1)
        p["signal"] = p["ret_fwd"] + np.random.default_rng(seed).normal(0, 0.01, len(p))  # signal IS the label
        if cfg.get("_shuffle_labels"):
            # a pipeline that leaks through the label channel keeps the link even after 'shuffling'
            pass
        m = anomaly.stock_pipeline(p, None, {**cfg, "_shuffle_labels": False}, seed)
        m["_shuffled"] = bool(cfg.get("_shuffle_labels"))  # acknowledges the flag but leaks through the label
        return m

    base_clean = run_clean({"weighting": "EW", "quantile": 0.1}, 0)
    r = leakage.shuffle_test(run_clean, {"weighting": "EW", "quantile": 0.1}, "mean_return", base_clean["mean_return"])
    assert r.passed is True
    base_leaky = run_leaky({"weighting": "EW", "quantile": 0.1}, 0)
    r2 = leakage.shuffle_test(run_leaky, {"weighting": "EW", "quantile": 0.1}, "mean_return", base_leaky["mean_return"])
    assert r2.passed is False


def test_pit_lineage_audit():
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    df = pd.DataFrame({"x": np.arange(5.0)}, index=idx)
    pit = PITFrame(df, {"x": pd.Series(idx, index=idx)})
    pit.add_feature("x_lag6", df["x"], ["x"], lag=pd.DateOffset(months=6))
    ok, n = pit.audit("x_lag6", pd.Series(idx, index=idx))
    assert ok is False and n == 5
    ok2, _ = pit.audit("x", pd.Series(idx, index=idx))
    assert ok2


def test_grid_reports_range():
    def run_fn(cfg, seed):
        return {"m": {"EW": 1.0, "VW": 0.6}[cfg["weighting"]] * (1.0 if cfg.get("k", "1") == "1" else 0.8)}
    ambs = [Ambiguity(id="a", question="w", config_key="weighting", options=["EW", "VW"], default="EW", sensitivity="high"),
            Ambiguity(id="b", question="k", config_key="k", options=["1", "3"], default="1", sensitivity="medium")]
    pts, rng = grid.run_grid(run_fn, {"weighting": "EW", "k": "1"}, ambs, "m", 1.0)
    assert len(pts) == 2 and rng == (0.6, 1.0)


def test_spec_eval_scores_hand_spec_against_signaldoc():
    row = next(r for r in signaldoc_rows() if r["Acronym"] == "Mom12m")
    spec = Spec.load(ROOT / "papers/momentum_french/spec.yaml")
    spec.method.variants["default"] = spec.method.variants["ew_decile_k3"]
    s = score(truth_for(row), pred_for(spec))
    assert s["weighting"] and s["holding_months"] and s["return"] and s["t_stat"] and s["sign"]


def test_shuffle_skill_rule_handles_error_metrics():
    from replicator.verify.leakage import metric_skill, shuffle_test
    assert metric_skill("test_error", 8.5) == 91.5 and metric_skill("accuracy", 91.5) == 91.5
    # clean classifier: shuffled labels -> near-chance error -> passes
    r = shuffle_test(lambda cfg, s: {"test_error": 85.7, "_shuffled": True}, {}, "test_error", 8.5)
    assert r.passed is True
    # leaky classifier: shuffled labels -> still low error -> fails
    r2 = shuffle_test(lambda cfg, s: {"test_error": 9.0, "_shuffled": True}, {}, "test_error", 8.5)
    assert r2.passed is False


def test_shuffle_requires_acknowledgement():
    from replicator.verify.leakage import shuffle_test
    # a script that ignores the flag returns the real number without _shuffled: not a leak verdict
    r = shuffle_test(lambda cfg, s: {"accuracy": 91.5}, {}, "accuracy", 91.5)
    assert r.passed is None and "acknowledge" in r.detail
    r2 = shuffle_test(lambda cfg, s: {"accuracy": 91.5, "_shuffled": True}, {}, "accuracy", 91.5)
    assert r2.passed is False
