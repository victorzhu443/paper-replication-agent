"""Stage-level tests for the CS verifier and the smoke-gate preconditions: every false-F mode
seen in the 14-paper sweep has a fixture here, and each must produce the intended verdict."""
import json
import shutil
from pathlib import Path

import pytest

from replicator.orchestrator import Orchestrator
from replicator.schema import Spec, Outcome

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "cs" / "make_fixture.py"


def _spec(metric: str) -> Spec:
    s = Spec.load(ROOT / "papers/mnist_mlp/spec.yaml")
    s.claims[0].metric = metric
    s.plan.budget.run_timeout_s = 60
    s.plan.seeds = [0]
    s.plan.compute_tier = 1
    return s


def _work(tmp_path: Path, mode: str) -> Orchestrator:
    orch = Orchestrator(tmp_path / mode, batch=True)
    orch.work.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIX, orch.work / "make_fixture.py")
    (orch.work / "reproduce.sh").write_text(f'#!/usr/bin/env bash\nexec "${{PYTHON:-python3}}" make_fixture.py {mode}\n')
    return orch


@pytest.mark.parametrize("mode,metric,expect_smoke,expect_shuffle", [
    ("clean", "accuracy", True, True),          # honors flag, reports chance -> judged, passes
    ("leaky", "accuracy", True, False),         # label leak -> judged, FAILS (a true F)
    ("ignores_flag", "accuracy", False, None),  # smoke gate rejects: flag not honored
    ("rl", "mean_return", True, True),          # random-policy null -> judged, passes
    ("na", "num_features_represented", True, None),  # declared not applicable -> A stays reachable
])
def test_cs_verifier_modes(tmp_path, mode, metric, expect_smoke, expect_shuffle):
    orch = _work(tmp_path, mode)
    spec = _spec(metric)
    orch.stage_triage(spec)
    spec.plan.compute_tier = 1
    sm = orch.smoke(spec, probe_scale=True)
    assert sm["passed"] is expect_smoke, sm["problems"]
    if not expect_smoke:
        assert any("shuffle" in p for p in sm["problems"])
        return
    recs = orch.stage_run(spec)
    from replicator.schema import Report
    rep = Report(paper_title="t", track="cs", family="train_and_eval", kind_of_test="re_implementation", data_tier="A", compute_tier=1)
    orch.stage_verify(spec, recs, rep, do_grid=False)
    sh = next(l for l in rep.leakage if l.test == "shuffle")
    assert sh.passed is expect_shuffle, sh.detail
    if mode == "na":
        from replicator.report import grade
        assert grade(rep, spec).integrity == "not_applicable"
