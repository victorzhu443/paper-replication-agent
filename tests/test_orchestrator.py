"""End-to-end on the hermetic momentum example (French data is cached after the first fetch),
plus the builder loop driven by a scripted fake model: the orchestrator, not the model, ends the stage."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from replicator.build.builder import Builder
from replicator.orchestrator import Orchestrator
from replicator.schema import Spec, Outcome

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def momentum_spec():
    return Spec.load(ROOT / "papers/momentum_french/spec.yaml")


def test_momentum_end_to_end(tmp_path, momentum_spec):
    orch = Orchestrator(tmp_path / "run", batch=True)
    spec = orch.stage_triage(momentum_spec)
    assert spec.plan.frozen_hash
    rep = orch.replicate(spec, prebuilt=ROOT / "papers/momentum_french")
    assert rep.failure is None
    head = next(c for c in rep.claims if c.claim_id == "T1A.K3.J12.ret")
    assert head.outcome == Outcome.match
    assert abs(head.our_value - 1.31) < 0.2
    tests = {l.test: l.passed for l in rep.leakage}
    assert tests["shuffle"] is True and tests["future_perturbation"] is True
    assert rep.grid_range and rep.grid_range[0] <= 1.31 <= rep.grid_range[1] + 0.5
    assert (tmp_path / "run/REPORT.md").exists() and (tmp_path / "run/report.json").exists()
    assert rep.grade.letter in ("A", "B")
    # the t-stat claim uses Newey-West for overlapping holding periods and should now match
    tstat = next(c for c in rep.claims if c.claim_id == "T1A.K3.J12.t")
    assert tstat.outcome == Outcome.match, tstat


def test_report_always_produced_on_failure(tmp_path, momentum_spec):
    orch = Orchestrator(tmp_path / "run", batch=True)
    spec = orch.stage_triage(momentum_spec)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "pipeline.py").write_text("def run(config, seed=0):\n    raise ValueError('boom')\n")
    rep = orch.replicate(spec, prebuilt=broken)
    assert rep.failure and "smoke" in rep.failure
    assert rep.grade.letter == "F"
    assert (tmp_path / "run/REPORT.md").exists()


class _Block(SimpleNamespace):
    pass


class FakeLLM:
    """Scripted model: turn 1 writes a broken pipeline and claims done; turn 2 (forced by the
    orchestrator) writes a working one and runs the smoke test."""
    cost_usd = 0.0

    def __init__(self, good_source: str):
        self.turn = 0
        self.good = good_source

    def tool_turn(self, stage, system, messages, tools, **kw):
        self.turn += 1
        if self.turn == 1:
            content = [_Block(type="tool_use", id="t1", name="write_file",
                              input={"path": "pipeline.py", "content": "def run(config, seed=0):\n    return {}\n"}),
                       _Block(type="tool_use", id="t2", name="run_smoke", input={})]
        elif self.turn == 2:
            content = [_Block(type="text", text="I am done.")]  # false claim of completion
        else:
            content = [_Block(type="tool_use", id="t3", name="write_file", input={"path": "pipeline.py", "content": self.good}),
                       _Block(type="tool_use", id="t4", name="run_smoke", input={})]
        return SimpleNamespace(content=content, stop_reason="tool_use")


def test_builder_loop_orchestrator_owns_stop(tmp_path, momentum_spec):
    orch = Orchestrator(tmp_path / "run", batch=True)
    spec = orch.stage_triage(momentum_spec)
    orch.stage_setup(spec)
    good = (ROOT / "papers/momentum_french/pipeline.py").read_text()
    fake = FakeLLM(good)
    b = Builder(fake, spec, orch.work, lambda: orch.smoke(spec), budget_minutes=5)
    res = b.run()
    assert res["smoke_passed"] is True
    assert fake.turn == 3  # the "I am done" turn did not end the stage
    assert res["blacklist_hits"] == []
