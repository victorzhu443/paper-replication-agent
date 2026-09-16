"""On-disk contracts. Every stage reads and writes these; nothing lives only in a conversation.

Three artifacts (DESIGN.md §9.3): spec.yaml (claims + ambiguities + plan + frozen hash),
runs/<hash>.json (one per run), report.json / REPORT.md.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class Track(str, Enum):
    finance = "finance"
    cs = "cs"


class Family(str, Enum):
    # Finance
    cross_sectional_anomaly = "cross_sectional_anomaly"   # signal -> sort -> portfolios -> alpha
    ml_return_prediction = "ml_return_prediction"         # features -> rolling train -> OOS R2
    # CS
    checkpoint_eval = "checkpoint_eval"                   # released weights -> eval script -> metric
    train_and_eval = "train_and_eval"                     # train (possibly reduced) -> eval -> metric


class Units(BaseModel):
    period: Literal["daily", "weekly", "monthly", "quarterly", "annual", "none"] = "none"
    annualized: bool = False
    scale: Literal["decimal", "percent", "bps", "points", "count", "none"] = "none"


class Claim(BaseModel):
    id: str
    where: str = Field(description='e.g. "Table 3, row 2, column VW"')
    evidence_page: Optional[int] = Field(None, description="1-indexed page the value was read from")
    metric: str = Field(description="sharpe | mean_return | t_stat | alpha | r2 | accuracy | f1 | bleu | perplexity | ...")
    units: Units = Units()
    value: float
    reported_precision: float = Field(0.0, description="half the last printed digit, e.g. 0.005 for 0.41")
    n_periods: Optional[int] = Field(None, description="sample length driving the derived tolerance")
    reported_std: Optional[float] = Field(None, description="paper's own std across seeds/runs, if any")
    reported_t_stat: Optional[float] = None
    sample_start: Optional[str] = None
    sample_end: Optional[str] = None
    method_variant: str = Field("default", description="key into method.variants")
    priority: Literal["headline", "secondary", "descriptive"] = "secondary"
    relation: Literal["eq", "gt", "ge", "lt", "le"] = Field(
        "eq", description='"eq": the paper reports this number; "gt"/"lt": a directional claim ("A exceeds B", "reaches X in fewer steps"), value = threshold')


class Ambiguity(BaseModel):
    id: str
    question: str
    config_key: str = Field(description="the config key this flips, e.g. weighting")
    options: list[str]
    default: str
    source: Literal["paper_text", "author_code", "conventions_kb", "guess"] = "guess"
    reason: str = ""
    sensitivity: Literal["low", "medium", "high"] = "medium"


class DataSource(BaseModel):
    canonical: str = Field(description="crsp_monthly | compustat_annual | french_library | mnist | ...")
    description: str = ""
    substitute: Optional[str] = Field(None, description="adapter actually used, set by triage")


class DataSpec(BaseModel):
    sources: list[DataSource] = []
    universe: dict[str, Any] = {}
    frequency: str = "monthly"
    preprocessing: dict[str, Any] = {}
    splits: dict[str, Any] = {}


class Method(BaseModel):
    signal: str = ""
    model: dict[str, Any] = {}
    evaluation: list[str] = []
    eval_protocol: dict[str, Any] = Field({}, description="CS: metric def, averaging, split identity, decoding params, n_runs")
    variants: dict[str, dict[str, Any]] = Field(default_factory=lambda: {"default": {}})


class DescriptiveStat(BaseModel):
    metric: str
    value: float
    where: str = ""


class PaperMeta(BaseModel):
    title: str
    authors: list[str] = []
    year: Optional[int] = None
    arxiv_id: Optional[str] = None
    version: Optional[str] = None
    track: Track
    family: Family


class Budget(BaseModel):
    """Minutes per stage, set by code from family defaults; the model may only request changes."""
    spec: int = 10
    setup: int = 20
    build: int = 25
    run: int = 20
    verify: int = 15
    report: int = 5          # reserved, un-killable
    hard_kill_minutes: int = 240
    max_cost_usd: float = 150.0
    run_timeout_s: int = 1800     # per reproduce.sh invocation (CS) 


class Plan(BaseModel):
    data_tier: Literal["A", "B", "C"] = "A"
    compute_tier: Literal[1, 2] = 1
    kind_of_test: Literal["reproduction", "re_implementation", "conceptual_replication", "mechanics_only"] = "re_implementation"
    target_claims: list[str] = []
    substitutions: dict[str, str] = Field({}, description="canonical source -> adapter used")
    tolerances: dict[str, float] = Field({}, description="claim id -> absolute tolerance, derived, frozen")
    success_criteria: str = ""
    budget: Budget = Budget()
    seeds: list[int] = [0, 1, 2]
    frozen_hash: Optional[str] = Field(None, description="sha256 over claims+tolerances, set once")
    frozen_at: Optional[str] = None


class Spec(BaseModel):
    paper: PaperMeta
    claims: list[Claim]
    data: DataSpec = DataSpec()
    method: Method = Method()
    baselines: list[str] = []
    ambiguities: list[Ambiguity] = []
    descriptive_stats: list[DescriptiveStat] = []
    plan: Plan = Plan()

    # ----- persistence
    def save(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(json.loads(self.model_dump_json()), sort_keys=False, allow_unicode=True))

    @classmethod
    def load(cls, path: Path) -> "Spec":
        return cls.model_validate(yaml.safe_load(path.read_text()))

    # ----- freeze (principle 0.7)
    def frozen_payload(self) -> str:
        payload = {
            "claims": [c.model_dump() for c in self.claims],
            "tolerances": self.plan.tolerances,
            "target_claims": self.plan.target_claims,
            "success_criteria": self.plan.success_criteria,
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def freeze(self) -> str:
        h = hashlib.sha256(self.frozen_payload().encode()).hexdigest()
        if self.plan.frozen_hash and self.plan.frozen_hash != h:
            raise FrozenSpecViolation(
                f"spec.claims/tolerances changed after freeze: {self.plan.frozen_hash[:12]} -> {h[:12]}"
            )
        self.plan.frozen_hash = h
        self.plan.frozen_at = self.plan.frozen_at or datetime.now(timezone.utc).isoformat()
        return h

    def assert_frozen(self) -> None:
        if not self.plan.frozen_hash:
            raise FrozenSpecViolation("spec is not frozen; triage must run before build")
        h = hashlib.sha256(self.frozen_payload().encode()).hexdigest()
        if h != self.plan.frozen_hash:
            raise FrozenSpecViolation("spec.claims or tolerances were edited after freeze")

    def default_config(self) -> dict[str, Any]:
        cfg = dict(self.method.variants.get("default", {}))
        for a in self.ambiguities:
            cfg.setdefault(a.config_key, a.default)
        return cfg

    def config_for_variant(self, variant: str) -> dict[str, Any]:
        cfg = self.default_config()
        cfg.update(self.method.variants.get(variant, {}))
        return cfg


class FrozenSpecViolation(RuntimeError):
    pass


# ----------------------------------------------------------------- runs & report

class NodeRecord(BaseModel):
    name: str
    input_hash: str
    wall_seconds: float
    cached: bool = False


class RunRecord(BaseModel):
    run_id: str
    purpose: Literal["smoke", "full", "seed", "grid", "shuffle", "future_perturbation", "reproduce"] = "full"
    config: dict[str, Any]
    seed: Optional[int] = None
    data_hashes: dict[str, str] = {}
    env_lock_hash: Optional[str] = None
    started_at: str
    wall_seconds: float = 0.0
    nodes: list[NodeRecord] = []
    metrics: dict[str, float] = {}
    intermediates: dict[str, Any] = {}
    error: Optional[str] = None

    def save(self, runs_dir: Path) -> Path:
        runs_dir.mkdir(parents=True, exist_ok=True)
        p = runs_dir / f"{self.run_id}.json"
        p.write_text(self.model_dump_json(indent=2))
        return p


class Outcome(str, Enum):
    match = "Match"
    consistent = "Consistent"
    mismatch = "Mismatch"
    untested = "Untested"


class ClaimResult(BaseModel):
    claim_id: str
    paper_value: float
    our_value: Optional[float] = None
    our_std: Optional[float] = None
    tolerance: Optional[float] = None
    outcome: Outcome
    note: str = ""


class LeakageResult(BaseModel):
    test: Literal["shuffle", "future_perturbation", "available_at_audit", "contamination_scan"]
    passed: Optional[bool]        # None = not run
    detail: str = ""


class GridPoint(BaseModel):
    config_key: str
    value: Any
    headline_value: Optional[float]
    delta_from_default: Optional[float]


class GradeVector(BaseModel):
    data_fidelity: str
    procedure_fidelity: str
    result: str
    integrity: str
    letter: str


class Report(BaseModel):
    paper_title: str
    track: str
    family: str
    kind_of_test: str
    data_tier: str
    compute_tier: int
    deviations: list[str] = []
    claims: list[ClaimResult] = []
    leakage: list[LeakageResult] = []
    grid: list[GridPoint] = []
    grid_range: Optional[tuple[float, float]] = None
    unexplained: list[str] = []
    ambiguities_used: list[dict[str, Any]] = []
    grade: Optional[GradeVector] = None
    reproducibility: dict[str, Any] = {}
    human_minutes: float = 0.0
    wall_minutes: float = 0.0
    cost_usd: float = 0.0
    stages_completed: list[str] = []
    failure: Optional[str] = None


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
