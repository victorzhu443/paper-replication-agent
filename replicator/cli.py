"""CLI: replicate <spec.yaml|pdf> ... ; dry-run ; eval-spec."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from .orchestrator import Orchestrator
from .schema import Spec

app = typer.Typer(add_completion=False, help="Paper replication agent")


def _checkpoint(spec: Spec, notes: list[str]) -> bool:
    rprint("[bold]Checkpoint: spec + plan[/bold]")
    for c in spec.claims:
        if c.priority == "headline":
            rprint(f"  headline {c.id}: {c.metric}={c.value} ({c.where}, p.{c.evidence_page}) tol={spec.plan.tolerances.get(c.id)}")
    for a in sorted(spec.ambiguities, key=lambda a: {"high": 0, "medium": 1, "low": 2}[a.sensitivity]):
        rprint(f"  [{a.sensitivity}] {a.config_key}={a.default} ({a.source}) — {a.question}")
    for n in notes:
        rprint(f"  triage: {n}")
    return typer.confirm("Proceed?", default=True)


_ARXIV = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")


def _resolve_source(source: Path, out: Path) -> Path:
    """A path to a PDF, or an arXiv ID like 2106.04028 (downloaded into OUT/paper/)."""
    if source.exists():
        return source
    m = _ARXIV.match(str(source))
    if m:
        from .spec.intake import fetch_arxiv
        rprint(f"downloading arXiv:{source} ...")
        return fetch_arxiv(str(source), out / "paper")
    rprint(f"[red]no such file:[/red] {source}. Pass a PDF path, an arXiv ID (e.g. 2106.04028), or a spec.yaml.")
    raise typer.Exit(code=2)


def _preflight_credentials() -> None:
    """Fail in one line before any work if the key is missing or malformed."""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        rprint("[red]No Anthropic credentials.[/red] Set ANTHROPIC_API_KEY (Console -> API Keys -> Create Key, copy the sk-ant-api03-... secret).")
        raise typer.Exit(code=2)
    if key and not key.startswith("sk-ant-api"):
        rprint(f"[red]ANTHROPIC_API_KEY does not look like an API key[/red] (starts with {key[:12]!r}, length {len(key)}). "
               "The secret starts with sk-ant-api03- and is ~108 chars; the short apikey_... string is only the key's ID.")
        raise typer.Exit(code=2)


@app.command()
def replicate(source: Path, out: Path = Path("runs/latest"), track: str = "finance", family: str = "cross_sectional_anomaly",
              batch: bool = False, dry_run: bool = False, prebuilt: Optional[Path] = None, osap: Optional[str] = None,
              hint: str = "", no_grid: bool = False):
    """SOURCE is a spec.yaml (skips extraction) or a PDF (runs the extractor; needs Anthropic credentials)."""
    if not (source.suffix in (".yaml", ".yml") and prebuilt is not None):
        _preflight_credentials()  # every other path makes model calls
    orch = Orchestrator(out, batch=batch, human_approve=None if batch else _checkpoint)
    if source.suffix in (".yaml", ".yml"):
        spec = Spec.load(source)
        if not spec.plan.frozen_hash:
            spec = orch.stage_triage(spec)
    else:
        source = _resolve_source(source, out)
        spec = orch.stage_spec(source, track, family, hint=hint, osap_acronym=osap)
    if dry_run:
        rprint(f"[green]dry run complete[/green]: tier {spec.plan.data_tier}/{spec.plan.compute_tier}, "
               f"{len(spec.plan.target_claims)} target claims, spec at {out / 'spec.yaml'}")
        raise typer.Exit()
    rep = orch.replicate(spec, prebuilt=prebuilt, do_grid=not no_grid)
    rprint(f"[bold]grade {rep.grade.letter}[/bold] — {out / 'REPORT.md'}")


@app.command("eval-spec")
def eval_spec(pred_dir: Path, limit: int = 50):
    """Score extracted specs in PRED_DIR/<Acronym>.yaml against OSAP SignalDoc fields."""
    from evals.spec_eval import score_dir
    res = score_dir(pred_dir, limit=limit)
    rprint(json.dumps(res, indent=1))


if __name__ == "__main__":
    app()
