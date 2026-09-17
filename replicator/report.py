"""Stage 5: report.json + REPORT.md. Grade is a vector first, a letter second."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import GradeVector, Outcome, Report, Spec


def grade(report: Report, spec: Spec) -> GradeVector:
    heads = [c for c in report.claims if any(s.id == c.claim_id and s.priority == "headline" for s in spec.claims)]
    outcomes = [c.outcome for c in heads] or [c.outcome for c in report.claims]
    ran = [l for l in report.leakage if l.passed is not None]
    failed = [l for l in ran if l.passed is False]
    na = any(l.test == "shuffle" and l.passed is None and l.detail.startswith("not applicable") for l in report.leakage)
    integrity = "failed" if failed else ("passed" if ran else ("not_applicable" if na else "not_run"))
    data_f = {"A": "tier A checkpoint", "B": "tier B gap measured", "C": "synthetic"}[report.data_tier]
    n_unexpl = len(report.unexplained)
    proc = "author code" if report.kind_of_test == "reproduction" else f"re-implemented, {n_unexpl} unexplained"
    if not outcomes:
        result = "no claims"
    elif all(o == Outcome.match for o in outcomes):
        result = "all headline Match"
    elif all(o in (Outcome.match, Outcome.consistent) for o in outcomes):
        result = "Consistent"
    elif any(o == Outcome.mismatch for o in outcomes):
        result = "Mismatch"
    else:
        result = "Untested"
    infra = report.failure and any(k in report.failure for k in ("RemoteProtocolError", "APIConnectionError", "APITimeoutError",
                                                                     "transport failed", "CostCapExceeded", "ReadTimeout"))
    compute = report.failure and any(k in report.failure for k in ("SCALE=0.1 probe", "would take ~", "does not fit"))
    if infra or compute:
        letter = "N"  # not graded: the harness/network failed, or the paper's configuration does not fit this machine
        if compute:
            proc = proc + "; paper-scale run does not fit the compute envelope"
    elif report.failure or integrity == "failed":
        letter = "F"
    elif result == "all headline Match" and integrity in ("passed", "not_applicable") and report.data_tier == "A":
        letter = "A"
    elif result in ("all headline Match", "Consistent") and integrity in ("passed", "not_applicable"):
        letter = "B"
    elif result in ("Untested", "no claims") or report.kind_of_test == "mechanics_only":
        letter = "C"
    else:
        letter = "C"  # mechanics verified and integrity passed, numbers do not match
    return GradeVector(data_fidelity=data_f, procedure_fidelity=proc, result=result, integrity=integrity, letter=letter)


def write(report: Report, spec: Spec, out_dir: Path) -> Path:
    report.grade = grade(report, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(report.model_dump_json(indent=2))
    md = [f"# Replication report: {report.paper_title}", ""]
    md += [f"**Kind of test:** {report.kind_of_test} · **Data tier:** {report.data_tier} · **Compute tier:** {report.compute_tier} · "
           f"**Track/family:** {report.track}/{report.family}", ""]
    g = report.grade
    md += [f"**Grade:** {'not graded (infrastructure failure)' if g.letter == 'N' else g.letter}  (data: {g.data_fidelity}; procedure: {g.procedure_fidelity}; result: {g.result}; integrity: {g.integrity})", ""]
    if report.failure:
        md += [f"> **Run did not complete:** {report.failure}", ""]
    md += ["## 1. Deviations from the paper", ""] + ([f"- {d}" for d in report.deviations] or ["- none recorded"]) + [""]
    md += ["## 2. Claims", "", "| claim | paper | ours | ±tol | outcome | note |", "|---|---|---|---|---|---|"]
    for c in report.claims:
        ours = "—" if c.our_value is None else f"{c.our_value:.4g}" + (f" ± {c.our_std:.2g}" if c.our_std else "")
        tol = "—" if c.tolerance is None else f"{c.tolerance:.3g}"
        md.append(f"| {c.claim_id} | {c.paper_value:.4g} | {ours} | {tol} | {c.outcome.value} | {c.note} |")
    md += ["", "## 3. Leakage and adversarial checks", "", "| test | result | detail |", "|---|---|---|"]
    for l in report.leakage:
        r = "not run" if l.passed is None else ("PASS" if l.passed else "FAIL")
        md.append(f"| {l.test} | {r} | {l.detail} |")
    md += ["", "## 4. Convention grid (sensitivity, not search)", ""]
    if report.grid_range:
        md.append(f"Headline ranges **{report.grid_range[0]:.4g} to {report.grid_range[1]:.4g}** across defensible conventions.")
    md += ["", "| flip | value | headline | Δ vs default |", "|---|---|---|---|"]
    for p in report.grid:
        hv = "—" if p.headline_value is None else f"{p.headline_value:.4g}"
        d = "—" if p.delta_from_default is None else f"{p.delta_from_default:+.4g}"
        md.append(f"| {p.config_key} | {p.value} | {hv} | {d} |")
    md += ["", "## 5. Ambiguities and defaults used", "", "| key | default | source | sensitivity |", "|---|---|---|---|"]
    for a in report.ambiguities_used:
        md.append(f"| {a['config_key']} | {a['default']} | {a['source']} | {a['sensitivity']} |")
    md += ["", "## 6. Unexplained", ""] + ([f"- {u}" for u in report.unexplained] or ["- nothing outstanding"])
    diag = report.reproducibility.get("diagnostics") or {}
    md += ["", "## 7. Diagnostics (what the build gate and the runs reported)", "", "```json", json.dumps(diag, indent=1, default=str), "```"]
    rest = {k: v for k, v in report.reproducibility.items() if k != "diagnostics"}
    md += ["", "## 8. Reproducibility", "", "```json", json.dumps(rest, indent=1, default=str), "```", ""]
    md += [f"Wall: {report.wall_minutes:.1f} min · Human: {report.human_minutes:.0f} min · Cost: ${report.cost_usd:.2f} · Stages: {', '.join(report.stages_completed)}"]
    p = out_dir / "REPORT.md"
    p.write_text("\n".join(md))
    return p
