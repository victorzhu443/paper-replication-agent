"""CS/ML template. The contract is a `reproduce.sh` in the work dir that writes `metrics.json`:
    {"seed": int, "split": str, "n_examples": int, "metrics": {"accuracy": 0.982, ...}}
One invocation per seed. The builder writes the code; this module only runs and parses.
Oracle-first: if the repo already has an eval script for released weights, `reproduce.sh` calls it.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def run_reproduce(workdir: Path, config: dict[str, Any], seed: int, timeout_s: int = 1800,
                  python: str | None = None) -> dict[str, Any]:
    env = {**os.environ, "SEED": str(seed), "REPLICATOR_CONFIG": json.dumps(config),
           "SMOKE": "1" if config.get("_smoke") else "0", "RUN_TIMEOUT_S": str(timeout_s),
           "HF_HOME": os.environ.get("HF_HOME", str(Path(__file__).resolve().parents[2] / "data_cache" / "hf")),
           "TOKENIZERS_PARALLELISM": "false"}
    if python:
        env["PYTHON"] = python
    workdir = workdir.resolve()
    out = workdir / f"metrics_seed{seed}.json"
    if out.exists():
        out.unlink()
    env["METRICS_OUT"] = str(out)
    proc = subprocess.run(["bash", "reproduce.sh"], cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"reproduce.sh failed (seed {seed}):\n{proc.stderr[-3000:]}")
    if not out.exists():
        raise RuntimeError("reproduce.sh did not write METRICS_OUT")
    return json.loads(out.read_text())


def aggregate_seeds(results: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({k for r in results for k in r["metrics"]})
    out: dict[str, float] = {"n_seeds": float(len(results))}
    for k in keys:
        vals = np.array([r["metrics"][k] for r in results if k in r["metrics"]], dtype=float)
        out[k] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    if results and "n_examples" in results[0]:
        out["n_examples"] = float(results[0]["n_examples"])
    return out


def split_size_checkpoint(reported: int | None, ours: int | None, rel_tol: float = 0.02) -> tuple[bool | None, str]:
    if reported is None or ours is None:
        return None, "no split size to compare"
    gap = abs(ours - reported) / max(reported, 1)
    return gap <= rel_tol, f"reported {reported}, ours {ours}, rel gap {gap:.3%}"
