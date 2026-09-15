"""Convention grid: every high-sensitivity ambiguity as a one-factor flip, run as one batch,
reported as a range. Sensitivity analysis, not search (DESIGN.md stage 9 part 1)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ..schema import Ambiguity, GridPoint


def run_grid(run_fn: Callable[[dict, int], dict], base_config: dict, ambiguities: list[Ambiguity], headline: str,
             baseline_value: float | None, seed: int = 0, max_workers: int = 4,
             sensitivities: tuple[str, ...] = ("high", "medium")) -> tuple[list[GridPoint], tuple[float, float] | None]:
    flips: list[tuple[str, str]] = []
    for a in ambiguities:
        if a.sensitivity not in sensitivities:
            continue
        for opt in a.options:
            if opt != str(base_config.get(a.config_key, a.default)):
                flips.append((a.config_key, opt))

    def one(flip):
        key, val = flip
        cfg = {**base_config, key: val}
        try:
            m = run_fn(cfg, seed)
            v = m.get(headline)
        except Exception as e:  # noqa: BLE001
            return GridPoint(config_key=key, value=val, headline_value=None, delta_from_default=None), str(e)
        d = None if (v is None or baseline_value is None) else v - baseline_value
        return GridPoint(config_key=key, value=val, headline_value=v, delta_from_default=d), None

    points: list[GridPoint] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for gp, err in ex.map(one, flips):
            points.append(gp)
    vals = [p.headline_value for p in points if p.headline_value is not None]
    if baseline_value is not None:
        vals.append(baseline_value)
    rng = (min(vals), max(vals)) if vals else None
    return points, rng
