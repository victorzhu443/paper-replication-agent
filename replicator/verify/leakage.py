"""Three mechanical leakage tests, three classes of leakage (DESIGN.md stage 8).

  shuffle             : label leakage (bad split, target encoding). Result survives shuffled labels => fail.
  future_perturbation : feature look-ahead. Replace data after t with noise; features at/before t must be identical.
  available_at_audit  : lineage timestamps precede decision time.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..schema import LeakageResult

LOWER_IS_BETTER = {"test_error", "error_rate", "error", "loss", "perplexity", "rmse", "mae", "mse", "nll", "val_loss"}
HIGHER_IS_BETTER = {"accuracy", "acc", "top1", "top5", "f1", "bleu", "rouge", "auc", "mean_return", "return", "reward",
                    "episode_return", "sharpe", "mean_return_pct", "r2", "oos_r2", "t_stat", "exact_match", "win_rate"}


def metric_known(metric: str) -> bool:
    m = metric.lower()
    return m in LOWER_IS_BETTER or m in HIGHER_IS_BETTER or any(m.endswith("_" + k) or m.startswith(k + "_") for k in LOWER_IS_BETTER | HIGHER_IS_BETTER)


def metric_skill(metric: str, value: float) -> float:
    """Distance from 'no skill' on a better-is-larger scale. Error-type metrics are inverted
    against their ceiling (100 for percent, 1 for decimal); everything else is taken as-is."""
    m = metric.lower()
    if m in LOWER_IS_BETTER or any(m.endswith("_" + k) for k in LOWER_IS_BETTER):
        if m in ("loss", "perplexity", "rmse", "mae", "mse"):
            return -abs(value)
        ceiling = 100.0 if value > 1.0 else 1.0
        return ceiling - value
    return abs(value)


def shuffle_test(run_fn: Callable[[dict, int], dict], config: dict, headline: str, baseline_value: float,
                 seed: int = 0) -> LeakageResult:
    """run_fn(config, seed) must honor config['_shuffle_labels']=True by shuffling labels/returns
    within each period before the model/sort sees them."""
    cfg = {**config, "_shuffle_labels": True}
    try:
        m = run_fn(cfg, seed)
    except Exception as e:  # noqa: BLE001
        return LeakageResult(test="shuffle", passed=None, detail=f"could not run: {e}")
    v = m.get(headline)
    t = m.get("t_stat")
    if v is None:
        return LeakageResult(test="shuffle", passed=None, detail="headline missing from shuffled run")
    if t is None and not metric_known(headline):
        return LeakageResult(test="shuffle", passed=None,
                             detail=f"headline metric '{headline}' has no known better-direction; shuffle test not run")
    if not m.get("_shuffled"):
        # A pipeline that ignores the flag reproduces its real number, which would read as a leak.
        # Without an acknowledgement the test cannot distinguish the two, so it is not run.
        return LeakageResult(test="shuffle", passed=None,
                             detail="pipeline did not acknowledge the label shuffle (_shuffled flag absent); test not run")
    # The leak indicator is a shuffled result that is still significant. The magnitude criterion
    # is only meaningful when the unshuffled baseline was itself a real effect, so it is used
    # only when no t-stat is available.
    if t is not None:
        survives = abs(t) > 2.0
    else:
        base_skill = metric_skill(headline, baseline_value)
        shuf_skill = metric_skill(headline, v)
        survives = base_skill > 0 and shuf_skill > 0.5 * base_skill
    return LeakageResult(test="shuffle", passed=not survives,
                         detail=f"shuffled {headline}={v:.4g} (t={t}), unshuffled={baseline_value:.4g}")


def future_perturbation_test(build_features: Callable[[pd.DataFrame], pd.DataFrame], raw: pd.DataFrame,
                             date_col: str = "date", cutoff_frac: float = 0.6, seed: int = 0) -> LeakageResult:
    """Change the future; the past must not move. Works on any feature builder without lineage."""
    try:
        raw = raw.sort_values(date_col).reset_index(drop=True)
        dates = raw[date_col].drop_duplicates().sort_values().to_numpy()
        t = dates[int(len(dates) * cutoff_frac)]
        base = build_features(raw.copy())
        rng = np.random.default_rng(seed)
        pert = raw.copy()
        fut = pert[date_col] > t
        num = pert.select_dtypes("number").columns
        pert.loc[fut, num] = rng.normal(size=(int(fut.sum()), len(num)))
        alt = build_features(pert)
        b = base[base[date_col] <= t].reset_index(drop=True)
        a = alt[alt[date_col] <= t].reset_index(drop=True)
        common = [c for c in b.columns if c in a.columns]
        b, a = b[common], a[common]
        nb = b.select_dtypes("number").to_numpy(dtype=float)
        na = a.select_dtypes("number").to_numpy(dtype=float)
        same_shape = nb.shape == na.shape
        moved = int((~np.isclose(nb, na, equal_nan=True)).sum()) if same_shape else -1
        passed = same_shape and moved == 0
        return LeakageResult(test="future_perturbation", passed=passed,
                             detail=f"cutoff={pd.Timestamp(t).date()}, cells moved={moved}, shape_equal={same_shape}")
    except Exception as e:  # noqa: BLE001
        return LeakageResult(test="future_perturbation", passed=None, detail=f"could not run: {e}")


def available_at_audit(pit, feature: str, decision_time: pd.Series) -> LeakageResult:
    try:
        ok, n = pit.audit(feature, decision_time)
        return LeakageResult(test="available_at_audit", passed=ok, detail=f"{n} rows with available_at > decision_time")
    except Exception as e:  # noqa: BLE001
        return LeakageResult(test="available_at_audit", passed=None, detail=f"could not run: {e}")
