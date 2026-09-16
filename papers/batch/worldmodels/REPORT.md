# Replication report: World Models

**Kind of test:** mechanics_only · **Data tier:** C · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** F  (data: synthetic; procedure: re-implemented, 0 unexplained; result: no claims; integrity: not_run)

> **Run did not complete:** SameFileError: PosixPath('/Users/vzhu/Developer/research-copy/runs/batch/worldmodels/work/patch.py') and PosixPath('/Users/vzhu/Developer/research-copy/runs/batch/worldmodels/work/patch.py') are the same file

## 1. Deviations from the paper

- openai_gym_carracing_v0 -> UNAVAILABLE: claims depending on this source are Untested
- vizdoom_takecover_v0 -> UNAVAILABLE: claims depending on this source are Untested
- random_policy_rollout_dataset -> UNAVAILABLE: claims depending on this source are Untested

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "52ffd1068f3942f460bad89e30b19f2b9fb7c3054362dcd2d5e7f2f4d93f90df",
 "frozen_at": "2026-09-15T16:12:55.568686+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/worldmodels/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py --reverify batchnorm dqn layernorm lora ppo worldmodels"
}
```

Wall: 0.0 min · Human: 0 min · Cost: $0.00 · Stages: setup