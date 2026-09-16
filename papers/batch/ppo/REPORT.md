# Replication report: Proximal Policy Optimization Algorithms

**Kind of test:** mechanics_only · **Data tier:** C · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** F  (data: synthetic; procedure: re-implemented, 0 unexplained; result: no claims; integrity: not_run)

> **Run did not complete:** SameFileError: PosixPath('/Users/vzhu/Developer/research-copy/runs/batch/ppo/work/metrics_seed1.json') and PosixPath('/Users/vzhu/Developer/research-copy/runs/batch/ppo/work/metrics_seed1.json') are the same file

## 1. Deviations from the paper

- openai_gym_mujoco -> UNAVAILABLE: claims depending on this source are Untested
- ale_atari -> UNAVAILABLE: claims depending on this source are Untested
- roboschool -> UNAVAILABLE: claims depending on this source are Untested

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
 "spec_frozen_hash": "f3ac1d57ceba896af33d21c8dd20e50b8d5bfa81119ab9dad6306602670fe721",
 "frozen_at": "2026-09-15T16:05:17.087494+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/ppo/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py --reverify batchnorm dqn layernorm lora ppo worldmodels"
}
```

Wall: 0.0 min · Human: 0 min · Cost: $0.00 · Stages: setup