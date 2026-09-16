# Replication report: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** F  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: no claims; integrity: not_run)

> **Run did not complete:** SameFileError: PosixPath('/Users/vzhu/Developer/research-copy/runs/batch/batchnorm/work/patch.py') and PosixPath('/Users/vzhu/Developer/research-copy/runs/batch/batchnorm/work/patch.py') are the same file

## 1. Deviations from the paper

- imagenet_ilsvrc2012 -> UNAVAILABLE: claims depending on this source are Untested

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
 "spec_frozen_hash": "a5a2b6e4c2d1c3c1098cd5ca9e1202d40683dbcab8837c10066476f242e64a01",
 "frozen_at": "2026-09-15T15:57:21.969457+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/batchnorm/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py --reverify batchnorm dqn layernorm lora ppo worldmodels"
}
```

Wall: 0.0 min · Human: 0 min · Cost: $0.00 · Stages: setup