# Replication report: Layer Normalization

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** F  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: no claims; integrity: not_run)

> **Run did not complete:** SameFileError: PosixPath('/Users/vzhu/Developer/research-copy/runs/batch/layernorm/work/metrics_seed1.json') and PosixPath('/Users/vzhu/Developer/research-copy/runs/batch/layernorm/work/metrics_seed1.json') are the same file

## 1. Deviations from the paper

- mscoco -> UNAVAILABLE: claims depending on this source are Untested
- bookcorpus -> UNAVAILABLE: claims depending on this source are Untested
- cnn_qa_corpus -> UNAVAILABLE: claims depending on this source are Untested
- iam_ondb -> UNAVAILABLE: claims depending on this source are Untested
- downstream_sentence_eval_sets -> UNAVAILABLE: claims depending on this source are Untested

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
 "spec_frozen_hash": "94e8f5277c341a10aed5fd4fb97b945b9a9a5e8d164a943c48b901b202c866e1",
 "frozen_at": "2026-09-15T15:57:52.084253+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/layernorm/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py --reverify batchnorm dqn layernorm lora ppo worldmodels"
}
```

Wall: 0.0 min · Human: 0 min · Cost: $0.00 · Stages: setup