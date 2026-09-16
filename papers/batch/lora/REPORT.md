# Replication report: LoRA: Low-Rank Adaptation of Large Language Models

**Kind of test:** mechanics_only · **Data tier:** C · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** F  (data: synthetic; procedure: re-implemented, 0 unexplained; result: no claims; integrity: not_run)

> **Run did not complete:** SameFileError: PosixPath('/Users/vzhu/Developer/research-copy/runs/batch/lora/work/patch.py') and PosixPath('/Users/vzhu/Developer/research-copy/runs/batch/lora/work/patch.py') are the same file

## 1. Deviations from the paper

- glue_sst2 -> UNAVAILABLE: claims depending on this source are Untested
- glue_benchmark -> UNAVAILABLE: claims depending on this source are Untested
- e2e_nlg -> UNAVAILABLE: claims depending on this source are Untested
- wikisql -> UNAVAILABLE: claims depending on this source are Untested
- samsum -> UNAVAILABLE: claims depending on this source are Untested
- hf_pretrained_roberta_base -> UNAVAILABLE: claims depending on this source are Untested
- hf_pretrained_roberta_large -> UNAVAILABLE: claims depending on this source are Untested
- hf_pretrained_deberta_xxl -> UNAVAILABLE: claims depending on this source are Untested
- hf_pretrained_gpt2_medium -> UNAVAILABLE: claims depending on this source are Untested
- hf_pretrained_gpt2 -> UNAVAILABLE: claims depending on this source are Untested
- gpt3_175b_checkpoint -> UNAVAILABLE: claims depending on this source are Untested

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
 "spec_frozen_hash": "fe8ba399176e0de1b7a807cc4e22a851efc224385a135df8897b6f7a50c2d4ad",
 "frozen_at": "2026-09-15T16:17:34.869541+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/lora/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py --reverify batchnorm dqn layernorm lora ppo worldmodels"
}
```

Wall: 0.0 min · Human: 0 min · Cost: $0.00 · Stages: setup