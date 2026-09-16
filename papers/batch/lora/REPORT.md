# Replication report: LoRA: Low-Rank Adaptation of Large Language Models

**Kind of test:** mechanics_only · **Data tier:** C · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** F  (data: synthetic; procedure: re-implemented, 0 unexplained; result: no claims; integrity: not_run)

> **Run did not complete:** RuntimeError: build did not pass smoke: {'passed': False, 'problems': ['a SCALE=0.1 run took 2019s, so the full run would take ~337 min > timeout 25 min: reduce the default scale (steps/epochs/data/model) until SCALE=0.1 finishes in under 135s'], 'metrics': {'accuracy': 52.34375, 'accuracy_full_ft': 53.125, 'accuracy_gap_lora_minus_ft': -0.78125, 'glue_avg': 52.34375, 'glue_avg_full_ft': 53.125, 'trainable_parameters_millions': 0.294912, 'trainable_parameters_millions_ft': 124.05504, 'trainable_param_fraction_percent': 0.2377267380672321, '_intermediates': {'n_train_examples': 256, 'n_eval_examples': 128, 'steps': 16, 'batch_size': 16, 'eval_curve_lora': [[8, 52.34375], [16, 52.34375]], 'eval_curve_ft': [[8, 47.65625], [16, 53.125]], 'train_seconds': {'lora': 1851.6, 'ft': 2365.5}, 'config_used': {'base_model': 'roberta_base_125m_as_paper', 'train_subset_size': 'full_train_split', 'train_subset_sampling': 'stratified_random_seed0', 'num_epochs': '60', 'batch_vs_steps': 'steps', 'lr_schedule': 'as_paper', 'learning_rate': 'lora=5e-4,ft=2e-5', 'batch_size': '16', 'max_seq_len': '512', 'lora_rank': '8', 'lora_alpha': '8', 'lora_target_modules': 'q_v_only', 'lora_layer_coverage': 'all_layers', 'lora_dropout': '0.0', 'weight_decay': '0.01', 'n_seeds': '5_median', 'checkpoint_selection': 'best_epoch_on_eval', 'error_bar_semantics': 'std_over_seeds', 'split': 'validation', 'mnli_metric': 'overall_m_and_mm_accuracy', 'glue_avg_definition': 'unweighted_mean_of_eight_task_metrics', 'wikisql_accuracy': 'logical_form', 'gpt3_lora_4p7m_config': 'rv_2', 'e2e_eval_script': 'official_e2e_metrics_script', 'head_treatment': 'trainable_not_counted', 'param_fraction_denominator': 'all_base_params_incl_embeddings', 'mnli_transfer_init': 'mnli_init_for_mrpc_rte_stsb', 'augmentation': 'none', 'mixed_precision': 'fp32', 'pooling': 'last_non_pad_token', 'epochs': 1, 'n_examples': 2048}, 'glue_avg_note': "only SST-2 was run; glue_avg is the unweighted mean over the 1 task run (= SST-2 accuracy), NOT the paper's 8-task average", 'scale': {'base_model': 'roberta-base', 'train_examples_seen': 256, 'paper_train_examples': 67349, 'paper_epochs': 60, 'steps_run': 16, 'paper_equivalent_steps': 252558, 'n_seeds': 1, 'paper_n_seeds': 5, 'eval_examples': 128, 'smoke': True, 'scale_factor': 1.0}}}, 'scale_probe': {'scale_0.1_seconds': 2019, 'estimated_full_seconds': 20190, 'run_timeout_s': 1500}}

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

Wall: 137.5 min · Human: 0 min · Cost: $0.00 · Stages: setup, build