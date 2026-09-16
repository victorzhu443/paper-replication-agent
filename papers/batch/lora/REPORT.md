# Replication report: LoRA: Low-Rank Adaptation of Large Language Models

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: Untested; integrity: not_run)

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
- Table-1 checkpoint (gate): trainable_params_millions_roberta_base_ft paper=125.0 ours=None gap=None; trainable_params_millions_roberta_base_lora paper=0.3 ours=None gap=None; trainable_params_millions_roberta_large_ft paper=355.0 ours=None gap=None; trainable_params_millions_roberta_large_lora paper=0.8 ours=None gap=None; trainable_params_millions_deberta_xxl_ft paper=1500.0 ours=None gap=None; trainable_params_millions_deberta_xxl_lora paper=4.7 ours=None gap=None; trainable_params_millions_gpt2_medium_ft paper=354.92 ours=None gap=None; trainable_params_millions_gpt2_medium_lora paper=0.35 ours=None gap=None; trainable_params_millions_gpt2_large_ft paper=774.03 ours=None gap=None; trainable_params_millions_gpt2_large_lora paper=0.77 ours=None gap=None; trainable_params_millions_gpt3_ft paper=175255.8 ours=None gap=None; trainable_params_millions_gpt3_lora_small paper=4.7 ours=None gap=None; trainable_params_millions_gpt3_lora_large paper=37.7 ours=None gap=None; wikisql_train_examples paper=56355.0 ours=None gap=None; wikisql_validation_examples paper=8421.0 ours=None gap=None; samsum_train_examples paper=14732.0 ours=None gap=None; samsum_test_examples paper=819.0 ours=None gap=None; e2e_train_examples_approx paper=42000.0 ours=None gap=None; e2e_validation_examples_approx paper=4600.0 ours=None gap=None; e2e_test_examples_approx paper=4600.0 ours=None gap=None; dart_examples_approx paper=82000.0 ours=None gap=None; webnlg_examples_approx paper=22000.0 ours=None gap=None; mnli_full_train_examples_thousands paper=392.0 ours=None gap=None; gpt3_transformer_layers paper=96.0 ours=None gap=None; gpt3_d_model_full_rank paper=12288.0 ours=None gap=None; gpt3_lora_checkpoint_size_reduction_factor paper=10000.0 ours=None gap=None; gpt3_vram_full_ft_tb paper=1.2 ours=None gap=None; gpt3_lora_training_speedup_percent paper=25.0 ours=None gap=None; latency_trials_averaged paper=100.0 ours=None gap=None; gpt3_fewshot_mnli_m_accuracy paper=40.6 ours=None gap=None; gpt3_fewshot_rte_accuracy paper=69.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| rob_base_lora_sst2 | 95.1 | 91.97 | 0.45 | Untested | reduced scale (compute tier 2): ours=91.97 is not comparable to the paper-scale value; |gap|=3.128 > tol 0.45 |
| rob_base_ft_sst2 | 94.8 | 91.97 | 1.55 | Untested | reduced scale (compute tier 2): ours=91.97 is not comparable to the paper-scale value; |gap|=2.828 > tol 1.554 |
| rob_base_lora_trainable_params | 0.3 | 0.2949 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| rob_base_ft_trainable_params | 125 | — | — | Untested | not targeted in plan |
| rob_base_lora_glue_avg | 87.2 | 91.97 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| rob_base_ft_glue_avg | 86.4 | 91.97 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| rob_base_lora_mnli | 87.5 | 91.97 | 0.65 | Untested | reduced scale (compute tier 2): ours=91.97 is not comparable to the paper-scale value; |gap|=4.472 > tol 0.65 |
| rob_base_ft_mnli | 87.6 | 91.97 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| rob_base_lora_mrpc | 89.7 | 91.97 | 1.45 | Untested | reduced scale (compute tier 2): ours=91.97 is not comparable to the paper-scale value; |gap|=2.272 > tol 1.45 |
| rob_base_lora_cola | 63.4 | — | — | Untested | not targeted in plan |
| rob_large_lora_sst2 | 96.2 | — | — | Untested | not targeted in plan |
| rob_large_ft_sst2 | 96.4 | — | — | Untested | not targeted in plan |
| deberta_lora_glue_avg | 91.3 | — | — | Untested | not targeted in plan |
| deberta_ft_glue_avg | 91.1 | — | — | Untested | not targeted in plan |
| gpt2m_lora_e2e_bleu | 70.4 | — | — | Untested | not targeted in plan |
| gpt2m_ft_e2e_bleu | 68.2 | — | — | Untested | not targeted in plan |
| gpt3_lora_wikisql | 73.4 | — | — | Untested | not targeted in plan |
| gpt3_ft_wikisql | 73.8 | — | — | Untested | not targeted in plan |
| gpt3_lora_mnli_m | 91.7 | — | — | Untested | not targeted in plan |
| gpt3_ft_mnli_m | 89.5 | — | — | Untested | not targeted in plan |
| gpt3_lora_trainable_params | 4.7 | — | — | Untested | not targeted in plan |
| gpt3_ft_trainable_params | 1.753e+05 | — | — | Untested | not targeted in plan |
| gpt3_weighttype_qv_r4_wikisql | 73.7 | — | — | Untested | not targeted in plan |
| gpt3_weighttype_q_only_wikisql | 70.4 | — | — | Untested | not targeted in plan |
| gpt3_rank_r1_qv_wikisql | 73.4 | — | — | Untested | not targeted in plan |
| gpt3_rank_r4_qv_wikisql | 73.7 | — | — | Untested | not targeted in plan |
| gpt2m_rank_r4_e2e_bleu | 70.38 | — | — | Untested | not targeted in plan |
| gpt2m_rank_r4_val_loss | 1.18 | — | — | Untested | not targeted in plan |
| gpt3_lowdata_mnli100_lora | 63.8 | — | — | Untested | not targeted in plan |
| gpt3_lowdata_mnli100_ft | 60.2 | — | — | Untested | not targeted in plan |
| adapter_latency_batch1_lora | 19.8 | — | — | Untested | not targeted in plan |
| adapter_latency_batch1_adapterH | 25.8 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | not run | shuffled accuracy=50.92, unshuffled=91.97; no null reference (chance_level / accuracy_random_policy / accuracy_baseline) reported, test not judged |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| base_model | roberta_base_125m_as_paper | paper_text | high |
| train_subset_size | full_train_split | paper_text | high |
| train_subset_sampling | stratified_random_seed0 | guess | medium |
| num_epochs | 60 | paper_text | high |
| batch_vs_steps | steps | conventions_kb | high |
| lr_schedule | as_paper | paper_text | medium |
| learning_rate | lora=5e-4,ft=2e-5 | paper_text | high |
| batch_size | 16 | paper_text | medium |
| max_seq_len | 512 | paper_text | low |
| lora_rank | 8 | paper_text | medium |
| lora_alpha | 8 | paper_text | medium |
| lora_target_modules | q_v_only | paper_text | medium |
| lora_layer_coverage | all_layers | paper_text | medium |
| lora_dropout | 0.0 | conventions_kb | low |
| weight_decay | 0.01 | conventions_kb | low |
| n_seeds | 5_median | paper_text | medium |
| checkpoint_selection | best_epoch_on_eval | paper_text | medium |
| error_bar_semantics | std_over_seeds | paper_text | low |
| split | validation | paper_text | high |
| mnli_metric | overall_m_and_mm_accuracy | paper_text | medium |
| glue_avg_definition | unweighted_mean_of_eight_task_metrics | conventions_kb | low |
| wikisql_accuracy | logical_form | paper_text | medium |
| gpt3_lora_4p7m_config | rv_2 | paper_text | medium |
| e2e_eval_script | official_e2e_metrics_script | conventions_kb | high |
| head_treatment | trainable_not_counted | paper_text | medium |
| param_fraction_denominator | all_base_params_incl_embeddings | paper_text | low |
| mnli_transfer_init | mnli_init_for_mrpc_rte_stsb | paper_text | medium |
| augmentation | none | conventions_kb | low |
| mixed_precision | fp32 | conventions_kb | low |
| pooling | last_non_pad_token | conventions_kb | medium |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "2aa7b80bebda147e4174b15fcf902cd087ff89cc1a41f90e298e5a2a4d8bbc1c",
 "frozen_at": "2026-09-16T17:46:01.628587+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [
  "full_roberta_base_ft_s0_c3f0ba765a59797c.json",
  "full_roberta_base_lora_s0_1ee830f510fb28fa.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/lora/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py --reverify lora ppo worldmodels"
}
```

Wall: 45.1 min · Human: 0 min · Cost: $0.00 · Stages: setup, build, run, verify