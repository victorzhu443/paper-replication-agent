# Replication report: A Mathematical Framework for Transformer Circuits

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: Untested; integrity: not_run)

## 1. Deviations from the paper

- Table-1 checkpoint (measure): n_heads_primary_models paper=12.0 ours=None gap=None; d_head_primary_models paper=64.0 ours=None gap=None; d_model_primary_models paper=768.0 ours=None gap=None; n_heads_secondary_model paper=32.0 ours=None gap=None; d_head_secondary_model paper=128.0 ours=None gap=None; d_model_secondary_model paper=4096.0 ours=None gap=None; n_context_tokens paper=2048.0 ours=None gap=None; vocab_size_approx paper=50000.0 ours=None gap=None; expanded_ov_matrix_entries_approx paper=2500000000.0 ours=None gap=None; ov_qk_matrix_rank_primary paper=64.0 ours=None gap=None; ov_qk_matrix_rank_secondary paper=128.0 ours=None gap=None; random_token_sequence_repeats paper=3.0 ours=None gap=None; mlp_share_of_standard_transformer_params paper=0.6667 ours=None gap=None; d_mlp_over_d_model_ratio paper=4.0 ours=None gap=None; d_head_over_d_model_ratio_lower paper=0.01 ours=None gap=None; d_head_over_d_model_ratio_upper paper=0.1 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| c_copying_heads_10_of_12 | 10 | 12 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_kcomposition_layer0_heads | 1 | 10 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_kcomposition_extra_heads_corrected | 2 | 0 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | not run | headline metric 'count_of_copying_heads' has no known better-direction; shuffle test not run |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | cosine | conventions_kb | medium |
| augmentation | none | conventions_kb | low |
| batch_vs_steps | steps | guess | high |
| mixed_precision | fp32 | conventions_kb | low |
| split | not_applicable | paper_text | low |
| data.training_corpus | openwebtext | conventions_kb | high |
| data.tokenizer | gpt2_bpe_50257 | conventions_kb | medium |
| model.size | paper_12x64_d768 | paper_text | high |
| model.n_ctx | 2048 | paper_text | medium |
| model.positional | sinusoidal_into_qk_only | paper_text | high |
| model.attention_scale | one_over_sqrt_dhead | conventions_kb | medium |
| model.layer_norm | ln_as_trained | paper_text | medium |
| eval.ln_folding | fold_ln_into_embedding | paper_text | medium |
| model.biases | no_biases | paper_text | low |
| train.optimizer | adamw_lr1e-3_wd0.1_clip1.0 | conventions_kb | medium |
| train.token_budget | fixed_20k_steps | guess | high |
| eval.copying_claim_model | one_layer_12head_64 | paper_text | high |
| model.two_layer_heads_per_layer | 12 | paper_text | medium |
| eval.copying_score | sum_positive_eig_over_sum_abs_eig | conventions_kb | high |
| eval.copying_threshold | score_gt_0.7 | guess | high |
| eval.prefix_matching_score | mean_attn_to_token_after_earlier_copy_on_repeated_random | conventions_kb | high |
| eval.induction_head_criterion | positive_qk_and_ov_eigenvalue_corner | paper_text | high |
| eval.random_seq | len50_rep3_uniform_with_start | guess | medium |
| eval.loss_gap | mean_loss_second_half_minus_first_half_of_repeated_seq | guess | high |
| eval.composition_baseline | mean_over_N_random_gaussian_pairs_same_shape | paper_text | medium |
| eval.composition_formula | both | paper_text | high |
| train.n_seeds | 1 | paper_text | high |
| eval.n_eval_sequences | 1 | paper_text | medium |
| eval.checkpoint | final | conventions_kb | low |
| eval.matrix_algebra | factored_lambda_AB_equals_BA | paper_text | low |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "783529a6fd8b4645c42ac023a7ff551d575dec355652d933e10bbded9df5c3ba",
 "frozen_at": "2026-09-15T16:22:20.419689+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [
  "full_one_layer_12head_64_s0_3908eda2ff5d51cf.json",
  "full_two_layer_12head_64_composition_buggy_s0_f85b7386d15918d3.json",
  "full_two_layer_12head_64_composition_corrected_s0_9e4cd30a823ea390.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/circuits/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py"
}
```

Wall: 18.9 min · Human: 0 min · Cost: $1.63 · Stages: setup, build, run, verify