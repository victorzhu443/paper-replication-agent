# Replication report: Layer Normalization

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** F  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: Untested; integrity: failed)

## 1. Deviations from the paper

- mscoco -> UNAVAILABLE: claims depending on this source are Untested
- bookcorpus -> UNAVAILABLE: claims depending on this source are Untested
- cnn_qa_corpus -> UNAVAILABLE: claims depending on this source are Untested
- iam_ondb -> UNAVAILABLE: claims depending on this source are Untested
- downstream_sentence_eval_sets -> UNAVAILABLE: claims depending on this source are Untested
- tolerance for oe_ln_speedup_60pct filled from our own SE (17.8) at verify time, not frozen
- tolerance for draw_test_nll_ln filled from our own SE (0.435) at verify time, not frozen
- tolerance for draw_test_nll_baseline filled from our own SE (0.435) at verify time, not frozen
- Table-1 checkpoint (gate): mnist_mlp_training_examples paper=55000.0 ours=None gap=None; mnist_mlp_hidden_units_per_layer paper=1000.0 ours=None gap=None; mnist_mlp_batch_size_large paper=128.0 ours=None gap=None; mnist_mlp_batch_size_small paper=4.0 ours=None gap=None; binarized_mnist_train_images paper=50000.0 ours=None gap=None; binarized_mnist_validation_images paper=10000.0 ours=None gap=None; binarized_mnist_test_images paper=10000.0 ours=None gap=None; draw_glimpses paper=64.0 ours=None gap=None; draw_lstm_hidden_units paper=256.0 ours=None gap=None; draw_minibatch_size paper=128.0 ours=None gap=None; draw_epochs_for_final_numbers paper=200.0 ours=None gap=None; draw_epochs_shown_in_figure paper=100.0 ours=None gap=None; mscoco_test_splits paper=5.0 ours=None gap=None; mscoco_images_per_test_split paper=1000.0 ours=None gap=None; mscoco_captions_per_test_split paper=5000.0 ours=None gap=None; order_embedding_validation_eval_interval_iterations paper=300.0 ours=None gap=None; skipthought_encoder_dim paper=2400.0 ours=None gap=None; skipthought_checkpoint_interval_iterations paper=50000.0 ours=None gap=None; skipthought_training_iterations paper=1000000.0 ours=None gap=None; skipthought_long_training_iterations paper=1700000.0 ours=None gap=None; iam_ondb_handwriting_sequences paper=12179.0 ours=None gap=None; iam_ondb_writers paper=221.0 ours=None gap=None; iam_ondb_avg_line_length paper=700.0 ours=None gap=None; handwriting_lstm_layers paper=3.0 ours=None gap=None; handwriting_lstm_cells_per_layer paper=400.0 ours=None gap=None; handwriting_output_mixture_components paper=20.0 ours=None gap=None; handwriting_window_vector_size paper=57.0 ours=None gap=None; handwriting_window_mixture_components paper=10.0 ours=None gap=None; handwriting_total_weights paper=3700000.0 ours=None gap=None; handwriting_minibatch_size paper=8.0 ours=None gap=None; handwriting_sequence_length paper=500.0 ours=None gap=None; cnn_qa_sentences_per_passage paper=4.0 ours=None gap=None; num_experimental_tasks paper=6.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| oe_ln_speedup_60pct | 60 | 107.2 ± 25 | 36.1 | Untested | reduced scale (compute tier 2): ours=107.2 is not comparable to the paper-scale value; |gap|=47.21 > tol 36.09 |
| draw_test_nll_ln | 82.09 | 111.8 ± 0.62 | 0.875 | Untested | reduced scale (compute tier 2): ours=111.8 is not comparable to the paper-scale value; |gap|=29.67 > tol 0.8754 |
| draw_test_nll_baseline | 82.36 | 111.8 ± 0.62 | 0.875 | Untested | reduced scale (compute tier 2): ours=111.8 is not comparable to the paper-scale value; |gap|=29.4 > tol 0.8754 |
| t2_sym_cap_r1 | 45.4 | — | — | Untested | no value produced |
| t2_sym_cap_r10 | 88.7 | — | — | Untested | no value produced |
| t2_sym_cap_meanr | 5.8 | — | — | Untested | no value produced |
| t2_sym_img_r1 | 36.3 | — | — | Untested | no value produced |
| t2_sym_img_r10 | 85.8 | — | — | Untested | no value produced |
| t2_sym_img_meanr | 9 | — | — | Untested | not targeted in plan |
| t2_oe_vendrov_cap_r1 | 46.7 | — | — | Untested | not targeted in plan |
| t2_oe_vendrov_cap_r10 | 88.9 | — | — | Untested | not targeted in plan |
| t2_oe_vendrov_cap_meanr | 5.7 | — | — | Untested | not targeted in plan |
| t2_oe_vendrov_img_r1 | 37.9 | — | — | Untested | not targeted in plan |
| t2_oe_vendrov_img_r10 | 85.9 | — | — | Untested | not targeted in plan |
| t2_oe_vendrov_img_meanr | 8.1 | — | — | Untested | not targeted in plan |
| t2_oe_ours_cap_r1 | 46.6 | — | — | Untested | not targeted in plan |
| t2_oe_ours_cap_r5 | 79.3 | — | — | Untested | not targeted in plan |
| t2_oe_ours_cap_r10 | 89.1 | — | — | Untested | not targeted in plan |
| t2_oe_ours_cap_meanr | 5.2 | — | — | Untested | not targeted in plan |
| t2_oe_ln_cap_r1 | 48.5 | — | — | Untested | not targeted in plan |
| t2_oe_ln_cap_r5 | 80.6 | — | — | Untested | not targeted in plan |
| t2_oe_ln_cap_r10 | 89.8 | — | — | Untested | not targeted in plan |
| t2_oe_ln_cap_meanr | 5.1 | — | — | Untested | not targeted in plan |
| t2_oe_ours_img_r1 | 37.8 | — | — | Untested | not targeted in plan |
| t2_oe_ours_img_r5 | 73.6 | — | — | Untested | not targeted in plan |
| t2_oe_ours_img_r10 | 85.7 | — | — | Untested | not targeted in plan |
| t2_oe_ours_img_meanr | 7.9 | — | — | Untested | not targeted in plan |
| t2_oe_ln_img_r1 | 38.9 | — | — | Untested | not targeted in plan |
| t2_oe_ln_img_r5 | 74.3 | — | — | Untested | not targeted in plan |
| t2_oe_ln_img_r10 | 86.3 | — | — | Untested | not targeted in plan |
| t2_oe_ln_img_meanr | 7.6 | — | — | Untested | not targeted in plan |
| t3_orig_sick_r | 0.848 | — | — | Untested | not targeted in plan |
| t3_orig_sick_rho | 0.778 | — | — | Untested | not targeted in plan |
| t3_orig_sick_mse | 0.287 | — | — | Untested | not targeted in plan |
| t3_orig_mr | 75.5 | — | — | Untested | not targeted in plan |
| t3_orig_cr | 79.3 | — | — | Untested | not targeted in plan |
| t3_orig_subj | 92.1 | — | — | Untested | not targeted in plan |
| t3_orig_mpqa | 86.9 | — | — | Untested | not targeted in plan |
| t3_ours_sick_r | 0.842 | — | — | Untested | not targeted in plan |
| t3_ourslen_sick_r | 0.854 | — | — | Untested | not targeted in plan |
| t3_ourslen_month_sick_r | 0.858 | — | — | Untested | not targeted in plan |
| t3_ours_sick_rho | 0.767 | — | — | Untested | not targeted in plan |
| t3_ourslen_sick_rho | 0.785 | — | — | Untested | not targeted in plan |
| t3_ourslen_month_sick_rho | 0.788 | — | — | Untested | not targeted in plan |
| t3_ours_sick_mse | 0.298 | — | — | Untested | not targeted in plan |
| t3_ourslen_sick_mse | 0.277 | — | — | Untested | not targeted in plan |
| t3_ourslen_month_sick_mse | 0.27 | — | — | Untested | not targeted in plan |
| t3_ours_mr | 77.3 | — | — | Untested | not targeted in plan |
| t3_ourslen_mr | 79.5 | — | — | Untested | not targeted in plan |
| t3_ourslen_month_mr | 79.4 | — | — | Untested | not targeted in plan |
| t3_ours_cr | 81.8 | — | — | Untested | not targeted in plan |
| t3_ourslen_cr | 82.6 | — | — | Untested | not targeted in plan |
| t3_ourslen_month_cr | 83.1 | — | — | Untested | not targeted in plan |
| t3_ours_subj | 92.6 | — | — | Untested | not targeted in plan |
| t3_ourslen_subj | 93.4 | — | — | Untested | not targeted in plan |
| t3_ourslen_month_subj | 93.7 | — | — | Untested | not targeted in plan |
| t3_ours_mpqa | 87.9 | — | — | Untested | not targeted in plan |
| t3_ourslen_mpqa | 89 | — | — | Untested | not targeted in plan |
| t3_ourslen_month_mpqa | 89.3 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | FAIL | shuffled convergence_time_ratio_vs_baseline=291.7 (t=None), unshuffled=107.2 |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | constant | conventions_kb | medium |
| learning_rate | 0.001 | conventions_kb | high |
| adam_betas_eps | beta1=0.9,beta2=0.999,eps=1e-8 | conventions_kb | medium |
| augmentation | none | conventions_kb | low |
| batch_vs_steps | epochs | paper_text | high |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | paper_text | medium |
| ln_eps | 1e-5 | conventions_kb | low |
| ln_param_init | gain=1,bias=0 | paper_text | high |
| ln_bias_handling | ln_beta_only | conventions_kb | low |
| activation | relu | conventions_kb | medium |
| weight_init | glorot_uniform | conventions_kb | medium |
| n_seeds | 1 | guess | high |
| max_epochs | 60 | paper_text | high |
| bn_scope | all_layers | paper_text | medium |
| variance_estimator | biased_LN_unbiased_BN_small_batch | paper_text | low |
| bn_inference_stats | running_average_momentum_0.9 | conventions_kb | medium |
| val_split | last_5000 | conventions_kb | low |
| regularization | none | conventions_kb | medium |
| grad_clip | none | guess | medium |
| replication_task | mnist_mlp | conventions_kb | high |
| speedup_metric | iterations_for_LN_to_reach_its_own_best_vs_baseline_to_reach_its_own_best | paper_text | high |
| checkpoint_criterion | sum_of_R@1_R@5_R@10 | conventions_kb | medium |
| reporting_statistic | final_epoch | conventions_kb | medium |
| hidden_width | 1000 | paper_text | medium |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "94e8f5277c341a10aed5fd4fb97b945b9a9a5e8d164a943c48b901b202c866e1",
 "frozen_at": "2026-09-15T15:57:52.084253+00:00",
 "env_lock_hash": "98fa794f4e123348",
 "runs": [
  "full_draw_baseline_s0_7ea3ba0f22c43618.json",
  "full_draw_baseline_s1_7ea3ba0f22c43618.json",
  "full_draw_ln_s0_1d1479697e5f3c19.json",
  "full_draw_ln_s1_1d1479697e5f3c19.json",
  "full_oe_ln_s0_0e1ac5c5a89553be.json",
  "full_oe_ln_s1_0e1ac5c5a89553be.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/layernorm/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py"
}
```

Wall: 79.2 min · Human: 0 min · Cost: $1.76 · Stages: setup, build, run, verify