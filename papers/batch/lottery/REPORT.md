# Replication report: The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: Untested; integrity: not_run)

## 1. Deviations from the paper

- Table-1 checkpoint (gate): mnist_training_examples_total paper=60000.0 ours=None gap=None; mnist_test_examples paper=10000.0 ours=None gap=None; mnist_validation_examples paper=5000.0 ours=None gap=None; mnist_training_examples_used paper=55000.0 ours=None gap=None; lenet_total_weights paper=266000.0 ours=None gap=None; lenet_output_layer_weights paper=1000.0 ours=None gap=None; lenet_hidden1_units paper=300.0 ours=None gap=None; lenet_hidden2_units paper=100.0 ours=None gap=None; lenet_batch_size paper=60.0 ours=None gap=None; lenet_training_iterations_per_round paper=50000.0 ours=None gap=None; lenet_pruning_rate_per_round_percent paper=20.0 ours=None gap=None; lenet_output_layer_pruning_rate_percent paper=10.0 ours=None gap=None; lenet_train_acc_100_threshold_at_50k_percent paper=2.0 ours=None gap=None; trials_per_point_main_body paper=5.0 ours=None gap=None; reinitializations_per_trial paper=3.0 ours=None gap=None; trials_per_point_appendix paper=3.0 ours=None gap=None; random_sparse_trials_figure1 paper=10.0 ours=None gap=None; eval_interval_iterations paper=100.0 ours=None gap=None; dropout_rate paper=0.5 ours=None gap=None; cifar10_training_examples_total paper=50000.0 ours=None gap=None; cifar10_training_examples_used paper=45000.0 ours=None gap=None; cifar10_test_examples paper=10000.0 ours=None gap=None; conv2_total_weights paper=4300000.0 ours=None gap=None; conv2_conv_weights paper=38000.0 ours=None gap=None; conv4_total_weights paper=2400000.0 ours=None gap=None; conv4_conv_weights paper=260000.0 ours=None gap=None; conv6_total_weights paper=1700000.0 ours=None gap=None; conv6_conv_weights paper=1100000.0 ours=None gap=None; conv2_training_iterations paper=20000.0 ours=None gap=None; conv4_training_iterations paper=25000.0 ours=None gap=None; conv6_training_iterations paper=30000.0 ours=None gap=None; vgg19_total_weights paper=20000000.0 ours=None gap=None; vgg19_training_iterations paper=112480.0 ours=None gap=None; vgg19_output_layer_params_unpruned paper=5120.0 ours=None gap=None; resnet18_total_weights paper=271000.0 ours=None gap=None; resnet18_training_iterations paper=30000.0 ours=None gap=None; resnet18_output_layer_params_unpruned paper=640.0 ours=None gap=None; resnet18_downsample_params_unpruned paper=2560.0 ours=None gap=None; conv2_fc_share_of_params_percent paper=99.0 ours=None gap=None; conv4_fc_share_of_params_percent paper=89.0 ours=None gap=None; conv6_fc_share_of_params_percent paper=35.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| lenet_ticket_pm_above_original | 2.9 | — | — | Untested | no value produced |
| lenet_reinit_pm_above_original | 21.1 | 100 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| lenet_acc_gain_at_50k | 0.35 | — | — | Untested | no value produced |
| lenet_acc_gain_at_early_stop_pm13_5 | 0.3 | — | — | Untested | no value produced |
| lenet_early_stop_38pct_earlier | 38 | — | — | Untested | no value produced |
| lenet_early_stop_return_pm3_6 | 3.6 | — | — | Untested | no value produced |
| lenet_ticket_vs_reinit_speedup_pm21 | 2.51 | — | — | Untested | no value produced |
| lenet_ticket_vs_reinit_acc_pm21 | 0.5 | — | — | Untested | no value produced |
| lenet_train_acc_100_pm5 | 5 | — | — | Untested | not targeted in plan |
| lenet_oneshot_faster_lower_bound | 17.6 | — | — | Untested | not targeted in plan |
| lenet_oneshot_faster_upper_bound | 67.5 | — | — | Untested | not targeted in plan |
| lenet_oneshot_acc_lower_bound | 5.17 | — | — | Untested | not targeted in plan |
| lenet_oneshot_acc_upper_bound | 95 | — | — | Untested | not targeted in plan |
| conv2_speedup_best | 3.5 | — | — | Untested | not targeted in plan |
| conv4_speedup_best | 3.5 | — | — | Untested | not targeted in plan |
| conv6_speedup_best | 2.5 | — | — | Untested | not targeted in plan |
| conv2_acc_gain_best | 3.4 | — | — | Untested | not targeted in plan |
| conv4_acc_gain_best | 3.5 | — | — | Untested | not targeted in plan |
| conv6_acc_gain_best | 3.3 | — | — | Untested | not targeted in plan |
| conv_all_above_original_pm2 | 2 | — | — | Untested | not targeted in plan |
| conv_dropout_initial_gain_conv2 | 2.1 | — | — | Untested | not targeted in plan |
| conv_dropout_initial_gain_conv4 | 3 | — | — | Untested | not targeted in plan |
| conv_dropout_initial_gain_conv6 | 2.4 | — | — | Untested | not targeted in plan |
| resnet18_lr001_best_ticket_acc | 89.5 | — | — | Untested | not targeted in plan |
| resnet18_lr01_unpruned_acc | 90.5 | — | — | Untested | not targeted in plan |
| resnet18_warmup_ticket_acc | 90.5 | — | — | Untested | not targeted in plan |
| resnet18_warmup_ticket_pm_threshold | 11.8 | — | — | Untested | not targeted in plan |
| vgg19_lr001_within_1pp_pm | 3.5 | — | — | Untested | not targeted in plan |
| vgg19_warmup_ticket_pm | 1.5 | — | — | Untested | not targeted in plan |
| vgg19_layerwise_vs_global_pm | 6.9 | — | — | Untested | not targeted in plan |
| conv4_val_acc_best_lr0003 | 78.6 | — | — | Untested | not targeted in plan |
| conv6_val_acc_best_lr0003 | 81.5 | — | — | Untested | not targeted in plan |
| conv4_dropout_val_acc | 82.6 | — | — | Untested | not targeted in plan |
| conv6_dropout_val_acc | 84.8 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | constant | paper_text | medium |
| augmentation | none | paper_text | low |
| batch_vs_steps | steps | paper_text | high |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | paper_text | medium |
| pruning_scope | layerwise | paper_text | high |
| output_layer_pruning_rate | half_of_base_rate_10pct | paper_text | medium |
| prune_biases | weights_only | conventions_kb | medium |
| iterative_strategy | resetting | paper_text | high |
| reinit_distribution | fresh_gaussian_glorot_per_layer | paper_text | high |
| n_seeds | 5_trials_3_reinits | paper_text | medium |
| early_stopping_criterion | min_validation_loss_every_100_iters | paper_text | medium |
| validation_split_seed | fixed_seed_0_random_sample | guess | low |
| accuracy_evaluation_point | both | paper_text | medium |
| train_iters_per_round | 50000_iterations_as_paper | paper_text | high |
| sparsity_levels | geometric_0.8_per_round_to_0.3pct | paper_text | medium |
| weight_init | gaussian_glorot | paper_text | medium |
| init_truncation | truncated_normal | conventions_kb | low |
| optimizer | adam_0.0012 | paper_text | medium |
| optimizer_state_reset | reset_each_round | guess | medium |
| input_normalization | scale_0_1 | conventions_kb | low |
| loss_function | softmax_cross_entropy | conventions_kb | low |
| bias_init | zeros | conventions_kb | low |
| pruning_rate_basis | fraction_of_surviving_weights | paper_text | high |
| pruning_weights_snapshot | final_iteration | paper_text | high |
| mask_application | frozen_zero_mask | paper_text | high |
| threshold_rule | last_grid_point_where_mean_ge_mean_unpruned | guess | high |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "8544bcf97afc63464e12b2e7d4664a9385bb379a9e8fbcb049d5f033856242e5",
 "frozen_at": "2026-09-15T16:16:02.560644+00:00",
 "env_lock_hash": "98fa794f4e123348",
 "runs": [
  "full_lenet_reinit_iterative_s0_94d265d16ac3b6e4.json",
  "full_lenet_reinit_iterative_s1_94d265d16ac3b6e4.json",
  "full_lenet_ticket_iterative_s0_d45fb5600be10777.json",
  "full_lenet_ticket_iterative_s1_d45fb5600be10777.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/lottery/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py"
}
```

Wall: 18.5 min · Human: 0 min · Cost: $2.24 · Stages: setup, build, run, verify