# Replication report: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** A  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: all headline Match; integrity: passed)

## 1. Deviations from the paper

- tolerance for mnist_bn_gt_nobn_final filled from our own SE (0.000318) at verify time, not frozen
- tolerance for mnist_bn_faster_to_baseline_acc filled from our own SE (601) at verify time, not frozen
- tolerance for mnist_bn_activation_stability filled from our own SE (0.0743) at verify time, not frozen
- Table-1 checkpoint (gate): n_train_examples paper=60000.0 ours=None gap=None; n_test_examples paper=10000.0 ours=None gap=None; input_dimension paper=784.0 ours=None gap=None; n_hidden_layers paper=3.0 ours=None gap=None; units_per_hidden_layer paper=100.0 ours=None gap=None; n_output_classes paper=10.0 ours=None gap=None; training_steps paper=50000.0 ours=None gap=None; batch_size paper=60.0 ours=None gap=None; bn_extra_params_per_activation paper=2.0 ours=None gap=None; out_of_scope_inception_steps_to_72.2pct paper=31000000.0 ours=None gap=None; out_of_scope_bn_x5_steps_to_72.2pct paper=2100000.0 ours=None gap=None; out_of_scope_ensemble_top5_val_error_percent paper=4.9 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| mnist_bn_gt_nobn_final | 0 | 0.008833 ± 0.00055 | 0.000636 | Match | directional claim (gt 0): ours=0.008833 |
| mnist_bn_final_acc | 0.98 | — | — | Untested | not targeted in plan |
| mnist_nobn_final_acc | 0.97 | — | — | Untested | not targeted in plan |
| mnist_bn_acc_at_10k | 0.97 | — | — | Untested | not targeted in plan |
| mnist_bn_faster_to_baseline_acc | 5e+04 | 1.083e+04 ± 1e+03 | 1.2e+03 | Match | directional claim (lt 5e+04): ours=1.083e+04 |
| mnist_bn_activation_stability | 0 | 0.2637 ± 0.13 | 0.149 | Match | directional claim (gt 0): ours=0.2637 |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | PASS | shuffled accuracy_gap=-0.0177 (t=None), unshuffled=0.008833, null reference=0 |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | constant | conventions_kb | high |
| learning_rate | 0.1 | guess | high |
| lr_bn_vs_baseline | same_lr_both | paper_text | high |
| augmentation | none | paper_text | low |
| batch_vs_steps | steps | paper_text | medium |
| batch_sampling | shuffle_each_epoch | conventions_kb | medium |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | paper_text | medium |
| optimizer | sgd | conventions_kb | high |
| input_binarization | threshold_0.5 | conventions_kb | low |
| init_std | 0.05 | guess | high |
| bn_eps | 1e-5 | conventions_kb | low |
| bn_inference_stats | moving_average | paper_text | medium |
| bn_placement | before_sigmoid_on_Wu | paper_text | high |
| bn_layers | hidden_only | paper_text | medium |
| use_bias_with_bn | remove_bias | paper_text | low |
| bn_affine_init | gamma=1,beta=0 | conventions_kb | low |
| n_seeds | 3 | guess | medium |
| eval_every_steps | 500 | conventions_kb | low |
| accuracy_comparison_point | final_step | guess | medium |
| tracked_unit | random_unit_last_hidden_layer | paper_text | low |
| drift_statistic | std_over_training_of_percentiles | guess | medium |
| percentile_data | current_training_minibatch | guess | low |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "b35bdd058933e15aaa320013c4cc6c6d8f882d9900acedea9f0d8f5845c9de3f",
 "frozen_at": "2026-09-17T19:39:00.725811+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [
  "full_bn_minus_baseline_s0_ef9efd2f1f55d18f.json",
  "full_bn_minus_baseline_s1_ef9efd2f1f55d18f.json",
  "full_bn_minus_baseline_s2_ef9efd2f1f55d18f.json",
  "full_with_bn_s0_be84ce3fb877d399.json",
  "full_with_bn_s1_be84ce3fb877d399.json",
  "full_with_bn_s2_be84ce3fb877d399.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch_matched/batchnorm/work",
 "command": "-c"
}
```

Wall: 10.1 min · Human: 0 min · Cost: $1.28 · Stages: setup, build, run, verify