# Replication report: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: Untested; integrity: not_run)

## 1. Deviations from the paper

- imagenet_ilsvrc2012 -> UNAVAILABLE: claims depending on this source are Untested
- tolerance for mnist_bn_beats_baseline_sign filled from our own SE (0.00135) at verify time, not frozen
- tolerance for inception_steps_to_72_2 filled from our own SE (5e+03) at verify time, not frozen
- tolerance for bn_baseline_steps_to_72_2 filled from our own SE (5e+03) at verify time, not frozen
- Table-1 checkpoint (gate): mnist_hidden_layers paper=3.0 ours=None gap=None; mnist_units_per_hidden_layer paper=100.0 ours=None gap=None; mnist_output_units paper=10.0 ours=None gap=None; mnist_input_dim paper=784.0 ours=None gap=None; mnist_training_steps paper=50000.0 ours=None gap=None; mnist_minibatch_size paper=60.0 ours=None gap=None; inception_parameters paper=13600000.0 ours=None gap=None; imagenet_minibatch_size paper=32.0 ours=None gap=None; imagenet_num_classes paper=1000.0 ours=None gap=None; imagenet_validation_images paper=50000.0 ours=None gap=None; imagenet_test_images paper=100000.0 ours=None gap=None; ensemble_num_models paper=6.0 ours=None gap=None; bn_x30_steps_to_74_8_percent paper=6000000.0 ours=None gap=None; bn_x5_speedup_factor_vs_inception paper=14.0 ours=None gap=None; shuffling_val_accuracy_gain_percent paper=1.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| mnist_bn_beats_baseline_sign | 0 | 0.00955 ± 0.0019 | 0.0027 | Untested | reduced scale (compute tier 2): ours=0.00955 is not comparable to the paper-scale value; |gap|=0.00955 > tol 0.0027 |
| inception_steps_to_72_2 | 3.1e+07 | 1.5e+04 ± 7.1e+03 | 6e+04 | Untested | reduced scale (compute tier 2): ours=1.5e+04 is not comparable to the paper-scale value; |gap|=3.098e+07 > tol 6e+04 |
| inception_max_acc | 72.2 | 0.9758 ± 0.0015 | 0.451 | Untested | reduced scale (compute tier 2): ours=0.9758 is not comparable to the paper-scale value; |gap|=71.22 > tol 0.4507 |
| bn_baseline_steps_to_72_2 | 1.33e+07 | 1.5e+04 ± 7.1e+03 | 6e+04 | Untested | reduced scale (compute tier 2): ours=1.5e+04 is not comparable to the paper-scale value; |gap|=1.328e+07 > tol 6e+04 |
| bn_baseline_max_acc | 72.7 | 0.9758 ± 0.0015 | 0.448 | Untested | reduced scale (compute tier 2): ours=0.9758 is not comparable to the paper-scale value; |gap|=71.72 > tol 0.4485 |
| bn_x5_steps_to_72_2 | 2.1e+06 | — | — | Untested | no value produced |
| bn_x5_max_acc | 73 | — | 0.447 | Untested | no value produced |
| bn_x30_steps_to_72_2 | 2.7e+06 | — | — | Untested | no value produced |
| bn_x30_max_acc | 74.8 | — | — | Untested | not targeted in plan |
| bn_x5_sigmoid_max_acc | 69.8 | — | — | Untested | not targeted in plan |
| bn_inception_single_crop_top1_err | 25.2 | — | — | Untested | not targeted in plan |
| bn_inception_single_crop_top5_err | 7.82 | — | — | Untested | not targeted in plan |
| bn_inception_multicrop_top1_err | 21.99 | — | — | Untested | not targeted in plan |
| bn_inception_multicrop_top5_err | 5.82 | — | — | Untested | not targeted in plan |
| bn_inception_ensemble_top1_err | 20.1 | — | — | Untested | not targeted in plan |
| bn_inception_ensemble_top5_err_val | 4.9 | — | — | Untested | not targeted in plan |
| bn_inception_ensemble_top5_err_test | 4.82 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | not run | pipeline did not acknowledge the label shuffle (_shuffled flag absent); test not run |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | constant | conventions_kb | high |
| learning_rate | 0.1 | guess | high |
| augmentation | none | conventions_kb | low |
| batch_vs_steps | steps | paper_text | medium |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | paper_text | medium |
| mnist_splits | train60000_test10000 | conventions_kb | low |
| optimizer | sgd | conventions_kb | high |
| init_std | 0.05 | guess | high |
| input_binarization | threshold_0.5 | conventions_kb | medium |
| bn_epsilon | 1e-5 | conventions_kb | low |
| bn_param_init | gamma1_beta0 | conventions_kb | medium |
| bn_inference_stats | moving_average | paper_text | medium |
| bn_momentum | 0.9 | guess | low |
| bn_layers | hidden_only | paper_text | medium |
| use_bias_with_bn | drop_bias | paper_text | low |
| weight_decay | 0.0 | conventions_kb | low |
| eval_steps | {10000,20000,30000,40000,50000} | guess | medium |
| n_seeds | 3 | conventions_kb | medium |
| loss | softmax_cross_entropy | paper_text | low |
| shuffle | shuffle_each_epoch | conventions_kb | low |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "a5a2b6e4c2d1c3c1098cd5ca9e1202d40683dbcab8837c10066476f242e64a01",
 "frozen_at": "2026-09-15T15:57:21.969457+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [
  "full_bn_baseline_s0_298e3ee1904fbd6a.json",
  "full_bn_baseline_s1_298e3ee1904fbd6a.json",
  "full_inception_s0_a93cd45caec315e9.json",
  "full_inception_s1_a93cd45caec315e9.json",
  "full_mnist_bn_vs_baseline_s0_848b8767dcda0a75.json",
  "full_mnist_bn_vs_baseline_s1_848b8767dcda0a75.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/batchnorm/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py --reverify batchnorm dqn layernorm lora ppo worldmodels"
}
```

Wall: 8.5 min · Human: 0 min · Cost: $0.00 · Stages: setup, build, run, verify