# Replication report: Deep Residual Learning for Image Recognition

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: Untested; integrity: not_run)

## 1. Deviations from the paper

- Table-1 checkpoint (gate): cifar10_train_images paper=50000.0 ours=None gap=None; cifar10_test_images paper=10000.0 ours=None gap=None; cifar10_num_classes paper=10.0 ours=None gap=None; cifar_resnet20_params_millions paper=0.27 ours=None gap=None; cifar_resnet32_params_millions paper=0.46 ours=None gap=None; cifar_resnet44_params_millions paper=0.66 ours=None gap=None; cifar_resnet56_params_millions paper=0.85 ours=None gap=None; cifar_resnet110_params_millions paper=1.7 ours=None gap=None; cifar_resnet1202_params_millions paper=19.4 ours=None gap=None; cifar_batch_size paper=128.0 ours=None gap=None; cifar_total_iterations paper=64000.0 ours=None gap=None; cifar_lr_decay_iter_1 paper=32000.0 ours=None gap=None; cifar_lr_decay_iter_2 paper=48000.0 ours=None gap=None; cifar_initial_lr paper=0.1 ours=None gap=None; weight_decay paper=0.0001 ours=None gap=None; momentum paper=0.9 ours=None gap=None; imagenet_batch_size paper=256.0 ours=None gap=None; imagenet_max_iterations paper=600000.0 ours=None gap=None; imagenet_train_images_millions paper=1.28 ours=None gap=None; imagenet_val_images paper=50000.0 ours=None gap=None; imagenet_test_images paper=100000.0 ours=None gap=None; imagenet_num_classes paper=1000.0 ours=None gap=None; resnet18_flops_billions paper=1.8 ours=None gap=None; resnet34_flops_billions paper=3.6 ours=None gap=None; resnet50_flops_billions paper=3.8 ours=None gap=None; resnet101_flops_billions paper=7.6 ours=None gap=None; resnet152_flops_billions paper=11.3 ours=None gap=None; vgg16_flops_billions paper=15.3 ours=None gap=None; vgg19_flops_billions paper=19.6 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| cifar_resnet20_test_error | 8.75 | — | — | Untested | no value produced |
| cifar_resnet32_test_error | 7.51 | — | — | Untested | no value produced |
| cifar_resnet44_test_error | 7.17 | — | — | Untested | no value produced |
| cifar_resnet56_test_error | 6.97 | — | — | Untested | no value produced |
| cifar_resnet110_best_test_error | 6.43 | — | — | Untested | no value produced |
| cifar_resnet110_mean_test_error | 6.61 | — | 0.325 | Untested | no value produced |
| cifar_resnet1202_test_error | 7.93 | — | — | Untested | no value produced |
| cifar_plain110_test_error_lower_bound | 60 | — | — | Untested | not targeted in plan |
| cifar_resnet1202_train_error_upper_bound | 0.1 | — | — | Untested | not targeted in plan |
| imagenet_plain18_top1 | 27.94 | — | — | Untested | no value produced |
| imagenet_resnet18_top1 | 27.88 | — | — | Untested | not targeted in plan |
| imagenet_plain34_top1 | 28.54 | — | — | Untested | not targeted in plan |
| imagenet_resnet34A_top1 | 25.03 | — | — | Untested | not targeted in plan |
| imagenet_resnet34A_top5 | 7.76 | — | — | Untested | not targeted in plan |
| imagenet_plain34_top5 | 10.02 | — | — | Untested | not targeted in plan |
| imagenet_resnet152_single_model_top5_val | 4.49 | — | — | Untested | not targeted in plan |
| imagenet_ensemble_top5_test | 3.57 | — | — | Untested | not targeted in plan |

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
| lr_schedule | as_paper | paper_text | high |
| augmentation | as_paper | paper_text | medium |
| batch_vs_steps | steps | conventions_kb | high |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | paper_text | medium |
| train_split_size | full_50k | paper_text | medium |
| train_subset_selection | class_balanced_random_seeded | guess | high |
| n_seeds | 3 | conventions_kb | high |
| shortcut_option_a_impl | stride2_slice_then_zero_pad | conventions_kb | medium |
| shortcut_zero_pad_placement | append_end | guess | low |
| init | he_normal_fan_out | conventions_kb | low |
| conv_bias | no_bias | conventions_kb | low |
| input_normalization | per_pixel_mean_only | paper_text | medium |
| bn_config | framework_default | conventions_kb | low |
| lr_warmup | depth>=110_only | paper_text | low |
| plain_lr_warmup | no_warmup | guess | medium |
| plain110_error_metric | test_error | guess | low |
| checkpoint_selection | final | conventions_kb | medium |
| wd_scope | all_params | conventions_kb | low |
| optimizer_nesterov | classical | conventions_kb | low |
| determinism | seeded_deterministic | guess | medium |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "e9a7d9e5e52e05ad5fce2842feb0604756f9eb843b49771927675c06bbfb0b8c",
 "frozen_at": "2026-09-15T15:51:59.705377+00:00",
 "env_lock_hash": "98fa794f4e123348",
 "runs": [
  "full_cifar_resnet110_s0_edf0bc025e4e3717.json",
  "full_cifar_resnet20_s0_e88bb2bc3c5f5338.json",
  "full_cifar_resnet32_s0_e9ac0937cf3d5d9c.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/resnet/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py"
}
```

Wall: 152.9 min · Human: 0 min · Cost: $3.15 · Stages: setup, build, run, verify