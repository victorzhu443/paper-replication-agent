# Replication report: The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** not graded (infrastructure failure)  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; paper-scale run does not fit the compute envelope; result: no claims; integrity: not_run)

> **Run did not complete:** RuntimeError: build did not pass smoke: {'passed': False, 'problems': ['SCALE=0.1 probe failed or exceeded 10 min: reproduce.sh timed out after 600s (seed 0); process group killed. reproduce.sh must honor SCALE.'], 'metrics': {'lenet_earlystop_38pct_faster': 0.0, 'lenet_testacc_gain_at_pm135': 0.6099999999999852, 'lenet_testacc_gain_at_50k': 0.6099999999999852, 'lenet_speedup_vs_reinit_at_pm21': 1.0, 'lenet_pm_threshold_winning_ticket': 41.05259203606311, 'lenet_pm_threshold_reinit': 80.03756574004508, 'lenet_oneshot_faster_lower_bound': 90.01878287002253, 'conv4_testacc_gain_best': 1.2000000000000028, 'early_stop_iteration_reduction': 0.0, 'test_accuracy_delta_vs_unpruned': 0.6099999999999852, 'early_stop_speedup_ratio_vs_reinit': 1.0, 'pm_threshold_accuracy_drop': 41.05259203606311, 'pm_lower_bound_faster_early_stop': 90.01878287002253, 'lenet_test_accuracy_unpruned': 94.33500000000001, 'lenet_test_accuracy_best_ticket': 94.945, 'conv4_test_accuracy_unpruned': 35.45, 'chance_level': 10.0, 'test_accuracy_delta_vs_unpruned_baseline': 0.0, 'early_stop_iteration_reduction_baseline': 0.0, '_intermediates': {'scale': {'scale_arg': 1.0, 'smoke': True, 'lenet': 'Lenet-300-100 paper config except iterations: Adam 1.2e-3, batch 60, 300 iters/round (paper 50000), 4 rounds, 2 trials', 'conv4': 'SCALED DOWN: quarter width (16/16/32/32 channels), 120 iters/round (paper 25000), 2 rounds, 2 trials (paper 5), val/test eval subsets 500/1000', 'n_reinits': 1}, 'lenet_pm': [100.0, 80.03756574004508, 64.06386175807664, 51.28136739293764, 41.05259203606311], 'lenet_es_iter': [300.0, 300.0, 275.0, 300.0, 300.0], 'lenet_test_acc_es': [94.33500000000001, 94.435, 94.735, 94.83500000000001, 94.945], 'lenet_test_acc_final': [94.33500000000001, 94.435, 94.255, 94.83500000000001, 94.945], 'lenet_reinit_pm': [100.0, 80.03756574004508, 64.06386175807664, 51.28136739293764, 41.05259203606311], 'lenet_reinit_es': [300.0, 275.0, 300.0, 300.0, 300.0], 'lenet_reinit_acc': [94.22999999999999, 93.94, 93.27000000000001, 93.225, 92.75999999999999], 'lenet_oneshot_pm': [100.0, 90.01878287002253, 60.07513148009016, 30.131480090157776], 'lenet_oneshot_es': [300.0, 250.0, 300.0, 300.0], 'lenet_oneshot_acc': [94.33500000000001, 94.15, 94.795, 94.50999999999999], 'conv4_pm': [100.0, 80.3139533356105, 64.53384876113402], 'conv4_es_iter': [120.0, 120.0, 100.0], 'conv4_test_acc_es': [35.45, 36.6, 36.650000000000006], 'n_train_mnist': 55000, 'n_val': 5000, 'n_test_mnist': 10000, 'runtime_s': 53.837504863739014, 'chain_s': 11.558074951171875}}, 'shuffle_precheck': {'unshuffled': 0.0, 'shuffled': -75.0, 'null': 0.0}}

## 1. Deviations from the paper

- none recorded

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
 "spec_frozen_hash": "f66481c9ab4587999adb710345756cfe703168ebdfaee1f331e01594dfd2693e",
 "frozen_at": "2026-09-16T23:26:32.068740+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch_matched/lottery/work",
 "command": "-c"
}
```

Wall: 52.9 min · Human: 0 min · Cost: $11.58 · Stages: setup, build