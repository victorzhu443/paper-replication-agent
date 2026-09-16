# Replication report: Toy Models of Superposition

**Kind of test:** mechanics_only · **Data tier:** C · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: synthetic; procedure: re-implemented, 0 unexplained; result: Untested; integrity: not_run)

## 1. Deviations from the paper

- synthetic -> UNAVAILABLE: claims depending on this source are Untested
- Table-1 checkpoint (measure): n_features_main_experiment paper=20.0 ours=None gap=None; m_hidden_main_experiment paper=5.0 ours=None gap=None; importance_decay_base_main_experiment paper=0.7 ours=None gap=None; n_sparsity_levels_in_paper_figure paper=7.0 ours=None gap=None; n_sparsity_levels_requested_sweep paper=5.0 ours=None gap=None; n_features_second_experiment paper=80.0 ours=None gap=None; m_hidden_second_experiment paper=20.0 ours=None gap=None; importance_decay_base_second_experiment paper=0.9 ours=None gap=None; n_features_uniform_superposition paper=400.0 ours=None gap=None; m_hidden_uniform_superposition paper=30.0 ours=None gap=None; models_per_point_phase_diagram paper=10.0 ours=None gap=None; models_trained_relu_hidden_n10_m5 paper=1000.0 ours=None gap=None; n_features_abs_value_model paper=100.0 ours=None gap=None; m_neurons_abs_value_model paper=40.0 ours=None gap=None; baseline_density_perturbation_experiment paper=0.05 ours=None gap=None; adversarial_vulnerability_increase_factor_lower_bound paper=3.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| c_relu_dense_topm_n20 | 5 | 12.88 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_relu_sparse_all_features_n20 | 20 | 12.88 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_linear_topm_n20 | 5 | 12.88 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_linear_orthogonal_superposition_zero | 0 | 3.58e-05 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_linear_topm_n80 | 20 | — | — | Untested | no value produced |
| c_dstar_dense_one | 1 | — | — | Untested | no value produced |
| c_dstar_sticky_half | 0.5 | — | — | Untested | no value produced |
| c_dim_antipodal_pair | 0.5 | — | — | Untested | no value produced |
| c_dim_tetrahedron | 0.75 | — | — | Untested | not targeted in plan |
| c_dim_triangle | 0.6667 | — | — | Untested | not targeted in plan |
| c_dim_pentagon | 0.4 | — | — | Untested | not targeted in plan |
| c_dim_square_antiprism | 0.375 | — | — | Untested | not targeted in plan |
| c_pentagon_digon_threshold | 2.5 | — | — | Untested | not targeted in plan |
| c_relu_hidden_8_features_5_neurons | 8 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | not run | headline metric 'num_features_represented' has no known better-direction; shuffle test not run |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | constant | conventions_kb | medium |
| augmentation | none | paper_text | low |
| batch_vs_steps | steps | conventions_kb | medium |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | conventions_kb | low |
| optimizer | adam | conventions_kb | medium |
| learning_rate | 1e-3 | conventions_kb | high |
| training_steps | 10000 | guess | high |
| batch_size | 1024 | guess | medium |
| weight_init | xavier_normal | conventions_kb | medium |
| data_resampling | fresh_each_step | conventions_kb | medium |
| n_restarts | 10_best_loss | conventions_kb | high |
| feature_count_rule | frobenius_norm_sq | paper_text | high |
| importance_index_base | zero_based | conventions_kb | low |
| loss_reduction | sum_over_features_mean_over_batch | conventions_kb | medium |
| hidden_bias | output_bias_only | paper_text | low |
| bias_sign | plus_b | paper_text | low |
| sparsity_grid | task_five | paper_text | low |
| uniform_sparsity_grid | 40_log_spaced | guess | medium |
| seed | 0-9 | guess | high |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "35ad73ce53bcb35788a8b4d9ef4cb6de75f8291b644d73f01d595840dcb50eef",
 "frozen_at": "2026-09-15T16:23:06.516147+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [
  "full_linear_n20_m5_s0_fb8988f6b48796d6.json",
  "full_relu_output_n20_m5_dense_s0_257ead2d9b348af3.json",
  "full_relu_output_n20_m5_s099_s0_2ec926c0be65018b.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/superposition/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py"
}
```

Wall: 15.1 min · Human: 0 min · Cost: $2.76 · Stages: setup, build, run, verify