# Replication report: Toy Models of Superposition

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier A checkpoint; procedure: re-implemented, 2 unexplained; result: Mismatch; integrity: not_run)

## 1. Deviations from the paper

- tolerance for c_dstar_sticky_half filled from our own SE (0.00191) at verify time, not frozen
- tolerance for c_dstar_dense_one filled from our own SE (0.00015) at verify time, not frozen
- tolerance for c_adversarial_vulnerability_3x filled from our own SE (0.0131) at verify time, not frozen
- tolerance for c_dim_tetrahedron filled from our own SE (1.83e-06) at verify time, not frozen
- tolerance for c_dim_triangle filled from our own SE (7.95e-08) at verify time, not frozen
- tolerance for c_dim_digon filled from our own SE (0) at verify time, not frozen
- tolerance for c_dim_pentagon filled from our own SE (1.58e-06) at verify time, not frozen
- tolerance for c_dim_square_antiprism filled from our own SE (4.19e-05) at verify time, not frozen
- Table-1 checkpoint (gate): n_features_uniform_geometry paper=400.0 ours=None gap=None; m_hidden_uniform_geometry paper=30.0 ours=None gap=None; importance_uniform_geometry paper=1.0 ours=None gap=None; n_features_basic_results paper=20.0 ours=None gap=None; m_hidden_basic_results paper=5.0 ours=None gap=None; importance_decay_basic_results paper=0.7 ours=None gap=None; n_features_larger_model paper=80.0 ours=None gap=None; m_hidden_larger_model paper=20.0 ours=None gap=None; importance_decay_larger_model paper=0.9 ours=None gap=None; models_per_phase_diagram_point paper=10.0 ours=None gap=None; n_features_pentagon_perturbation paper=5.0 ours=None gap=None; m_hidden_pentagon_perturbation paper=2.0 ours=None gap=None; baseline_feature_density_pentagon paper=0.05 ours=None gap=None; n_features_local_orthogonal_basis paper=20.0 ours=None gap=None; m_hidden_local_orthogonal_basis paper=10.0 ours=None gap=None; n_features_relu_hidden paper=10.0 ours=None gap=None; m_neurons_relu_hidden paper=5.0 ours=None gap=None; importance_decay_relu_hidden paper=0.75 ours=None gap=None; restarts_relu_hidden paper=1000.0 ours=None gap=None; n_features_abs_value_sweep paper=100.0 ours=None gap=None; m_neurons_abs_value_sweep paper=40.0 ours=None gap=None; importance_decay_abs_value_sweep paper=0.8 ours=None gap=None; n_features_abs_value_demo paper=3.0 ours=None gap=None; m_neurons_abs_value_demo paper=6.0 ours=None gap=None; adversarial_max_attack_norm_fraction paper=0.1 ours=None gap=None; n_correlated_geometry_features paper=6.0 ours=None gap=None; n_features_learning_dynamics_geometry paper=6.0 ours=None gap=None; m_hidden_learning_dynamics_geometry paper=3.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| c_dstar_sticky_half | 0.5 | 0.505 ± 0.0033 | 0.00383 | Mismatch | |gap|=0.004994 > tol 0.003828 |
| c_dstar_dense_one | 1 | 1.001 ± 0.00026 | 0.005 | Match | |gap|=0.001178 <= tol 0.005 |
| c_adversarial_vulnerability_3x | 3 | 1.463 ± 0.023 | 0.0261 | Mismatch | |gap|=1.537 > tol 0.02611 |
| c_dim_tetrahedron | 0.75 | 0.75 ± 3.2e-06 | 0.00375 | Match | |gap|=3.378e-06 <= tol 0.00375 |
| c_dim_triangle | 0.6667 | 0.6667 ± 1.4e-07 | 0.00333 | Match | |gap|=1.391e-07 <= tol 0.003333 |
| c_dim_digon | 0.5 | 0.5 | 0.0025 | Match | |gap|=0 <= tol 0.0025 |
| c_dim_pentagon | 0.4 | 0.4 ± 2.7e-06 | 0.002 | Match | |gap|=2.666e-06 <= tol 0.002 |
| c_dim_square_antiprism | 0.375 | 0.3706 ± 7.3e-05 | 0.00187 | Mismatch | |gap|=0.004447 > tol 0.001875 |
| c_dim_antipodal_formula | 0.5 | — | — | Untested | not targeted in plan |
| c_pentagon_digon_threshold | 2.5 | — | — | Untested | not targeted in plan |
| c_adv_train_norm_to_kill_superposition | 0.8 | — | — | Untested | not targeted in plan |
| c_relu_hidden_all_monosemantic_boundary | 0.5 | — | — | Untested | not targeted in plan |
| c_relu_hidden_3feat_2neuron_low | 0.2 | — | — | Untested | not targeted in plan |
| c_relu_hidden_5feat_3neuron | 0.15 | — | — | Untested | not targeted in plan |
| c_relu_hidden_6feat_4neuron | 0.12 | — | — | Untested | not targeted in plan |
| c_relu_hidden_8feat_5neuron_low | 0.05 | — | — | Untested | not targeted in plan |
| c_relu_hidden_feature_pairs | 0.04 | — | — | Untested | not targeted in plan |
| c_corr_no_features_collapsed | 0.05 | — | — | Untested | not targeted in plan |
| c_corr_all_sets_collapsed_low | 0.25 | — | — | Untested | not targeted in plan |
| c_energy_level_jump_peak | 0.7071 | — | — | Untested | not targeted in plan |
| c_confused_feature_weight | 0.7071 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | not run | headline metric 'dimensions_per_feature' has no known better-direction; shuffle test not run |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | constant | conventions_kb | medium |
| augmentation | none | conventions_kb | low |
| batch_vs_steps | steps | conventions_kb | medium |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | conventions_kb | low |
| optimizer | adam | conventions_kb | medium |
| learning_rate | 1e-3 | conventions_kb | medium |
| training_steps | 10000 | guess | high |
| batch_size | 1024 | guess | medium |
| data_resampling | fresh_each_step | conventions_kb | medium |
| bias_sign | plus_b | paper_text | low |
| bias_init | zeros | conventions_kb | medium |
| weight_init | pytorch_default_kaiming_uniform | guess | high |
| n_restarts | as_paper_where_stated | paper_text | high |
| n3m2_restarts | same_as_n2m1_ten_drop_worst | guess | medium |
| phase_grid_resolution | 40x40 | guess | medium |
| phase_classification_thresholds | continuous_2d_colormap_no_thresholds | paper_text | high |
| pentagon_threshold_direction | relative_density_0.4x_varied_feature_sparser | paper_text | medium |
| adv_baseline_model | dense_end_model_of_same_sweep | guess | high |
| adv_attack_scale | 0.1_of_average_input_l2_norm | paper_text | medium |
| sparsity_grid | 40_log_spaced_1_to_10 | guess | medium |
| eval_samples | 100000 | conventions_kb | low |
| importance_indexing | zero_indexed | paper_text | low |
| dstar_aggregation | lowest_loss_of_restarts | conventions_kb | medium |
| energy_jump_sparsity | any_sparsity_where_all_features_become_digons | paper_text | high |
| n_seeds | 5 | guess | medium |

## 6. Unexplained

- c_adversarial_vulnerability_3x (secondary): |gap|=1.537 > tol 0.02611
- c_dim_square_antiprism (secondary): |gap|=0.004447 > tol 0.001875

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "fe9468f7daf146c61363bf33b3130137cf6c184f90e646b8e0274d561cae3b87",
 "frozen_at": "2026-09-16T23:26:32.122893+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [
  "full_adversarial_eval_uniform_s0_ab0ab3d73c17edde.json",
  "full_adversarial_eval_uniform_s1_ab0ab3d73c17edde.json",
  "full_adversarial_eval_uniform_s2_ab0ab3d73c17edde.json",
  "full_relu_output_uniform_s0_bc075f4771e35054.json",
  "full_relu_output_uniform_s1_bc075f4771e35054.json",
  "full_relu_output_uniform_s2_bc075f4771e35054.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch_matched/superposition/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py --reverify superposition"
}
```

Wall: 463.5 min · Human: 0 min · Cost: $0.00 · Stages: setup, build, run, verify