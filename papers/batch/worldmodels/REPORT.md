# Replication report: World Models

**Kind of test:** conceptual_replication · **Data tier:** B · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier B gap measured; procedure: re-implemented, 0 unexplained; result: Untested; integrity: passed)

## 1. Deviations from the paper

- openai_gym_carracing_v0 -> gym_env: simulator; Atari-scale runs are compute tier 2 on CPU
- vizdoom_takecover_v0 -> gym_env: simulator; Atari-scale runs are compute tier 2 on CPU
- random_policy_rollout_dataset -> UNAVAILABLE: claims depending on this source are Untested
- Table-1 checkpoint (measure): carracing_random_rollouts_collected paper=10000.0 ours=None gap=None; doom_random_rollouts_collected paper=10000.0 ours=None gap=None; carracing_latent_dim_Nz paper=32.0 ours=None gap=None; doom_latent_dim_Nz paper=64.0 ours=None gap=None; carracing_vae_param_count paper=4348547.0 ours=None gap=None; carracing_mdnrnn_param_count paper=422368.0 ours=None gap=None; carracing_controller_param_count paper=867.0 ours=None gap=None; doom_vae_param_count paper=4446915.0 ours=None gap=None; doom_mdnrnn_param_count paper=1678785.0 ours=None gap=None; doom_controller_param_count paper=1088.0 ours=None gap=None; carracing_lstm_hidden_units paper=256.0 ours=None gap=None; doom_lstm_hidden_units paper=512.0 ours=None gap=None; n_gaussian_mixtures paper=5.0 ours=None gap=None; vae_training_epochs paper=1.0 ours=None gap=None; mdnrnn_training_epochs paper=20.0 ours=None gap=None; cma_es_population_size paper=64.0 ours=None gap=None; rollouts_per_candidate_solution paper=16.0 ours=None gap=None; generations_to_solve_carracing paper=1800.0 ours=None gap=None; best_agent_eval_rollouts paper=1024.0 ours=None gap=None; best_agent_eval_interval_generations paper=25.0 ours=None gap=None; zonly_hidden_controller_param_count paper=1443.0 ours=None gap=None; zonly_hidden_units paper=40.0 ours=None gap=None; doom_max_timesteps_per_rollout paper=2100.0 ours=None gap=None; doom_solve_threshold_timesteps paper=750.0 ours=None gap=None; carracing_solve_threshold_score paper=900.0 ours=None gap=None; input_frame_resolution_pixels paper=64.0 ours=None gap=None; doom_done_probability_cutoff_percent paper=50.0 ours=None gap=None; doomrnn_training_temperature paper=1.15 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| carracing_full_world_model | 906 | 55.05 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| carracing_v_only | 632 | 55.05 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| carracing_v_only_hidden | 788 | 55.05 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| carracing_baseline_dqn | 343 | — | — | Untested | no value produced |
| carracing_baseline_a3c_continuous | 591 | — | — | Untested | no value produced |
| carracing_baseline_a3c_discrete | 652 | — | — | Untested | no value produced |
| carracing_baseline_gym_leader | 838 | — | — | Untested | no value produced |
| carracing_best_agent_1024_rollouts | 900.5 | 55.05 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| doom_transfer_actual_tau115 | 1092 | — | — | Untested | not targeted in plan |
| doom_virtual_tau115 | 918 | — | — | Untested | not targeted in plan |
| doom_virtual_tau010 | 2086 | — | — | Untested | not targeted in plan |
| doom_actual_tau010 | 193 | — | — | Untested | not targeted in plan |
| doom_virtual_tau050 | 2060 | — | — | Untested | not targeted in plan |
| doom_actual_tau050 | 196 | — | — | Untested | not targeted in plan |
| doom_virtual_tau100 | 1145 | — | — | Untested | not targeted in plan |
| doom_actual_tau100 | 868 | — | — | Untested | not targeted in plan |
| doom_virtual_tau130 | 732 | — | — | Untested | not targeted in plan |
| doom_actual_tau130 | 753 | — | — | Untested | not targeted in plan |
| doom_random_policy | 210 | — | — | Untested | not targeted in plan |
| doom_gym_leader | 820 | — | — | Untested | not targeted in plan |
| doomrnn_best_agent_virtual_1024 | 959 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | PASS | shuffled mean_return=30.94 (t=None), unshuffled=55.05, null reference=37.78 |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | constant | author_code | medium |
| augmentation | none | paper_text | low |
| batch_vs_steps | steps | conventions_kb | high |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | conventions_kb | medium |
| optimizer | adam_1e-4 | author_code | medium |
| batch_size | 100 | author_code | low |
| kl_weight | kl_tolerance_0.5_per_dim | author_code | high |
| recon_reduction | sum_over_pixels | conventions_kb | high |
| rnn_seq_len | 1000 | author_code | medium |
| temperature_carracing | not_applicable_deterministic_h | paper_text | low |
| cma_sigma_init | 0.1 | author_code | high |
| n_generations | 100 | conventions_kb | high |
| n_random_rollouts | 500 | conventions_kb | high |
| frame_skip | 1 | conventions_kb | medium |
| n_seeds | 3 | conventions_kb | medium |
| environment | cartpole_v1_pixels | guess | high |
| controller_state_input | h_only | paper_text | medium |
| controller_bias | with_bias | paper_text | low |
| n_eval_episodes | 100 | paper_text | low |
| mdnrnn_baseline | marginal_gaussian_fit_to_z | guess | medium |
| dispersion_measure | std_across_trials | conventions_kb | medium |
| baseline_n_trials | unknown_as_quoted | paper_text | low |
| n_virtual_rollouts | 100 | paper_text | medium |
| random_policy | uniform_over_action_space | guess | high |
| z_encode_mode | sample | paper_text | medium |
| action_mapping | clip_negative_to_zero | guess | medium |
| dream_init | random_real_frame_latent | guess | high |
| dream_max_steps | 2100 | paper_text | medium |
| fitness_seed_policy | resample_each_generation | guess | medium |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "04877b6a0eaa06140dc25601848cea20f8a8fe6bf6c9434100c619bee9b4d378",
 "frozen_at": "2026-09-16T17:46:02.722342+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [
  "full_carracing_full_zh_s0_cd7c8f65bfbc0d05.json",
  "full_carracing_z_only_hidden40_s0_ee00f18fdcd409c7.json",
  "full_carracing_z_only_linear_s0_096f31d16d38e88c.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/worldmodels/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py --reverify lora ppo worldmodels"
}
```

Wall: 24.0 min · Human: 0 min · Cost: $0.00 · Stages: setup, build, run, verify