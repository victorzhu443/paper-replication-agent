# Replication report: Proximal Policy Optimization Algorithms

**Kind of test:** mechanics_only · **Data tier:** C · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** F  (data: synthetic; procedure: re-implemented, 0 unexplained; result: Untested; integrity: failed)

## 1. Deviations from the paper

- openai_gym_mujoco -> UNAVAILABLE: claims depending on this source are Untested
- ale_atari -> UNAVAILABLE: claims depending on this source are Untested
- roboschool -> UNAVAILABLE: claims depending on this source are Untested
- Table-1 checkpoint (measure): num_mujoco_environments paper=7.0 ours=None gap=None; seeds_per_environment paper=3.0 ours=None gap=None; runs_averaged_per_setting paper=21.0 ours=None gap=None; training_timesteps_per_mujoco_env paper=1000000.0 ours=None gap=None; scoring_episodes_window paper=100.0 ours=None gap=None; num_atari_games paper=49.0 ours=None gap=None; atari_frames paper=40000000.0 ours=None gap=None; atari_timesteps paper=10000000.0 ours=None gap=None; atari_trials_per_game paper=3.0 ours=None gap=None; policy_hidden_layers paper=2.0 ours=None gap=None; policy_hidden_units_per_layer paper=64.0 ours=None gap=None; horizon_T_mujoco paper=2048.0 ours=None gap=None; adam_stepsize_mujoco paper=0.0003 ours=None gap=None; num_epochs_mujoco paper=10.0 ours=None gap=None; minibatch_size_mujoco paper=64.0 ours=None gap=None; discount_gamma paper=0.99 ours=None gap=None; gae_lambda paper=0.95 ours=None gap=None; clip_epsilon_default paper=0.2 ours=None gap=None; kl_divergence_at_ppo_update paper=0.02 ours=None gap=None; kl_adaptation_threshold_factor paper=1.5 ours=None gap=None; kl_adaptation_beta_factor paper=2.0 ours=None gap=None; beta_initial_value paper=1.0 ours=None gap=None; horizon_T_atari paper=128.0 ours=None gap=None; num_epochs_atari paper=3.0 ours=None gap=None; minibatch_size_atari paper=256.0 ours=None gap=None; num_actors_atari paper=8.0 ours=None gap=None; vf_coeff_c1_atari paper=1.0 ours=None gap=None; entropy_coeff_c2_atari paper=0.01 ours=None gap=None; horizon_T_roboschool paper=512.0 ours=None gap=None; num_epochs_roboschool paper=15.0 ours=None gap=None; minibatch_size_roboschool paper=4096.0 ours=None gap=None; num_actors_roboschool_locomotion paper=32.0 ours=None gap=None; num_actors_roboschool_flagrun paper=128.0 ours=None gap=None; flagrun_target_reset_interval_timesteps paper=200.0 ours=None gap=None; table2_ties_alltraining paper=0.0 ours=None gap=None; table2_ties_last100 paper=1.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| t1_clip_eps02 | 0.82 | 1 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| t1_no_clip | -0.39 | 1 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| t1_clip_eps01 | 0.76 | 1 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| t1_clip_eps03 | 0.7 | — | — | Untested | no value produced |
| t1_adaptive_kl_0003 | 0.68 | — | — | Untested | no value produced |
| t1_adaptive_kl_001 | 0.74 | — | — | Untested | no value produced |
| t1_adaptive_kl_003 | 0.71 | — | — | Untested | no value produced |
| t1_fixed_kl_beta03 | 0.62 | — | — | Untested | no value produced |
| t1_fixed_kl_beta1 | 0.71 | — | — | Untested | not targeted in plan |
| t1_fixed_kl_beta3 | 0.72 | — | — | Untested | not targeted in plan |
| t1_fixed_kl_beta10 | 0.69 | — | — | Untested | not targeted in plan |
| t2_alltraining_ppo_wins | 30 | — | — | Untested | not targeted in plan |
| t2_alltraining_acer_wins | 18 | — | — | Untested | not targeted in plan |
| t2_alltraining_a2c_wins | 1 | — | — | Untested | not targeted in plan |
| t2_last100_ppo_wins | 19 | — | — | Untested | not targeted in plan |
| t2_last100_acer_wins | 28 | — | — | Untested | not targeted in plan |
| t2_last100_a2c_wins | 1 | — | — | Untested | not targeted in plan |
| t6_pong_ppo | 20.7 | — | — | Untested | not targeted in plan |
| t6_breakout_ppo | 274.8 | — | — | Untested | not targeted in plan |
| cartpole_solved_threshold | 475 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | FAIL | shuffled avg_normalized_score=1 (t=None), unshuffled=1 |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| env_suite | gym_cartpole_v1_200k_steps | guess | high |
| lr_schedule | constant | paper_text | medium |
| augmentation | none | conventions_kb | low |
| batch_vs_steps | steps | conventions_kb | high |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | paper_text | medium |
| eval_return_source | train_episode_returns_last_100 | paper_text | high |
| episode_buffer_rule | global_last_100_finished_episodes | guess | medium |
| seeds | 3 | paper_text | high |
| score_normalization_reference | max_over_all_settings_and_seeds_per_env | guess | high |
| advantage_normalization | per_batch | author_code | high |
| obs_normalization | none | guess | medium |
| entropy_coef | 0.0 | paper_text | medium |
| shared_trunk | separate | paper_text | medium |
| value_net_training | same_mlp_same_adam_same_epochs_weight1 | guess | medium |
| kl_estimator | analytic_per_state | paper_text | medium |
| max_grad_norm | none | author_code | medium |
| clip_vloss | none | paper_text | low |
| rollout_length_and_actors | T=2048,N=1 | paper_text | medium |
| ppo_epochs_minibatch | K=10,M=64 | paper_text | medium |
| total_timesteps | 200000 | guess | high |
| noclip_update_rule | same_K_epochs_ratio_objective | paper_text | high |
| terminal_bootstrap | bootstrap_on_truncation_zero_on_terminal | conventions_kb | high |
| policy_std_parameterization | state_independent_learned_logstd_init0 | conventions_kb | low |
| init_scheme | orthogonal_sqrt2_policy0.01 | conventions_kb | medium |
| adam_eps | 1e-5 | author_code | low |
| atari_wrappers | mnih2016_standard_deepmind_wrappers | paper_text | medium |
| atari_reward_clipping | clip_train_report_raw | conventions_kb | medium |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "f3ac1d57ceba896af33d21c8dd20e50b8d5bfa81119ab9dad6306602670fe721",
 "frozen_at": "2026-09-15T16:05:17.087494+00:00",
 "env_lock_hash": "98fa794f4e123348",
 "runs": [
  "full_clip_eps_0.1_s0_ac3417ec7231ab24.json",
  "full_clip_eps_0.1_s1_ac3417ec7231ab24.json",
  "full_clip_eps_0.1_s2_ac3417ec7231ab24.json",
  "full_clip_eps_0.2_s0_b312643af839db5c.json",
  "full_clip_eps_0.2_s1_b312643af839db5c.json",
  "full_clip_eps_0.2_s2_b312643af839db5c.json",
  "full_no_clip_no_penalty_s0_fdcfea8a72fd3fab.json",
  "full_no_clip_no_penalty_s1_fdcfea8a72fd3fab.json",
  "full_no_clip_no_penalty_s2_fdcfea8a72fd3fab.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/ppo/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py"
}
```

Wall: 26.0 min · Human: 0 min · Cost: $0.62 · Stages: setup, build, run, verify