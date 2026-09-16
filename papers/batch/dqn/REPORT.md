# Replication report: Playing Atari with Deep Reinforcement Learning

**Kind of test:** conceptual_replication · **Data tier:** B · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier B gap measured; procedure: re-implemented, 0 unexplained; result: Untested; integrity: not_run)

## 1. Deviations from the paper

- arcade_learning_environment -> UNAVAILABLE: claims depending on this source are Untested
- human_expert_scores -> UNAVAILABLE: claims depending on this source are Untested
- published_baseline_scores -> UNAVAILABLE: claims depending on this source are Untested
- Table-1 checkpoint (measure): n_games_evaluated paper=7.0 ours=None gap=None; n_games_beating_prior_methods paper=6.0 ours=None gap=None; n_games_surpassing_human_expert paper=3.0 ours=None gap=None; training_frames_total paper=10000000.0 ours=None gap=None; replay_memory_frames paper=1000000.0 ours=None gap=None; minibatch_size paper=32.0 ours=None gap=None; epsilon_initial paper=1.0 ours=None gap=None; epsilon_final_train paper=0.1 ours=None gap=None; epsilon_anneal_frames paper=1000000.0 ours=None gap=None; epsilon_eval paper=0.05 ours=None gap=None; frame_skip_k_default paper=4.0 ours=None gap=None; frame_skip_k_space_invaders paper=3.0 ours=None gap=None; input_frames_stacked paper=4.0 ours=None gap=None; input_width_height paper=84.0 ours=None gap=None; downsampled_intermediate_height paper=110.0 ours=None gap=None; conv1_filters paper=16.0 ours=None gap=None; conv1_kernel paper=8.0 ours=None gap=None; conv1_stride paper=4.0 ours=None gap=None; conv2_filters paper=32.0 ours=None gap=None; conv2_kernel paper=4.0 ours=None gap=None; conv2_stride paper=2.0 ours=None gap=None; fc_hidden_units paper=256.0 ours=None gap=None; min_valid_actions paper=4.0 ours=None gap=None; max_valid_actions paper=18.0 ours=None gap=None; minibatch_updates_per_epoch paper=50000.0 ours=None gap=None; training_epochs_plotted paper=100.0 ours=None gap=None; eval_steps_per_curve_point paper=10000.0 ours=None gap=None; raw_frame_height paper=210.0 ours=None gap=None; raw_frame_width paper=160.0 ours=None gap=None; atari_color_palette_size paper=128.0 ours=None gap=None; atari_frame_rate_hz paper=60.0 ours=None gap=None; human_play_hours_per_game paper=2.0 ours=None gap=None; minutes_per_epoch_approx paper=30.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| c_dqn_breakout | 168 | 226.2 ± 74 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_dqn_seaquest | 1705 | 226.2 ± 74 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_dqn_pong | 20 | 226.2 ± 74 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_dqn_brider | 4092 | 226.2 ± 74 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_dqn_enduro | 470 | 226.2 ± 74 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_dqn_qbert | 1952 | 226.2 ± 74 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_dqn_sinvaders | 581 | 226.2 ± 74 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_rand_brider | 354 | 226.2 ± 74 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_rand_breakout | 1.2 | — | — | Untested | not targeted in plan |
| c_rand_enduro | 0 | — | — | Untested | not targeted in plan |
| c_rand_pong | -20.4 | — | — | Untested | not targeted in plan |
| c_rand_qbert | 157 | — | — | Untested | not targeted in plan |
| c_rand_seaquest | 110 | — | — | Untested | not targeted in plan |
| c_rand_sinvaders | 179 | — | — | Untested | not targeted in plan |
| c_sarsa_brider | 996 | — | — | Untested | not targeted in plan |
| c_sarsa_breakout | 5.2 | — | — | Untested | not targeted in plan |
| c_sarsa_enduro | 129 | — | — | Untested | not targeted in plan |
| c_sarsa_pong | -19 | — | — | Untested | not targeted in plan |
| c_sarsa_qbert | 614 | — | — | Untested | not targeted in plan |
| c_sarsa_seaquest | 665 | — | — | Untested | not targeted in plan |
| c_sarsa_sinvaders | 271 | — | — | Untested | not targeted in plan |
| c_cont_brider | 1743 | — | — | Untested | not targeted in plan |
| c_cont_breakout | 6 | — | — | Untested | not targeted in plan |
| c_cont_enduro | 159 | — | — | Untested | not targeted in plan |
| c_cont_pong | -17 | — | — | Untested | not targeted in plan |
| c_cont_qbert | 960 | — | — | Untested | not targeted in plan |
| c_cont_seaquest | 723 | — | — | Untested | not targeted in plan |
| c_cont_sinvaders | 268 | — | — | Untested | not targeted in plan |
| c_human_brider | 7456 | — | — | Untested | not targeted in plan |
| c_human_breakout | 31 | — | — | Untested | not targeted in plan |
| c_human_enduro | 368 | — | — | Untested | not targeted in plan |
| c_human_pong | -3 | — | — | Untested | not targeted in plan |
| c_human_qbert | 1.89e+04 | — | — | Untested | not targeted in plan |
| c_human_seaquest | 2.801e+04 | — | — | Untested | not targeted in plan |
| c_human_sinvaders | 3690 | — | — | Untested | not targeted in plan |
| c_hnbest_brider | 3616 | — | — | Untested | not targeted in plan |
| c_hnbest_breakout | 52 | — | — | Untested | not targeted in plan |
| c_hnbest_enduro | 106 | — | — | Untested | not targeted in plan |
| c_hnbest_pong | 19 | — | — | Untested | not targeted in plan |
| c_hnbest_qbert | 1800 | — | — | Untested | not targeted in plan |
| c_hnbest_seaquest | 920 | — | — | Untested | not targeted in plan |
| c_hnbest_sinvaders | 1720 | — | — | Untested | not targeted in plan |
| c_hnpix_brider | 1332 | — | — | Untested | not targeted in plan |
| c_hnpix_breakout | 4 | — | — | Untested | not targeted in plan |
| c_hnpix_enduro | 91 | — | — | Untested | not targeted in plan |
| c_hnpix_pong | -16 | — | — | Untested | not targeted in plan |
| c_hnpix_qbert | 1325 | — | — | Untested | not targeted in plan |
| c_hnpix_seaquest | 800 | — | — | Untested | not targeted in plan |
| c_hnpix_sinvaders | 1145 | — | — | Untested | not targeted in plan |
| c_dqnbest_brider | 5184 | — | — | Untested | not targeted in plan |
| c_dqnbest_breakout | 225 | — | — | Untested | not targeted in plan |
| c_dqnbest_enduro | 661 | — | — | Untested | not targeted in plan |
| c_dqnbest_pong | 21 | — | — | Untested | not targeted in plan |
| c_dqnbest_qbert | 4500 | — | — | Untested | not targeted in plan |
| c_dqnbest_seaquest | 1740 | — | — | Untested | not targeted in plan |
| c_dqnbest_sinvaders | 1075 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | not run | could not run: reproduce.sh failed (seed 0):
 |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | constant | conventions_kb | high |
| augmentation | none | paper_text | low |
| batch_vs_steps | steps | conventions_kb | medium |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | conventions_kb | low |
| gamma | 0.99 | conventions_kb | high |
| target_network | none_current_weights | paper_text | high |
| target_sync_interval | 500 | guess | medium |
| learning_starts | 1000 | conventions_kb | medium |
| replay_capacity | 50000 | conventions_kb | medium |
| replay_capacity_unit | agent_steps_transitions | paper_text | medium |
| frame_count_definition | agent_steps | guess | high |
| train_freq | 1_per_step | paper_text | medium |
| epsilon_schedule | linear_1_to_0.1_over_10pct_steps | paper_text | medium |
| loss_fn | mse | paper_text | medium |
| optimizer | rmsprop_0.95_decay | guess | medium |
| hidden_sizes | [128,128] | guess | low |
| eval_episodes | unspecified | paper_text | high |
| eval_checkpoint | final_network | guess | high |
| n_seeds | 3 | conventions_kb | high |
| eval_partial_episodes | exclude | guess | low |
| episodic_life | game_over_only | guess | medium |
| downsample_method | bilinear_bottom_crop | guess | medium |
| frame_stack_selection | last_4_acted_frames | guess | medium |
| random_baseline_protocol | same_steps_and_frameskip_as_dqn | guess | medium |
| weight_init | kaiming | conventions_kb | low |
| random_starts | none | paper_text | low |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "02e83d6b07a3c8d108e4ea27cbdbfa15f7557ef3336e679a3ad80f24b8dfe457",
 "frozen_at": "2026-09-15T16:05:23.100757+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [
  "full_dqn_avg_eps_greedy_k3_s0_a4be4b062aa0ac38.json",
  "full_dqn_avg_eps_greedy_s0_f95a880616082076.json",
  "full_random_policy_s0_30d406037cd3759f.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/dqn/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py --reverify batchnorm dqn layernorm lora ppo worldmodels"
}
```

Wall: 34.4 min · Human: 0 min · Cost: $0.00 · Stages: setup, build, run, verify