# Replication report: Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: Untested; integrity: passed)

## 1. Deviations from the paper

- Table-1 checkpoint (measure): imdb_prefix_count paper=25000.0 ours=None gap=None; completions_sampled_per_prefix paper=4.0 ours=None gap=None; preference_pairs_per_prefix paper=6.0 ours=None gap=None; total_sentiment_preference_pairs paper=150000.0 ours=None gap=None; imdb_prefix_length_min_tokens paper=2.0 ours=None gap=None; imdb_prefix_length_max_tokens paper=8.0 ours=None gap=None; anthropic_hh_dialogues paper=170000.0 ours=None gap=None; sentiment_sweep_runs paper=22.0 ours=None gap=None; eval_interval_steps paper=100.0 ours=None gap=None; beta_default paper=0.1 ours=None gap=None; beta_tldr paper=0.5 ours=None gap=None; batch_size paper=64.0 ours=None gap=None; learning_rate paper=1e-06 ours=None gap=None; warmup_steps paper=150.0 ours=None gap=None; sft_epochs_imdb paper=1.0 ours=None gap=None; reward_model_epochs_imdb paper=3.0 ours=None gap=None; ppo_batch_samples_per_step paper=1024.0 ours=None gap=None; best_of_n_plateau_dialogue paper=128.0 ours=None gap=None; human_raters paper=25.0 ours=None gap=None; judgments_per_rater paper=25.0 ours=None gap=None; dpo_vs_ppo0_comparisons paper=150.0 ours=None gap=None; ppo1_vs_ppo0_comparisons paper=100.0 ours=None gap=None; sft_vs_ppo0_comparisons paper=125.0 ours=None gap=None; dpo_ppo_judgments_collected paper=275.0 ours=None gap=None; ppo_ppo_judgments_collected paper=200.0 ours=None gap=None; n_respondents_dpo paper=272.0 ours=None gap=None; n_respondents_sft paper=122.0 ours=None gap=None; n_respondents_ppo1 paper=199.0 ours=None gap=None; human_tie_percentage paper=1.0 ours=None gap=None; max_model_size_billions_params paper=6.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| tldr_dpo_winrate_temp0 | 61 | 100 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| tldr_ppo_winrate_temp0 | 57 | — | — | Untested | no value produced |
| dpo_vs_ppo_human_headtohead | 58 | 100 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ood_dpo_temp0 | 0.36 | 100 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ood_dpo_temp025 | 0.31 | 100 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ood_ppo_temp0 | 0.26 | — | — | Untested | no value produced |
| ood_ppo_temp025 | 0.23 | — | — | Untested | no value produced |
| tbl2_gpt4S_win_dpo | 47 | — | — | Untested | no value produced |
| tbl2_gpt4C_win_dpo | 54 | — | — | Untested | not targeted in plan |
| tbl2_human_win_dpo | 58 | — | — | Untested | not targeted in plan |
| tbl2_gpt4S_win_sft | 27 | — | — | Untested | not targeted in plan |
| tbl2_gpt4C_win_sft | 32 | — | — | Untested | not targeted in plan |
| tbl2_human_win_sft | 43 | — | — | Untested | not targeted in plan |
| tbl2_gpt4S_win_ppo1 | 13 | — | — | Untested | not targeted in plan |
| tbl2_gpt4C_win_ppo1 | 12 | — | — | Untested | not targeted in plan |
| tbl2_human_win_ppo1 | 17 | — | — | Untested | not targeted in plan |
| tbl2_gpt4S_h_agree_dpo | 70 | — | — | Untested | not targeted in plan |
| tbl2_gpt4C_h_agree_dpo | 67 | — | — | Untested | not targeted in plan |
| tbl2_hh_agree_dpo | 65 | — | — | Untested | not targeted in plan |
| tbl2_gpt4S_h_agree_sft | 77 | — | — | Untested | not targeted in plan |
| tbl2_gpt4C_h_agree_sft | 79 | — | — | Untested | not targeted in plan |
| tbl2_gpt4S_h_agree_ppo1 | 86 | — | — | Untested | not targeted in plan |
| tbl2_gpt4C_h_agree_ppo1 | 85 | — | — | Untested | not targeted in plan |
| tbl2_hh_agree_ppo1 | 87 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | PASS | shuffled win_rate=50 (t=None), unshuffled=100 |
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
| batch_unit | pairs | conventions_kb | medium |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | paper_text | medium |
| logprob_aggregation | sum | conventions_kb | high |
| prompt_masking | completion_only | conventions_kb | high |
| reference_model | preferred_ft_on_chosen | paper_text | high |
| beta | 0.1 | paper_text | high |
| optimizer | rmsprop_1e-6_as_paper | paper_text | high |
| max_steps | 1000 | guess | medium |
| max_length | 128 | guess | medium |
| eval_temperature | 0.0 | paper_text | medium |
| decoding_params | pure_temperature_sampling | guess | medium |
| n_eval_prompts | 256 | guess | medium |
| gpt4_winrate_denominator | human_judgment_set | guess | medium |
| sft_stopping | one_epoch_subset | paper_text | medium |
| judge | ground_truth_attribute_classifier | conventions_kb | high |
| tie_handling | discard | paper_text | low |
| n_seeds | 3 | conventions_kb | medium |
| preference_rule | deterministic_attribute_rule | conventions_kb | high |
| grad_clip | none | guess | low |
| regularization | model_default | guess | low |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "23296a3175be78418de799f0991498bed6d6a352248e9824df946e82ecda1603",
 "frozen_at": "2026-09-15T16:14:39.787483+00:00",
 "env_lock_hash": "98fa794f4e123348",
 "runs": [
  "full_dpo_ood_cnndm_s0_d648c024ba0f6f8f.json",
  "full_dpo_tldr_s0_201f25a36e44068c.json",
  "full_human_eval_dpo_vs_ppo_s0_0149cd1dc607932f.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/dpo/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py"
}
```

Wall: 8.9 min · Human: 0 min · Cost: $2.69 · Stages: setup, build, run, verify