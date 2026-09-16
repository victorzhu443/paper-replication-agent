# Replication report: Locating and Editing Factual Associations in GPT

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/checkpoint_eval

**Grade:** C  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: Untested; integrity: not_run)

## 1. Deviations from the paper

- Table-1 checkpoint (measure): counterfact_records_total paper=21919.0 ours=None gap=None; counterfact_subjects_total paper=20391.0 ours=None gap=None; counterfact_objects_total paper=749.0 ours=None gap=None; counterfact_counterfactual_statements_total paper=21595.0 ours=None gap=None; counterfact_paraphrase_prompts_total paper=42876.0 ours=None gap=None; counterfact_neighborhood_prompts_total paper=82650.0 ours=None gap=None; counterfact_generation_prompts_total paper=62346.0 ours=None gap=None; counterfact_records_per_relation paper=645.0 ours=None gap=None; counterfact_paraphrase_prompts_per_record paper=2.0 ours=None gap=None; counterfact_neighborhood_prompts_per_record paper=10.0 ours=None gap=None; counterfact_generation_prompts_per_record paper=3.0 ours=None gap=None; zsre_eval_records paper=10000.0 ours=None gap=None; counterfact_test_records_gpt2xl paper=7500.0 ours=None gap=None; counterfact_test_records_gptj paper=2000.0 ours=None gap=None; causal_tracing_prompts paper=1000.0 ours=None gap=None; causal_tracing_noise_repeats_per_prompt paper=10.0 ours=None gap=None; clean_mean_prob_pct paper=27.0 ours=None gap=None; corrupted_mean_prob_pct paper=8.47 ours=None gap=None; gpt2xl_layers paper=48.0 ours=None gap=None; gptneox_layers paper=44.0 ours=None gap=None; gptj_layers paper=28.0 ours=None gap=None; k_star_prefix_texts paper=20.0 ours=None gap=None; covariance_k_samples paper=100000.0 ours=None gap=None; hyperparam_sweep_subset_size paper=50.0 ours=None gap=None; hypernetwork_training_subset_size paper=10000.0 ours=None gap=None; human_eval_raters paper=15.0 ours=None gap=None; human_eval_judgments_per_criterion paper=150.0 ours=None gap=None; human_eval_facts paper=50.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| ct_ate_gpt2xl | 18.6 | 27.16 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ct_aie_state_peak | 8.7 | 22.25 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ct_aie_mlp_peak | 6.6 | 22.25 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ct_aie_attn_last_subject | 1.6 | 22.25 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ct_peak_layer_state | 15 | 12 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ct_peak_layer_mlp | 17 | 12 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ct_peak_layer_attn | 32 | 12 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ct_clean_prob | 27 | — | — | Untested | not targeted in plan |
| ct_corrupted_prob | 8.47 | — | — | Untested | not targeted in plan |
| ct_restore_max_state_last_subject | 19.5 | 25.3 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| ct_restore_layer15_state | 15 | — | — | Untested | not targeted in plan |
| ct_restore_max_mlp | 23.6 | — | — | Untested | not targeted in plan |
| ct_restore_mlp_layer17 | 15 | — | — | Untested | not targeted in plan |
| ct_restore_max_attn | 19.4 | — | — | Untested | not targeted in plan |
| ct_restore_attn_layer32 | 16.5 | — | — | Untested | not targeted in plan |
| zsre_rome_efficacy | 99.8 | — | — | Untested | not targeted in plan |
| zsre_rome_paraphrase | 88.1 | — | — | Untested | not targeted in plan |
| zsre_rome_specificity | 24.2 | — | — | Untested | not targeted in plan |
| zsre_unedited_efficacy | 22.2 | — | — | Untested | not targeted in plan |
| zsre_ftl_paraphrase | 47.2 | — | — | Untested | not targeted in plan |
| cf_rome_S_gpt2xl | 89.2 | — | — | Untested | not targeted in plan |
| cf_rome_ES_gpt2xl | 100 | — | — | Untested | not targeted in plan |
| cf_rome_EM_gpt2xl | 97.9 | — | — | Untested | not targeted in plan |
| cf_rome_PS_gpt2xl | 96.4 | — | — | Untested | not targeted in plan |
| cf_rome_PM_gpt2xl | 62.7 | — | — | Untested | not targeted in plan |
| cf_rome_NS_gpt2xl | 75.4 | — | — | Untested | not targeted in plan |
| cf_rome_NM_gpt2xl | 4.2 | — | — | Untested | not targeted in plan |
| cf_rome_GE_gpt2xl | 621.9 | — | — | Untested | not targeted in plan |
| cf_rome_RS_gpt2xl | 41.9 | — | — | Untested | not targeted in plan |
| cf_unedited_S_gpt2xl | 30.5 | — | — | Untested | not targeted in plan |
| cf_unedited_ES_gpt2xl | 22.2 | — | — | Untested | not targeted in plan |
| cf_unedited_NS_gpt2xl | 78.1 | — | — | Untested | not targeted in plan |
| cf_ft_S_gpt2xl | 65.1 | — | — | Untested | not targeted in plan |
| cf_ftl_S_gpt2xl | 66.9 | — | — | Untested | not targeted in plan |
| cf_kn_S_gpt2xl | 35.6 | — | — | Untested | not targeted in plan |
| cf_ke_S_gpt2xl | 52.2 | — | — | Untested | not targeted in plan |
| cf_mend_S_gpt2xl | 57.9 | — | — | Untested | not targeted in plan |
| cf_rome_S_gptj | 91.5 | — | — | Untested | not targeted in plan |
| cf_rome_PS_gptj | 99.1 | — | — | Untested | not targeted in plan |
| cf_rome_NS_gptj | 78.9 | — | — | Untested | not targeted in plan |
| cf_rome_S_gpt2m | 87.4 | — | — | Untested | not targeted in plan |
| cf_rome_ES_gpt2m | 100 | — | — | Untested | not targeted in plan |
| cf_rome_EM_gpt2m | 94.9 | — | — | Untested | not targeted in plan |
| cf_rome_PS_gpt2m | 96.4 | — | — | Untested | not targeted in plan |
| cf_rome_NS_gpt2m | 71.8 | — | — | Untested | not targeted in plan |
| cf_unedited_S_gpt2m | 33.4 | — | — | Untested | not targeted in plan |
| cf_unedited_ES_gpt2m | 25 | — | — | Untested | not targeted in plan |
| cf_rome_S_gpt2l | 88.2 | — | — | Untested | not targeted in plan |
| zsre_rome_efficacy_gpt2m | 96.6 | — | — | Untested | not targeted in plan |
| zsre_rome_paraphrase_gpt2m | 79.8 | — | — | Untested | not targeted in plan |
| zsre_rome_specificity_gpt2m | 21.3 | — | — | Untested | not targeted in plan |
| zsre_unedited_efficacy_gpt2m | 18.8 | — | — | Untested | not targeted in plan |
| zsre_rome_efficacy_gpt2l | 99.6 | — | — | Untested | not targeted in plan |
| rome_peak_layer_gpt2xl | 18 | — | — | Untested | not targeted in plan |
| rome_ablation_noprefix_S | 86.1 | — | — | Untested | not targeted in plan |
| rome_ablation_longprefix_S | 89.3 | — | — | Untested | not targeted in plan |
| human_consistency_rome_vs_gpt | 125 | — | — | Untested | not targeted in plan |
| human_consistency_rome_vs_ftl | 97 | — | — | Untested | not targeted in plan |
| human_fluency_rome_vs_gpt | 61 | — | — | Untested | not targeted in plan |
| human_fluency_rome_vs_ftl | 64 | — | — | Untested | not targeted in plan |
| human_consistency_ratio_rome_ftl | 1.8 | — | — | Untested | not targeted in plan |
| human_fluency_ratio_rome_ftl | 1.3 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | not run | headline metric 'average_indirect_effect' has no known better-direction; shuffle test not run |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| eval.split | test | paper_text | medium |
| eval.averaging | macro | conventions_kb | medium |
| eval.preprocessing | as_repo | conventions_kb | high |
| model.checkpoint_version | paper_date | conventions_kb | low |
| model.layer_index_base | zero_indexed | paper_text | high |
| rome.edit_layer | scaled_mid_layer (about 0.37*L, e.g. 8 of 24 for gpt2-medium) | guess | high |
| rome.edit_layer_gptj | model_specific_mid_layer_from_gptj_causal_trace (released hparams: layer 5 of 28) | author_code | high |
| rome.k_star_prefixes | 20 texts: ten of length 5 and ten of length 10 (Appendix E.5) | paper_text | medium |
| rome.cov_source | released_precomputed_stats | conventions_kb | high |
| eval.target_prob | mean_per_token_prob | guess | high |
| tracing.target_prob | first_token_prob | guess | high |
| tracing.noise_parameterization | nu_is_std (sigma = 3*sigma_t) | paper_text | high |
| tracing.noise_nu | 3*sigma_t measured on the same model's embeddings | paper_text | high |
| tracing.n_noise_samples_seed | 10_samples_seed0 | paper_text | medium |
| tracing.n_prompts | 10 (CPU budget) | paper_text | high |
| tracing.subject_span_rule | substring_char_span_to_token_span_with_leading_space | guess | high |
| tracing.bucket_aggregation | average within bucket per prompt then average across prompts | guess | medium |
| eval.restore_weights_between_records | restore_after_each_record | paper_text | high |
| data.counterfact_subset_rule | first_n_by_case_id | guess | medium |
| eval.zsre_match_rule | first_answer_token_argmax | guess | medium |
| eval.generation_decoding | as_repo (top-k sampling, repo defaults) | conventions_kb | medium |
| eval.ge_scale | times_100 (to match Table 4 values near 620) | paper_text | high |
| eval.ge_weights | as_repo (n=[2,3], weights [2/3, 4/3], arithmetic mean over generations) | author_code | medium |
| eval.rs_tfidf | released_precomputed_tfidf_vectorizer | conventions_kb | medium |
| eval.reference_texts | released_reference_texts | conventions_kb | low |
| runtime.dtype | float32_cpu | guess | low |
| split | test | conventions_kb | high |
| averaging | macro | conventions_kb | medium |
| preprocessing | as_repo | conventions_kb | high |
| checkpoint_version | paper_date | conventions_kb | high |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "d6c6222464747236ecffa81bc0f36a7eda295d30cdf4050f96351e4cd8ea21c0",
 "frozen_at": "2026-09-15T16:35:12.385943+00:00",
 "env_lock_hash": "b4796a207a0b6fde",
 "runs": [
  "full_ct_attn_window10_s0_e512003b74187b65.json",
  "full_ct_mlp_window10_s0_385575fe4d870308.json",
  "full_ct_single_state_s0_673ce02c47a23fb1.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/rome/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py"
}
```

Wall: 16.4 min · Human: 0 min · Cost: $6.28 · Stages: setup, build, run, verify