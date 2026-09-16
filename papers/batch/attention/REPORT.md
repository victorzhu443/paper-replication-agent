# Replication report: Attention Is All You Need

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: Mismatch; integrity: passed)

## 1. Deviations from the paper

- Table-1 checkpoint (gate): wmt14_ende_training_sentence_pairs_millions paper=4.5 ours=None gap=None; wmt14_ende_shared_bpe_vocab_tokens paper=37000.0 ours=None gap=None; wmt14_enfr_training_sentences_millions paper=36.0 ours=None gap=None; wmt14_enfr_wordpiece_vocab_tokens paper=32000.0 ours=None gap=None; tokens_per_training_batch_source paper=25000.0 ours=None gap=None; tokens_per_training_batch_target paper=25000.0 ours=None gap=None; num_gpus paper=8.0 ours=None gap=None; base_step_time_seconds paper=0.4 ours=None gap=None; base_train_steps paper=100000.0 ours=None gap=None; base_train_hours paper=12.0 ours=None gap=None; big_step_time_seconds paper=1.0 ours=None gap=None; big_train_steps paper=300000.0 ours=None gap=None; big_train_days paper=3.5 ours=None gap=None; warmup_steps paper=4000.0 ours=None gap=None; adam_beta1 paper=0.9 ours=None gap=None; adam_beta2 paper=0.98 ours=None gap=None; adam_epsilon paper=1e-09 ours=None gap=None; base_params_millions paper=65.0 ours=None gap=None; big_params_millions paper=213.0 ours=None gap=None; parsing_wsj_training_sentences_thousands paper=40.0 ours=None gap=None; parsing_semisupervised_sentences_millions paper=17.0 ours=None gap=None; parsing_wsj_vocab_tokens_thousands paper=16.0 ours=None gap=None; parsing_semisup_vocab_tokens_thousands paper=32.0 ours=None gap=None; parsing_layers paper=4.0 ours=None gap=None; parsing_d_model paper=1024.0 ours=None gap=None; encoder_decoder_layers_base paper=6.0 ours=None gap=None; d_model_base paper=512.0 ours=None gap=None; d_ff_base paper=2048.0 ours=None gap=None; heads_base paper=8.0 ours=None gap=None; d_k_d_v_base paper=64.0 ours=None gap=None; dropout_base paper=0.1 ours=None gap=None; dropout_big_ende paper=0.3 ours=None gap=None; dropout_big_enfr paper=0.1 ours=None gap=None; label_smoothing paper=0.1 ours=None gap=None; beam_size_translation paper=4.0 ours=None gap=None; length_penalty_alpha_translation paper=0.6 ours=None gap=None; max_output_length_offset_translation paper=50.0 ours=None gap=None; checkpoints_averaged_base paper=5.0 ours=None gap=None; checkpoints_averaged_big paper=20.0 ours=None gap=None; beam_size_parsing paper=21.0 ours=None gap=None; length_penalty_alpha_parsing paper=0.3 ours=None gap=None; max_output_length_offset_parsing paper=300.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| c_t2_big_ende_bleu | 28.4 | 98.69 ± 0.22 | 1.05 | Mismatch | |gap|=70.29 > tol 1.05 |
| c_t2_big_enfr_bleu | 41.8 | 98.69 ± 0.22 | 1.05 | Mismatch | |gap|=56.89 > tol 1.05 |
| c_t2_base_ende_bleu | 27.3 | — | 1.05 | Untested | no value produced |
| c_t2_base_enfr_bleu | 38.1 | — | 1.05 | Untested | no value produced |
| c_t2_base_flops | 3.3e+18 | — | — | Untested | no value produced |
| c_t2_big_flops | 2.3e+19 | 1.448e+13 | — | Untested | no derived tolerance; needs n_periods, t-stat or std |
| c_t3_base_ppl_dev | 4.92 | — | 0.103 | Untested | no value produced |
| c_t3_base_bleu_dev | 25.8 | — | 1.05 | Untested | no value produced |
| c_t3_base_params | 65 | — | — | Untested | not targeted in plan |
| c_t3_big_ppl_dev | 4.33 | — | — | Untested | not targeted in plan |
| c_t3_big_bleu_dev | 26.4 | — | — | Untested | not targeted in plan |
| c_t3_big_params | 213 | — | — | Untested | not targeted in plan |
| c_t3_A_h1_ppl | 5.29 | — | — | Untested | not targeted in plan |
| c_t3_A_h1_bleu | 24.9 | — | — | Untested | not targeted in plan |
| c_t3_A_h16_bleu | 25.8 | — | — | Untested | not targeted in plan |
| c_text_single_head_delta | 0.9 | — | — | Untested | not targeted in plan |
| c_t3_C_N2_ppl | 6.11 | — | — | Untested | not targeted in plan |
| c_t3_C_N2_bleu | 23.7 | — | — | Untested | not targeted in plan |
| c_t3_D_drop0_ppl | 5.77 | — | — | Untested | not targeted in plan |
| c_t3_D_drop0_bleu | 24.6 | — | — | Untested | not targeted in plan |
| c_t3_E_learned_pe_ppl | 4.92 | — | — | Untested | not targeted in plan |
| c_t3_E_learned_pe_bleu | 25.7 | — | — | Untested | not targeted in plan |
| c_t4_parse_wsj_only_f1 | 91.3 | — | — | Untested | not targeted in plan |
| c_t4_parse_semisup_f1 | 92.7 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | PASS | shuffled bleu=0.04461 (t=None), unshuffled=98.69 |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | as_paper | paper_text | high |
| warmup_steps | 10_percent_of_total_steps | conventions_kb | high |
| attention_dropout | same_as_residual_p_drop | paper_text | medium |
| ppl_definition | unsmoothed_ce | paper_text | medium |
| length_penalty_form | gnmt_wu2016 | paper_text | medium |
| vocab_sharing_enfr | shared | guess | low |
| base_enfr_config | same_as_base_ende_100k_pdrop0.1 | paper_text | low |
| checkpoint_interval | 10_minutes_as_base | paper_text | low |
| augmentation | none | paper_text | low |
| batch_vs_steps | steps | conventions_kb | medium |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | paper_text | medium |
| reduced_task | reverse | guess | high |
| bleu_tool | sacrebleu_13a | conventions_kb | medium |
| baseline_arch | mean_pooled_context_ffn | guess | high |
| checkpoint_averaging | none | paper_text | low |
| beam_size | 1_greedy | conventions_kb | medium |
| init | xavier_uniform | conventions_kb | medium |
| grad_clip | none | conventions_kb | low |
| norm_placement | post_norm | paper_text | medium |
| label_smoothing | 0.1_as_paper | paper_text | medium |
| n_seeds | 3 | conventions_kb | medium |
| enfr_reference_value | 41.8_table2 | paper_text | low |
| claim_status | untested | conventions_kb | high |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "12e3f385c46de4225c6a250a2028616d2b1768dd5cebcb210eae8c16bb0f15c3",
 "frozen_at": "2026-09-15T15:53:29.208516+00:00",
 "env_lock_hash": "98fa794f4e123348",
 "runs": [
  "full_base_dev_newstest2013_s0_a8135110cffb1e52.json",
  "full_base_ende_s0_a82da3ae89942005.json",
  "full_base_enfr_s0_eb204e4c9c67ca8b.json",
  "full_big_ende_s0_9f197b1f566fa738.json",
  "full_big_enfr_s0_3a61557c0efadcef.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/attention/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py"
}
```

Wall: 83.7 min · Human: 0 min · Cost: $0.51 · Stages: setup, build, run, verify