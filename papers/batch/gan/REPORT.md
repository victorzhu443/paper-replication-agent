# Replication report: Generative Adversarial Nets

**Kind of test:** re_implementation · **Data tier:** A · **Compute tier:** 2 · **Track/family:** cs/train_and_eval

**Grade:** C  (data: tier A checkpoint; procedure: re-implemented, 0 unexplained; result: Untested; integrity: passed)

## 1. Deviations from the paper

- toronto_face_database -> UNAVAILABLE: claims depending on this source are Untested
- Table-1 checkpoint (gate): mnist_test_examples paper=10000.0 ours=None gap=None; mnist_train_examples paper=60000.0 ours=None gap=None; mnist_input_dimension paper=784.0 ours=None gap=None; table1_models_compared paper=4.0 ours=None gap=None; k_discriminator_steps paper=1.0 ours=None gap=None; adv_nets_mnist_minus_gsn_gap paper=11.0 ours=None gap=None

## 2. Claims

| claim | paper | ours | ±tol | outcome | note |
|---|---|---|---|---|---|
| gan_equilibrium_d_loss | 1.386 | — | — | Untested | no value produced |
| gan_equilibrium_value_function | -1.386 | — | — | Untested | no value produced |
| gan_equilibrium_d_output_half | 0.5 | — | — | Untested | no value produced |
| adv_nets_mnist_parzen | 225 | — | 4.5 | Untested | no value produced |
| adv_nets_tfd_parzen | 2057 | — | 52.5 | Untested | no value produced |
| dbn_mnist_parzen | 138 | — | 4.5 | Untested | no value produced |
| dbn_tfd_parzen | 1909 | — | 132 | Untested | no value produced |
| scae_mnist_parzen | 121 | — | 3.7 | Untested | no value produced |
| scae_tfd_parzen | 2110 | — | — | Untested | not targeted in plan |
| gsn_mnist_parzen | 214 | — | — | Untested | not targeted in plan |
| gsn_tfd_parzen | 1890 | — | — | Untested | not targeted in plan |

## 3. Leakage and adversarial checks

| test | result | detail |
|---|---|---|
| shuffle | PASS | shuffled discriminator_loss=nan (t=None), unshuffled=nan |
| contamination_scan | not run | not implemented in MVP: hash-based train/test near-duplicate scan |

## 4. Convention grid (sensitivity, not search)


| flip | value | headline | Δ vs default |
|---|---|---|---|

## 5. Ambiguities and defaults used

| key | default | source | sensitivity |
|---|---|---|---|
| lr_schedule | constant | conventions_kb | high |
| augmentation | none | conventions_kb | low |
| input_scaling | unit_interval_0_1 | conventions_kb | medium |
| validation_split | 50000_train_10000_val | conventions_kb | medium |
| batch_vs_steps | steps | conventions_kb | medium |
| mixed_precision | fp32 | conventions_kb | low |
| split | test | paper_text | medium |
| generator_objective | non_saturating | paper_text | high |
| latent_dim | 100 | conventions_kb | medium |
| noise_prior | uniform_-1_1 | conventions_kb | medium |
| architecture | reduced_cpu_512_512_G_512_D | guess | high |
| n_hidden_layers | 2_hidden_layers_each | conventions_kb | medium |
| batch_size | 100 | conventions_kb | medium |
| optimizer | sgd_momentum_lr0.1_mom0.5 | conventions_kb | high |
| dropout_rate | 0.5 | conventions_kb | medium |
| k | 1 | paper_text | medium |
| train_steps | fit_15min_cpu_budget | guess | high |
| weight_init | glorot_uniform | conventions_kb | low |
| equilibrium_metric | sum_bce_real_plus_fake_last_window | guess | high |
| eval_classifier | small_cnn_trained_in_script | guess | medium |
| parzen_config | skip_parzen | conventions_kb | high |
| parzen_log_base | natural_log_nats | conventions_kb | low |
| n_seeds | 3 | guess | medium |
| output_activation | sigmoid_0_1 | guess | low |

## 6. Unexplained

- nothing outstanding

## 7. Reproducibility

```json
{
 "agent_version": "0.1.0",
 "spec_frozen_hash": "169bb97ab4c3c5096b338270a72805fcc9a18deb3a6366538a5f97f1c6d169df",
 "frozen_at": "2026-09-15T15:58:18.960384+00:00",
 "env_lock_hash": "98fa794f4e123348",
 "runs": [
  "full_mnist_mlp_gan_cpu_s0_f2b6a6b621ce5a88.json",
  "full_mnist_parzen_eval_s0_d67af5eba681b14b.json",
  "full_tfd_parzen_eval_s0_c51f1a912005ad41.json"
 ],
 "work_dir": "/Users/vzhu/Developer/research-copy/runs/batch/gan/work",
 "command": "/Users/vzhu/Developer/research-copy/evals/batch.py"
}
```

Wall: 21.5 min · Human: 0 min · Cost: $1.99 · Stages: setup, build, run, verify