import json, os, time, sys
import numpy as np
import torch
from multiprocessing import get_context
import lth

torch.set_num_threads(1)


def main():
    SEED = int(os.environ.get("SEED", "0"))
    SMOKE = os.environ.get("SMOKE", "0") == "1"
    SCALE = float(os.environ.get("SCALE", "1.0"))
    OUT = os.environ.get("METRICS_OUT", "metrics.json")
    CFG = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}")
    SHUFFLE = bool(CFG.get("_shuffle_labels", False))
    NPROC = int(CFG.get("n_procs", min(12, os.cpu_count() or 4)))

    # ---- ambiguity knobs (defaults = spec defaults)
    cfg = dict(
        lr_schedule=CFG.get("lr_schedule", "as_paper"),
        augmentation=CFG.get("augmentation", "as_paper"),       # none for MNIST/Conv-2/4/6
        batch_vs_steps=CFG.get("batch_vs_steps", "steps"),
        mixed_precision=CFG.get("mixed_precision", "fp32"),
        split=CFG.get("split", "test"),
        prune_parameter_types=CFG.get("prune_parameter_types", "weights_only"),
        n_pruning_rounds=CFG.get("n_pruning_rounds", "per_arch_axis_limits"),
        n_trials=int(CFG.get("n_trials", 5)),
        val_split_seed=CFG.get("val_split_seed", "fixed_random_seed_shared"),
        adam_betas_eps=CFG.get("adam_betas_eps", "tf_defaults_0.9_0.999_1e-8"),
        dropout_placement=CFG.get("dropout_placement", "after_each_maxpool_and_fc"),
        conv_padding=CFG.get("conv_padding", "same"),
        iterative_strategy=CFG.get("iterative_strategy", "reset_to_theta0_each_round"),
        output_layer_prune_rate=CFG.get("output_layer_prune_rate", "half_of_other_layers"),
        early_stop_criterion=CFG.get("early_stop_criterion", "min_validation_loss_every_100_iters"),
        loss=CFG.get("loss", "softmax_cross_entropy"),
        pm_denominator=CFG.get("pm_denominator", "all_weight_tensors_incl_unpruned_layers"),
        oneshot_pm_grid=CFG.get("oneshot_pm_grid", "single_trained_run_many_thresholds"),
        random_sparsity_allocation=CFG.get("random_sparsity_allocation",
                                           "match_per_layer_counts_of_winning_ticket"),
        reinit_glorot_scale=CFG.get("reinit_glorot_scale", "original_dense_layer_dims"),
        glorot_variant=CFG.get("glorot_variant", "normal_std_sqrt2_over_fanin_plus_fanout"),
        warmup_interaction=CFG.get("warmup_interaction", "warmup_then_original_step_decay_every_round"),
        speedup_aggregation=CFG.get("speedup_aggregation", "ratio_of_trial_averages"),
        input_normalization=CFG.get("input_normalization", "scale_0_1"),
        val_split_seed_value=int(CFG.get("val_split_seed_value", 1234)),
    )

    S = SCALE
    n_trials = 2 if SMOKE else cfg["n_trials"]
    n_conv_trials = 2 if SMOKE else int(CFG.get("n_conv_trials", 3))
    n_reinit = 1 if SMOKE else int(CFG.get("n_reinits", 3))
    oneshot_keep = ([0.9, 0.6, 0.3] if SMOKE else
                    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.125, 0.07, 0.04])

    t0 = time.time()
    ctx = get_context("spawn")

    def pmap(fn, tasks):
        if len(tasks) == 1 or NPROC <= 1:
            return [fn(t) for t in tasks]
        with ctx.Pool(min(NPROC, len(tasks))) as p:
            return p.map(fn, tasks)

    # ---------------- LeNet iterative pruning chains (one per trial)
    chains = pmap(lth.lenet_chain, [(t + SEED * 100, cfg, S, SMOKE, SHUFFLE) for t in range(n_trials)])
    t_chain = time.time() - t0
    print("lenet chains %.1fs" % t_chain, flush=True)
    rows_by_trial = [c["rows"] for c in chains]
    pms, es = lth.curve(rows_by_trial, "es_iter")
    _, acc_es = lth.curve(rows_by_trial, "test_acc_es")
    _, acc_fin = lth.curve(rows_by_trial, "test_acc_final")

    # ---------------- random reinit of the winning-ticket masks
    tasks = []
    for c in chains:
        for rnd, mask in enumerate(c["masks"]):
            for r in range(n_reinit):
                tasks.append((c["trial"], rnd, mask, cfg, S, SMOKE, SHUFFLE,
                              c["trial"] * 1000 + rnd * 10 + r))
    reinit = pmap(lth.lenet_reinit, tasks)
    print("reinit done %.1fs" % (time.time() - t0), flush=True)
    n_rounds = len(chains[0]["masks"])
    re_rows = []
    for rnd in range(n_rounds):
        sub = [x for x in reinit if x["round"] == rnd]
        re_rows.append(dict(pm=float(np.mean([x["pm"] for x in sub])),
                            es_iter=float(np.mean([x["es_iter"] for x in sub])),
                            test_acc_es=float(np.mean([x["test_acc_es"] for x in sub]))))
    re_pms = [r["pm"] for r in re_rows]
    re_es = [r["es_iter"] for r in re_rows]
    re_acc = [r["test_acc_es"] for r in re_rows]

    # ---------------- one-shot pruning (single trained dense run per trial, many thresholds)
    os_tasks = [(c["trial"], k, c["theta0"], c["trained0"], cfg, S, SMOKE, SHUFFLE)
                for c in chains for k in oneshot_keep]
    osr = pmap(lth.lenet_oneshot, os_tasks)
    print("oneshot done %.1fs" % (time.time() - t0), flush=True)
    os_pms, os_es, os_acc = [], [], []
    for k in oneshot_keep:
        sub = [x for x in osr if x["keep"] == k]
        os_pms.append(float(np.mean([x["pm"] for x in sub])))
        os_es.append(float(np.mean([x["es_iter"] for x in sub])))
        os_acc.append(float(np.mean([x["test_acc_es"] for x in sub])))
    os_pms = [100.0] + os_pms
    os_es = [es[0]] + os_es
    os_acc = [acc_es[0]] + os_acc

    # ---------------- Conv-4 (scaled down) iterative pruning
    conv = pmap(lth.conv_chain, [(t + SEED * 100, cfg, S, SMOKE, SHUFFLE) for t in range(n_conv_trials)])
    print("conv done %.1fs" % (time.time() - t0), flush=True)
    c_rows = [c["rows"] for c in conv]
    c_pms, c_es = lth.curve(c_rows, "es_iter")
    _, c_acc = lth.curve(c_rows, "test_acc_es")

    # ---------------- claim values
    i21 = lth.nearest(pms, 21.0)
    i135 = lth.nearest(pms, 13.5)
    claims = {}
    claims["lenet_earlystop_38pct_faster"] = 100.0 * (1.0 - es[i21] / es[0])
    claims["lenet_testacc_gain_at_pm135"] = acc_es[i135] - acc_es[0]
    claims["lenet_testacc_gain_at_50k"] = float(max(a - acc_fin[0] for a in acc_fin))
    claims["lenet_speedup_vs_reinit_at_pm21"] = re_es[i21] / es[i21]
    claims["lenet_pm_threshold_winning_ticket"] = lth.drop_threshold(pms, acc_es)
    claims["lenet_pm_threshold_reinit"] = lth.drop_threshold(re_pms, re_acc)
    claims["lenet_oneshot_faster_lower_bound"] = lth.faster_lower_bound(os_pms, os_es)
    claims["conv4_testacc_gain_best"] = float(max(a - c_acc[0] for a in c_acc))

    metrics = dict(claims)
    metrics.update({
        "early_stop_iteration_reduction": claims["lenet_earlystop_38pct_faster"],
        "test_accuracy_delta_vs_unpruned": claims["lenet_testacc_gain_at_pm135"],
        "early_stop_speedup_ratio_vs_reinit": claims["lenet_speedup_vs_reinit_at_pm21"],
        "pm_threshold_accuracy_drop": claims["lenet_pm_threshold_winning_ticket"],
        "pm_lower_bound_faster_early_stop": claims["lenet_oneshot_faster_lower_bound"],
        "lenet_test_accuracy_unpruned": acc_es[0],
        "lenet_test_accuracy_best_ticket": float(max(acc_es)),
        "conv4_test_accuracy_unpruned": c_acc[0],
        "chance_level": 10.0,
        "test_accuracy_delta_vs_unpruned_baseline": 0.0,
        "early_stop_iteration_reduction_baseline": 0.0,
    })

    L = lth._lenet_cfgs(cfg, S, SMOKE)
    C = lth._conv_cfgs(cfg, S, SMOKE)
    inter = dict(
        scale=dict(
            scale_arg=SCALE, smoke=SMOKE,
            lenet=("Lenet-300-100 paper config except iterations: Adam 1.2e-3, batch 60, "
                   "%d iters/round (paper 50000), %d rounds, %d trials" % (L["iters"], L["rounds"], n_trials)),
            conv4=("SCALED DOWN: quarter width (16/16/32/32 channels), %d iters/round (paper 25000), "
                   "%d rounds, %d trials (paper 5), val/test eval subsets %s/%s"
                   % (C["iters"], C["rounds"], n_conv_trials, C["n_val"], C["n_test"])),
            n_reinits=n_reinit,
        ),
        lenet_pm=pms, lenet_es_iter=es, lenet_test_acc_es=acc_es, lenet_test_acc_final=acc_fin,
        lenet_reinit_pm=re_pms, lenet_reinit_es=re_es, lenet_reinit_acc=re_acc,
        lenet_oneshot_pm=os_pms, lenet_oneshot_es=os_es, lenet_oneshot_acc=os_acc,
        conv4_pm=c_pms, conv4_es_iter=c_es, conv4_test_acc_es=c_acc,
        n_train_mnist=55000, n_val=5000, n_test_mnist=10000,
        runtime_s=time.time() - t0, chain_s=t_chain,
    )
    metrics["_intermediates"] = inter

    out = dict(seed=SEED, split=cfg["split"], n_examples=55000, scale=SCALE, shuffled=SHUFFLE,
               matched_scale=False, metrics=metrics)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", OUT, "in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
