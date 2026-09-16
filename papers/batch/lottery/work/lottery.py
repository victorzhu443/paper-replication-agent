"""Lottery Ticket Hypothesis: LeNet-300-100 on MNIST, iterative magnitude pruning.

Re-implementation from spec. CPU only. One seed (trial) per invocation.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

# ---------------------------------------------------------------- config

DEFAULTS = {
    "lr_schedule": "constant",
    "augmentation": "none",
    "batch_vs_steps": "steps",
    "mixed_precision": "fp32",
    "split": "test",
    "pruning_scope": "layerwise",
    "output_layer_pruning_rate": "half_of_base_rate_10pct",
    "prune_biases": "weights_only",
    "iterative_strategy": "resetting",
    "reinit_distribution": "fresh_gaussian_glorot_per_layer",
    "n_seeds": "5_trials_3_reinits",
    "early_stopping_criterion": "min_validation_loss_every_100_iters",
    "validation_split_seed": "fixed_seed_0_random_sample",
    "accuracy_evaluation_point": "both",
    "train_iters_per_round": "50000_iterations_as_paper",
    "sparsity_levels": "geometric_0.8_per_round_to_0.3pct",
    "weight_init": "gaussian_glorot",
    "init_truncation": "truncated_normal",
    "optimizer": "adam_0.0012",
    "optimizer_state_reset": "reset_each_round",
    "input_normalization": "scale_0_1",
    "loss_function": "softmax_cross_entropy",
    "bias_init": "zeros",
    "pruning_rate_basis": "fraction_of_surviving_weights",
    "pruning_weights_snapshot": "final_iteration",
    "mask_application": "frozen_zero_mask",
    "threshold_rule": "last_grid_point_where_mean_ge_mean_unpruned",
    # compute-tier knobs (deviations from paper, recorded in _intermediates.scale)
    "steps_per_round": 12000,
    "n_rounds": 18,
    "n_reinits": 2,
    "eval_interval": 200,
    "batch_size": 60,
    "base_prune_rate": 0.2,
}


def cfg_get(cfg, key):
    return cfg.get(key, DEFAULTS[key])


# ---------------------------------------------------------------- data

def load_mnist(cfg, n_train_cap=None):
    from torchvision import datasets
    tr = datasets.MNIST(DATA_ROOT, train=True, download=True)
    te = datasets.MNIST(DATA_ROOT, train=False, download=True)
    Xtr = tr.data.float()
    Xte = te.data.float()
    norm = cfg_get(cfg, "input_normalization")
    if norm == "scale_0_1":
        Xtr, Xte = Xtr / 255.0, Xte / 255.0
    elif norm == "mean_std_0.1307_0.3081":
        Xtr = (Xtr / 255.0 - 0.1307) / 0.3081
        Xte = (Xte / 255.0 - 0.1307) / 0.3081
    Xtr = Xtr.reshape(len(Xtr), -1)
    Xte = Xte.reshape(len(Xte), -1)
    Ytr, Yte = tr.targets.long(), te.targets.long()
    # validation split
    rule = cfg_get(cfg, "validation_split_seed")
    n = len(Xtr)
    if rule == "last_5000_of_train":
        idx = np.arange(n)
    else:
        g = np.random.default_rng(0 if rule == "fixed_seed_0_random_sample" else 1234)
        idx = g.permutation(n)
    val_idx, train_idx = idx[:5000], idx[5000:]
    Xva, Yva = Xtr[val_idx], Ytr[val_idx]
    Xtr2, Ytr2 = Xtr[train_idx], Ytr[train_idx]
    if n_train_cap is not None:
        Xtr2, Ytr2 = Xtr2[:n_train_cap], Ytr2[:n_train_cap]
        Xva, Yva = Xva[:1000], Yva[:1000]
        Xte, Yte = Xte[:2000], Yte[:2000]
    return (Xtr2, Ytr2), (Xva, Yva), (Xte, Yte)


# ---------------------------------------------------------------- model

LAYERS = [(784, 300), (300, 100), (100, 10)]


def glorot_init(shape, gen, truncated=True):
    fan_in, fan_out = shape[1], shape[0]
    std = math.sqrt(2.0 / (fan_in + fan_out))
    if truncated:
        w = torch.empty(shape)
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2 * std, b=2 * std, generator=gen)
        return w
    return torch.randn(shape, generator=gen) * std


class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(i, o) for i, o in LAYERS])

    def forward(self, x):
        for k, lin in enumerate(self.fc):
            x = lin(x)
            if k < len(self.fc) - 1:
                x = F.relu(x)
        return x


def sample_init(cfg, gen):
    trunc = cfg_get(cfg, "init_truncation") == "truncated_normal"
    kind = cfg_get(cfg, "weight_init")
    ws = []
    for i, o in LAYERS:
        if kind == "he_normal":
            w = torch.randn((o, i), generator=gen) * math.sqrt(2.0 / i)
        elif kind == "uniform_glorot":
            lim = math.sqrt(6.0 / (i + o))
            w = (torch.rand((o, i), generator=gen) * 2 - 1) * lim
        else:
            w = glorot_init((o, i), gen, trunc)
        ws.append(w)
    bs = [torch.zeros(o) if cfg_get(cfg, "bias_init") == "zeros"
          else glorot_init((o, 1), gen, trunc).flatten() for i, o in LAYERS]
    return ws, bs


# ---------------------------------------------------------------- training

def evaluate(model, X, Y, bs=5000):
    with torch.no_grad():
        loss, correct = 0.0, 0
        for i in range(0, len(X), bs):
            out = model(X[i:i + bs])
            loss += F.cross_entropy(out, Y[i:i + bs], reduction="sum").item()
            correct += (out.argmax(1) == Y[i:i + bs]).sum().item()
    return loss / len(X), correct / len(X)


def train_one(cfg, init, masks, data, steps, seed, shuffle_labels=False):
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = data
    ws, bs_ = init
    model = Lenet()
    with torch.no_grad():
        for k, lin in enumerate(model.fc):
            lin.weight.copy_(ws[k] * masks[k])
            lin.bias.copy_(bs_[k])
    lr = {"adam_0.0012": 1.2e-3, "adam_0.001": 1e-3, "sgd_0.8": 0.8}[cfg_get(cfg, "optimizer")]
    if cfg_get(cfg, "optimizer").startswith("adam"):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    sched = None
    if cfg_get(cfg, "lr_schedule") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    elif cfg_get(cfg, "lr_schedule") == "step":
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(steps * 0.5), int(steps * 0.75)], 0.1)

    batch = int(cfg_get(cfg, "batch_size"))
    n = len(Xtr)
    g = torch.Generator().manual_seed(seed * 7919 + 13)
    interval = int(cfg_get(cfg, "eval_interval"))
    perm = torch.randperm(n, generator=g)
    pos = 0
    hist = []  # (step, val_loss, val_acc, test_acc, train_acc)
    for step in range(1, steps + 1):
        if pos + batch > n:
            perm = torch.randperm(n, generator=g)
            pos = 0
        idx = perm[pos:pos + batch]
        pos += batch
        xb, yb = Xtr[idx], Ytr[idx]
        if shuffle_labels:
            yb = yb[torch.randperm(len(yb), generator=g)]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        with torch.no_grad():
            for k, lin in enumerate(model.fc):
                lin.weight.grad.mul_(masks[k])
        opt.step()
        with torch.no_grad():
            for k, lin in enumerate(model.fc):
                lin.weight.mul_(masks[k])
        if sched is not None:
            sched.step()
        if step % interval == 0 or step == steps:
            vl, va = evaluate(model, Xva, Yva)
            _, ta = evaluate(model, Xte, Yte)
            hist.append((step, vl, va, ta))
    # train accuracy at end (subsample for speed)
    _, tracc = evaluate(model, Xtr[:10000], Ytr[:10000])
    es = min(hist, key=lambda r: r[1])
    final = hist[-1]
    ws_final = [lin.weight.detach().clone() for lin in model.fc]
    return {
        "early_stop_iter": es[0],
        "val_loss_min": es[1],
        "val_acc_at_es": es[2],
        "test_acc_at_es": es[3],
        "test_acc_final": final[3],
        "val_acc_final": final[2],
        "train_acc_final": tracc,
    }, ws_final


# ---------------------------------------------------------------- pruning

def prune_masks(cfg, masks, weights):
    base = float(cfg_get(cfg, "base_prune_rate"))
    orule = cfg_get(cfg, "output_layer_pruning_rate")
    new = []
    if cfg_get(cfg, "pruning_scope") == "global":
        allw = torch.cat([(w.abs()[m.bool()]).flatten() for w, m in zip(weights, masks)])
        k = int(base * allw.numel())
        thr = allw.sort().values[k - 1] if k > 0 else -1
        for w, m in zip(weights, masks):
            new.append(((w.abs() > thr) & m.bool()).float())
        return new
    for li, (w, m) in enumerate(zip(weights, masks)):
        rate = base
        if li == len(masks) - 1:
            rate = {"half_of_base_rate_10pct": base / 2, "same_as_base_20pct": base,
                    "never_pruned": 0.0}[orule]
        surv = w.abs()[m.bool()].flatten()
        k = int(round(rate * surv.numel()))
        if k <= 0:
            new.append(m.clone())
            continue
        thr = surv.sort().values[k - 1]
        nm = ((w.abs() > thr) & m.bool()).float()
        new.append(nm)
    return new


def pm_percent(masks):
    tot = sum(m.numel() for m in masks)
    live = sum(int(m.sum().item()) for m in masks)
    return 100.0 * live / tot


# ---------------------------------------------------------------- thresholds

def threshold_pm(levels, accs, ref, rule):
    """Sparsest Pm (contiguous from dense) whose mean accuracy >= ref."""
    best = levels[0]
    for pm, a in zip(levels, accs):
        if a >= ref:
            best = pm
        else:
            if rule == "first_grid_point_where_mean_below_unpruned":
                return pm
            break
    return best


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    seed = int(os.environ.get("SEED", "0"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    scale = float(os.environ.get("SCALE", "1.0") or 1.0)
    cfg = json.loads(os.environ.get("REPLICATOR_CONFIG", "") or "{}")
    out_path = os.environ.get("METRICS_OUT", "metrics.json")
    shuffle = bool(cfg.get("_shuffle_labels", False))

    steps = int(cfg_get(cfg, "steps_per_round"))
    n_rounds = int(cfg_get(cfg, "n_rounds"))
    n_reinits = int(cfg_get(cfg, "n_reinits"))
    if cfg_get(cfg, "train_iters_per_round") == "2_epochs":
        steps = 2 * (55000 // int(cfg_get(cfg, "batch_size")))
    elif cfg_get(cfg, "train_iters_per_round") == "1_epoch":
        steps = 55000 // int(cfg_get(cfg, "batch_size"))
    cap = None
    if smoke:
        steps, n_rounds, n_reinits, cap = 300, 4, 1, 5000
        cfg = {**cfg, "eval_interval": 100}
    else:
        steps = max(200, int(round(steps * scale)))
    data = load_mnist(cfg, cap)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = data

    gen = torch.Generator().manual_seed(seed)
    init = sample_init(cfg, gen)
    masks = [torch.ones_like(w) for w in init[0]]

    levels, ticket, reinit = [], [], []
    for rnd in range(n_rounds + 1):
        pm = pm_percent(masks)
        res, wfinal = train_one(cfg, init, masks, data, steps, seed * 1000 + rnd, shuffle)
        levels.append(pm)
        ticket.append(res)
        rres = []
        for r in range(n_reinits if rnd > 0 else 0):
            g2 = torch.Generator().manual_seed(seed * 100000 + rnd * 100 + r + 1)
            rinit = sample_init(cfg, g2)
            if cfg_get(cfg, "reinit_distribution") == "shuffle_surviving_init_values":
                rinit = ([w.flatten()[torch.randperm(w.numel(), generator=g2)].reshape(w.shape)
                          for w in init[0]], init[1])
            rr, _ = train_one(cfg, rinit, masks, data, steps, seed * 1000 + rnd + 500, shuffle)
            rres.append(rr)
        reinit.append(rres)
        if rnd < n_rounds:
            snap = wfinal if cfg_get(cfg, "pruning_weights_snapshot") == "final_iteration" else wfinal
            if cfg_get(cfg, "iterative_strategy") == "continued_training":
                init = ([w.clone() for w in wfinal], init[1])
            masks = prune_masks(cfg, masks, snap)

    # ---- aggregate curves
    split = cfg_get(cfg, "split")
    key_es = "test_acc_at_es" if split == "test" else "val_acc_at_es"
    key_fin = "test_acc_final" if split == "test" else "val_acc_final"
    acc_es = [100 * r[key_es] for r in ticket]
    acc_fin = [100 * r[key_fin] for r in ticket]
    es_iter = [r["early_stop_iter"] for r in ticket]
    train_acc = [100 * r["train_acc_final"] for r in ticket]
    rn_acc_es, rn_acc_fin, rn_es_iter = [], [], []
    for rr in reinit:
        if rr:
            rn_acc_es.append(float(np.mean([100 * x[key_es] for x in rr])))
            rn_acc_fin.append(float(np.mean([100 * x[key_fin] for x in rr])))
            rn_es_iter.append(float(np.mean([x["early_stop_iter"] for x in rr])))
        else:
            rn_acc_es.append(acc_es[0])
            rn_acc_fin.append(acc_fin[0])
            rn_es_iter.append(float(es_iter[0]))

    rule = cfg_get(cfg, "threshold_rule")
    thr_ticket_acc = threshold_pm(levels, acc_es, acc_es[0], rule)
    thr_reinit_acc = threshold_pm(levels, rn_acc_es, acc_es[0], rule)
    thr_es = threshold_pm(levels, [-x for x in es_iter], -es_iter[0], rule)
    thr_train100 = threshold_pm(levels, train_acc, 99.95, rule)

    def nearest(target):
        return int(np.argmin([abs(l - target) for l in levels]))

    i21 = nearest(21.1)
    i13 = nearest(13.5)
    metrics = {
        "sparsity_threshold_matching_original_accuracy": thr_ticket_acc,
        "sparsity_threshold_matching_original_accuracy__lenet_reinit_iterative": thr_reinit_acc,
        "accuracy_improvement_over_unpruned": float(max(np.array(acc_fin) - acc_fin[0])),
        "accuracy_improvement_over_unpruned__early_stop": float(max(np.array(acc_es) - acc_es[0])),
        "accuracy_improvement_over_unpruned__early_stop_pm13_5": float(acc_es[i13] - acc_es[0]),
        "early_stop_iteration_reduction": 100.0 * (1.0 - es_iter[i21] / max(es_iter[0], 1)),
        "sparsity_threshold_matching_original_early_stop": thr_es,
        "early_stop_speedup_ratio_vs_reinit": float(rn_es_iter[i21] / max(es_iter[i21], 1)),
        "accuracy_gap_ticket_minus_reinit": float(acc_es[i21] - rn_acc_es[i21]),
        "sparsity_threshold_train_accuracy_100": thr_train100,
        "early_stop_speedup_ratio_vs_unpruned": float(max(
            es_iter[0] / max(e, 1) for e in es_iter)),
        "accuracy": acc_fin[0],
    }
    metrics["_intermediates"] = {
        "levels_pm_percent": [round(x, 3) for x in levels],
        "ticket_acc_at_early_stop": [round(x, 3) for x in acc_es],
        "ticket_acc_final": [round(x, 3) for x in acc_fin],
        "reinit_acc_at_early_stop": [round(x, 3) for x in rn_acc_es],
        "ticket_early_stop_iter": es_iter,
        "reinit_early_stop_iter": rn_es_iter,
        "ticket_train_acc_final": [round(x, 3) for x in train_acc],
        "unpruned_acc_final": acc_fin[0],
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "runtime_s": round(time.time() - t0, 1),
        "scale": {
            "steps_per_round": steps,
            "paper_steps_per_round": 50000,
            "epochs_equiv": round(steps * int(cfg_get(cfg, "batch_size")) / max(len(Xtr), 1), 2),
            "n_rounds": n_rounds,
            "n_reinits_per_level": n_reinits,
            "paper_reinits_per_level": 3,
            "trials_per_invocation": 1,
            "paper_trials": 5,
            "eval_interval": int(cfg_get(cfg, "eval_interval")),
            "paper_eval_interval": 100,
            "smoke": smoke,
            "scale_arg": scale,
        },
    }
    payload = {"seed": seed, "split": split, "n_examples": int(len(Xte)),
               "shuffled": shuffle, "metrics": metrics}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "_intermediates"}, indent=1))
    print("runtime", time.time() - t0)


if __name__ == "__main__":
    main()
