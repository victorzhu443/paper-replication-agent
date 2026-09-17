"""Lottery Ticket Hypothesis re-implementation (Frankle & Carbin 2019).

Iterative magnitude pruning with rewinding to theta_0 (Strategy 1), LeNet-300-100/MNIST
and Conv-4/CIFAR-10 (scaled down: half width, fewer iterations -- compute tier 2).
"""
from __future__ import annotations
import json, math, os, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import get_context

DATA = os.environ.get("LTH_DATA", "/Users/vzhu/Developer/research-copy/data_cache")

# ---------------------------------------------------------------- data
_cache = {}


def load_data(name, cfg, shuffle_labels=False, seed=0, train_sub=None):
    key = (name, shuffle_labels, seed, train_sub)
    if key in _cache:
        return _cache[key]
    if name == "mnist":
        z = np.load(os.path.join(DATA, "mnist", "mnist.npz"))
        Xtr, ytr, Xte, yte = z["Xtr"][:, None], z["ytr"], z["Xte"][:, None], z["yte"]
    else:
        z = np.load(os.path.join(DATA, "cifar10", "cifar10.npz"))
        Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    # fixed validation split shared across trials (ambiguity: val_split_seed)
    rng = np.random.RandomState(cfg.get("val_split_seed_value", 1234))
    perm = rng.permutation(len(Xtr))
    vidx, tidx = perm[:5000], perm[5000:]
    Xv, yv = Xtr[vidx], ytr[vidx]
    Xt, yt = Xtr[tidx], ytr[tidx]
    if train_sub is not None and train_sub < len(Xt):
        Xt, yt = Xt[:train_sub], yt[:train_sub]
    if shuffle_labels:
        # shuffle TRAINING labels only (leakage / no-skill control)
        yt = np.random.RandomState(seed + 999).permutation(yt)

    def prep(X):
        t = torch.from_numpy(np.ascontiguousarray(X)).float().div_(255.0)
        if cfg.get("input_normalization", "scale_0_1") == "per_channel_standardize":
            m = t.mean(dim=(0, 2, 3), keepdim=True)
            s = t.std(dim=(0, 2, 3), keepdim=True) + 1e-8
            t = (t - m) / s
        return t
    out = dict(Xtr=prep(Xt), ytr=torch.from_numpy(np.ascontiguousarray(yt)).long(),
               Xv=prep(Xv), yv=torch.from_numpy(np.ascontiguousarray(yv)).long(),
               Xte=prep(Xte), yte=torch.from_numpy(np.ascontiguousarray(yte)).long())
    _cache.clear()
    _cache[key] = out
    return out


# ---------------------------------------------------------------- models
class Net(nn.Module):
    """Sequential net exposing prunable weight tensors with a layer kind."""

    def __init__(self, arch, cfg, width=1.0, dropout=0.0):
        super().__init__()
        self.kinds = []
        layers = []
        pad = 1 if cfg.get("conv_padding", "same") == "same" else 0
        if arch == "lenet":
            layers += [nn.Flatten(), nn.Linear(784, 300), nn.ReLU(),
                       nn.Linear(300, 100), nn.ReLU(), nn.Linear(100, 10)]
            self.kinds = ["fc", "fc", "out"]
        else:
            chans = {"conv2": [[64, 64]], "conv4": [[64, 64], [128, 128]],
                     "conv6": [[64, 64], [128, 128], [256, 256]]}[arch]
            inc, side = 3, 32
            kinds = []
            for group in chans:
                for c in group:
                    c = max(4, int(round(c * width)))
                    layers += [nn.Conv2d(inc, c, 3, padding=pad), nn.ReLU()]
                    kinds.append("conv")
                    inc = c
                    if pad == 0:
                        side -= 2
                layers += [nn.MaxPool2d(2)]
                side //= 2
                if dropout:
                    layers += [nn.Dropout(dropout)]
            layers += [nn.Flatten(), nn.Linear(inc * side * side, 256), nn.ReLU()]
            kinds.append("fc")
            if dropout:
                layers += [nn.Dropout(dropout)]
            layers += [nn.Linear(256, 256), nn.ReLU()]
            kinds.append("fc")
            if dropout:
                layers += [nn.Dropout(dropout)]
            layers += [nn.Linear(256, 10)]
            kinds.append("out")
            self.kinds = kinds
        self.net = nn.Sequential(*layers)
        self.prunable = [m for m in self.net if isinstance(m, (nn.Linear, nn.Conv2d))]
        assert len(self.prunable) == len(self.kinds)

    def forward(self, x):
        return self.net(x)


def glorot_init(model, gen):
    for m in model.prunable:
        w = m.weight
        fan_in = w[0].numel()
        fan_out = w.shape[0] * (w[0, 0].numel() if w.dim() == 4 else 1)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        with torch.no_grad():
            w.copy_(torch.randn(w.shape, generator=gen) * std)
            if m.bias is not None:
                m.bias.zero_()


# ---------------------------------------------------------------- train/eval
def evaluate(model, X, y, bs=1000):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out = model(X[i:i + bs])
            loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
            correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return loss / len(X), 100.0 * correct / len(X)


def train_run(arch, cfg, data, theta0, masks, iters, lr, bs, eval_every, seed,
              width=1.0, dropout=0.0, n_val=None, n_test=None):
    """Train a (masked) net from theta0; return early-stop stats."""
    torch.manual_seed(seed)
    model = Net(arch, cfg, width=width, dropout=dropout)
    with torch.no_grad():
        for m, w0 in zip(model.prunable, theta0):
            m.weight.copy_(w0)
            if m.bias is not None:
                m.bias.zero_()
        for m, mk in zip(model.prunable, masks):
            m.weight.mul_(mk)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xv, yv = data["Xv"], data["yv"]
    Xte, yte = data["Xte"], data["yte"]
    if n_val:
        Xv, yv = Xv[:n_val], yv[:n_val]
    if n_test:
        Xte, yte = Xte[:n_test], yte[:n_test]
    g = torch.Generator().manual_seed(seed + 7)
    n = len(Xtr)
    perm = torch.randperm(n, generator=g)
    pos = 0
    vlosses, taccs, iters_at = [], [], []
    for it in range(1, iters + 1):
        if pos + bs > n:
            perm = torch.randperm(n, generator=g)
            pos = 0
        idx = perm[pos:pos + bs]
        pos += bs
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()
        with torch.no_grad():  # keep pruned weights at zero
            for m, mk in zip(model.prunable, masks):
                m.weight.mul_(mk)
        if it % eval_every == 0 or it == iters:
            vl, _ = evaluate(model, Xv, yv)
            _, ta = evaluate(model, Xte, yte)
            vlosses.append(vl)
            taccs.append(ta)
            iters_at.append(it)
    k = int(np.argmin(vlosses))
    final_w = [m.weight.detach().clone() for m in model.prunable]
    return dict(es_iter=iters_at[k], test_acc_es=taccs[k], test_acc_final=taccs[-1],
                val_loss_min=vlosses[k], weights=final_w)


# ---------------------------------------------------------------- pruning
def prune_rates(arch, cfg):
    out_rate_opt = cfg.get("output_layer_prune_rate", "half_of_other_layers")
    if arch == "lenet":
        r = {"fc": 0.2, "conv": 0.2}
    elif arch == "conv6":
        r = {"conv": 0.15, "fc": 0.2}
    else:
        r = {"conv": 0.10, "fc": 0.2}
    if out_rate_opt == "half_of_other_layers":
        r["out"] = 0.1
    elif out_rate_opt == "same_as_other_layers":
        r["out"] = r["fc"]
    else:
        r["out"] = 0.0
    return r


def prune_step(weights, masks, kinds, rates):
    new = []
    for w, mk, kind in zip(weights, masks, kinds):
        rate = rates[kind]
        alive = mk.bool()
        n_alive = int(alive.sum())
        k = int(round(rate * n_alive))
        nm = mk.clone()
        if k > 0:
            vals = w.abs()[alive]
            thr = torch.kthvalue(vals, k).values
            nm[(w.abs() <= thr) & alive] = 0.0
        new.append(nm)
    return new


def pm_of(masks):
    tot = sum(m.numel() for m in masks)
    alive = sum(int(m.sum()) for m in masks)
    return 100.0 * alive / tot


# ---------------------------------------------------------------- experiment tasks
def _lenet_cfgs(cfg, S, smoke):
    return dict(arch="lenet", lr=cfg.get("lenet_lr", 0.0012), bs=60,
                iters=max(200, int(round((300 if smoke else 50000) * S))),
                eval_every=(50 if smoke else 100),  # paper: evaluate every 100 iterations
                rounds=(4 if smoke else int(cfg.get("lenet_rounds", 18))))


def _conv_cfgs(cfg, S, smoke):
    iters = max(120, int(round((120 if smoke else 4000) * S)))
    # keep the eval budget proportional to SCALE so cost scales linearly
    n_evals = 3 if smoke else max(4, int(round(15 * S)))
    return dict(arch="conv4", lr=cfg.get("conv4_lr", 0.0003), bs=60, width=float(cfg.get("conv_width", 0.25)),
                iters=iters, eval_every=max(10, iters // n_evals),
                rounds=(2 if smoke else int(cfg.get("conv4_rounds", 10))),
                n_val=(500 if smoke else 1500), n_test=(1000 if smoke else 2500),
                train_sub=(3000 if smoke else None))


def lenet_chain(args):
    trial, cfg, S, smoke, shuffle = args
    torch.set_num_threads(1)
    L = _lenet_cfgs(cfg, S, smoke)
    data = load_data("mnist", cfg, shuffle_labels=shuffle, seed=trial)
    gen = torch.Generator().manual_seed(1000 + trial)
    proto = Net("lenet", cfg)
    glorot_init(proto, gen)
    theta0 = [m.weight.detach().clone() for m in proto.prunable]
    masks = [torch.ones_like(w) for w in theta0]
    rates = prune_rates("lenet", cfg)
    rows, masks_hist = [], []
    trained0 = None
    for rnd in range(L["rounds"] + 1):
        r = train_run("lenet", cfg, data, theta0, masks, L["iters"], L["lr"], L["bs"],
                      L["eval_every"], seed=trial * 100 + rnd)
        rows.append(dict(round=rnd, pm=pm_of(masks), es_iter=r["es_iter"],
                         test_acc_es=r["test_acc_es"], test_acc_final=r["test_acc_final"]))
        masks_hist.append([m.clone() for m in masks])
        if rnd == 0:
            trained0 = r["weights"]
        masks = prune_step(r["weights"], masks, proto.kinds, rates)
    return dict(trial=trial, rows=rows, masks=masks_hist, theta0=theta0, trained0=trained0)


def lenet_reinit(args):
    trial, rnd, mask, cfg, S, smoke, shuffle, rseed = args
    torch.set_num_threads(1)
    L = _lenet_cfgs(cfg, S, smoke)
    data = load_data("mnist", cfg, shuffle_labels=shuffle, seed=trial)
    gen = torch.Generator().manual_seed(50000 + rseed)
    proto = Net("lenet", cfg)
    glorot_init(proto, gen)  # Glorot scale from ORIGINAL dense dims
    theta0 = [m.weight.detach().clone() for m in proto.prunable]
    r = train_run("lenet", cfg, data, theta0, mask, L["iters"], L["lr"], L["bs"],
                  L["eval_every"], seed=70000 + rseed)
    return dict(trial=trial, round=rnd, pm=pm_of(mask), es_iter=r["es_iter"],
                test_acc_es=r["test_acc_es"], test_acc_final=r["test_acc_final"])


def lenet_oneshot(args):
    trial, keep, theta0, trained, cfg, S, smoke, shuffle = args
    torch.set_num_threads(1)
    L = _lenet_cfgs(cfg, S, smoke)
    data = load_data("mnist", cfg, shuffle_labels=shuffle, seed=trial)
    kinds = ["fc", "fc", "out"]
    masks = []
    for w, kind in zip(trained, kinds):
        k_keep = keep if kind != "out" else min(1.0, keep + (1 - keep) / 2)  # output at half rate
        n = w.numel()
        k = int(round((1 - k_keep) * n))
        m = torch.ones_like(w)
        if k > 0:
            thr = torch.kthvalue(w.abs().flatten(), k).values
            m[w.abs() <= thr] = 0.0
        masks.append(m)
    r = train_run("lenet", cfg, data, theta0, masks, L["iters"], L["lr"], L["bs"],
                  L["eval_every"], seed=trial * 100 + int(keep * 1000) + 3)
    return dict(trial=trial, pm=pm_of(masks), keep=keep, es_iter=r["es_iter"],
                test_acc_es=r["test_acc_es"], test_acc_final=r["test_acc_final"])


def conv_chain(args):
    trial, cfg, S, smoke, shuffle = args
    torch.set_num_threads(1)
    C = _conv_cfgs(cfg, S, smoke)
    data = load_data("cifar10", cfg, shuffle_labels=shuffle, seed=trial,
                     train_sub=C["train_sub"])
    gen = torch.Generator().manual_seed(2000 + trial)
    proto = Net("conv4", cfg, width=C["width"])
    glorot_init(proto, gen)
    theta0 = [m.weight.detach().clone() for m in proto.prunable]
    masks = [torch.ones_like(w) for w in theta0]
    rates = prune_rates("conv4", cfg)
    rows = []
    for rnd in range(C["rounds"] + 1):
        r = train_run("conv4", cfg, data, theta0, masks, C["iters"], C["lr"], C["bs"],
                      C["eval_every"], seed=trial * 100 + rnd, width=C["width"],
                      n_val=C["n_val"], n_test=C["n_test"])
        rows.append(dict(round=rnd, pm=pm_of(masks), es_iter=r["es_iter"],
                         test_acc_es=r["test_acc_es"], test_acc_final=r["test_acc_final"]))
        masks = prune_step(r["weights"], masks, proto.kinds, rates)
    return dict(trial=trial, rows=rows)


# ---------------------------------------------------------------- aggregation helpers
def curve(rows_by_trial, key):
    """rows_by_trial: list (per trial) of list of row dicts -> (pms, means)."""
    n = len(rows_by_trial[0])
    pms = [float(np.mean([t[i]["pm"] for t in rows_by_trial])) for i in range(n)]
    vals = [float(np.mean([t[i][key] for t in rows_by_trial])) for i in range(n)]
    return pms, vals


def nearest(pms, target):
    return int(np.argmin([abs(p - target) for p in pms]))


def drop_threshold(pms, accs):
    """First Pm (scanning towards sparser) whose mean acc falls below the dense mean."""
    base = accs[0]
    for p, a in zip(pms[1:], accs[1:]):
        if a < base:
            return float(p)
    return float(pms[-1])


def faster_lower_bound(pms, es):
    """Smallest Pm in the contiguous block (starting near dense) where es < dense es."""
    base = es[0]
    order = np.argsort(-np.array(pms))
    best = float(pms[0])
    started = False
    for i in order:
        if i == 0:
            continue
        if es[i] < base:
            best = float(pms[i]); started = True
        elif started:
            break
    return best
