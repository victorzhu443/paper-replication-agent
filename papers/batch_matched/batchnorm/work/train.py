"""BN vs no-BN 3x100 sigmoid MLP on MNIST (Ioffe & Szegedy 2015, Sec 4.1 / Fig 1)."""
import json, os, time, math
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("NUM_THREADS", "8")))

CFG_DEFAULTS = dict(
    lr_schedule="constant", learning_rate=0.1, lr_bn_vs_baseline="same_lr_both",
    augmentation="none", batch_vs_steps="steps", batch_sampling="shuffle_each_epoch",
    mixed_precision="fp32", split="test", optimizer="sgd",
    input_binarization="threshold_0.5", init_std=0.05, bn_eps=1e-5,
    bn_inference_stats="moving_average", bn_placement="before_sigmoid_on_Wu",
    bn_layers="hidden_only", use_bias_with_bn="remove_bias", bn_affine_init="gamma=1,beta=0",
    n_seeds=3, eval_every_steps=500, accuracy_comparison_point="final_step",
    tracked_unit="random_unit_last_hidden_layer",
    drift_statistic="std_over_training_of_percentiles",
    percentile_data="current_training_minibatch",
    batch_size=60, training_steps=50000,
)


def get_cfg():
    cfg = dict(CFG_DEFAULTS)
    raw = os.environ.get("REPLICATOR_CONFIG", "") or "{}"
    try:
        user = json.loads(raw)
    except Exception:
        user = {}
    cfg.update({k: v for k, v in user.items()})
    return cfg


def load_mnist(cfg):
    import torchvision
    root = os.environ.get("MNIST_ROOT", "data_cache")
    tr = torchvision.datasets.MNIST(root=root, train=True, download=True)
    te = torchvision.datasets.MNIST(root=root, train=False, download=True)
    Xtr = tr.data.numpy().astype(np.float32) / 255.0
    Xte = te.data.numpy().astype(np.float32) / 255.0
    ytr = tr.targets.numpy().astype(np.int64)
    yte = te.targets.numpy().astype(np.int64)
    b = cfg["input_binarization"]
    def binz(X, rng=None):
        if b == "threshold_0.5":
            return (X > 0.5).astype(np.float32)
        if b == "threshold_0":
            return (X > 0.0).astype(np.float32)
        if b == "bernoulli_sample":
            return (rng.random(X.shape) < X).astype(np.float32)
        return X
    rng = np.random.default_rng(0)
    Xtr = binz(Xtr, rng).reshape(len(Xtr), -1)
    Xte = binz(Xte, rng).reshape(len(Xte), -1)
    return (torch.from_numpy(Xtr), torch.from_numpy(ytr),
            torch.from_numpy(Xte), torch.from_numpy(yte))


class MLP(nn.Module):
    def __init__(self, bn, cfg):
        super().__init__()
        self.bn = bn
        self.cfg = cfg
        dims = [784, 100, 100, 100]
        bias = not (bn and cfg["use_bias_with_bn"] == "remove_bias")
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(3)])
        self.out = nn.Linear(100, 10)
        self.bns = nn.ModuleList([nn.BatchNorm1d(100, eps=float(cfg["bn_eps"])) for _ in range(3)]) if bn else None
        self.out_bn = nn.BatchNorm1d(10, eps=float(cfg["bn_eps"])) if (bn and cfg["bn_layers"] == "hidden_and_output") else None
        std = cfg["init_std"]
        for m in list(self.lins) + [self.out]:
            if std == "glorot":
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.normal_(m.weight, 0.0, float(std))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_pre=False):
        pre = None
        for i, lin in enumerate(self.lins):
            z = lin(x)
            if self.bn and self.cfg["bn_placement"] == "before_sigmoid_on_Wu":
                z = self.bns[i](z)
                if i == 2:
                    pre = z
                x = torch.sigmoid(z)
            else:
                if i == 2:
                    pre = z
                x = torch.sigmoid(z)
                if self.bn:
                    x = self.bns[i](x)
        logits = self.out(x)
        if self.out_bn is not None:
            logits = self.out_bn(logits)
        return (logits, pre) if return_pre else logits


def evaluate(model, Xte, yte, cfg):
    model.eval()
    if cfg["bn_inference_stats"] == "batch_stats":
        model.train()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(Xte), 1000):
            logits = model(Xte[i:i + 1000])
            correct += (logits.argmax(1) == yte[i:i + 1000]).sum().item()
    model.train()
    return correct / len(Xte)


def train_one(bn, cfg, seed, steps, eval_every, data, shuffle_labels):
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(seed + (1000 if bn else 0))
    rng = np.random.default_rng(seed + (1000 if bn else 0))
    model = MLP(bn, cfg)
    lr = float(cfg["learning_rate"])
    if bn and cfg["lr_bn_vs_baseline"] == "higher_lr_for_bn":
        lr *= 5.0
    if cfg["optimizer"] == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif cfg["optimizer"] == "adagrad":
        opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    sched = None
    if cfg["lr_schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    elif cfg["lr_schedule"] == "exponential_decay":
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1.0 / max(steps, 1)))
    elif cfg["lr_schedule"] == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(steps // 3, 1), gamma=0.1)
    lossf = nn.CrossEntropyLoss()
    bs = int(cfg["batch_size"])
    n = len(Xtr)
    if cfg["tracked_unit"] == "fixed_index_unit_last_hidden_layer":
        unit = 0
    else:
        unit = int(rng.integers(100))
    perm = torch.from_numpy(rng.permutation(n)) if cfg["batch_sampling"] == "shuffle_each_epoch" else torch.arange(n)
    ptr = 0
    curve, pct_track = [], []
    model.train()
    for step in range(1, steps + 1):
        if cfg["batch_sampling"] == "random_with_replacement":
            idx = torch.from_numpy(rng.integers(0, n, bs))
        else:
            if ptr + bs > n:
                perm = torch.from_numpy(rng.permutation(n)) if cfg["batch_sampling"] == "shuffle_each_epoch" else torch.arange(n)
                ptr = 0
            idx = perm[ptr:ptr + bs]
            ptr += bs
        xb, yb = Xtr[idx], ytr[idx]
        if shuffle_labels:  # shuffle labels within the batch before training
            yb = yb[torch.from_numpy(rng.permutation(len(yb)))]
        logits, pre = model(xb, return_pre=True)
        loss = lossf(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if step % eval_every == 0 or step == steps:
            acc = evaluate(model, Xte, yte, cfg)
            curve.append((step, acc))
            with torch.no_grad():
                v = pre[:, unit].detach().numpy()
            pct_track.append(np.percentile(v, [15, 50, 85]))
    return curve, np.array(pct_track)


def drift(pcts, stat):
    if stat == "range_of_percentiles_over_training":
        d = pcts.max(0) - pcts.min(0)
    elif stat == "mean_abs_change_between_consecutive_evals":
        d = np.abs(np.diff(pcts, axis=0)).mean(0)
    else:
        d = pcts.std(0)
    return float(np.mean(d))


def main():
    t0 = time.time()
    cfg = get_cfg()
    seed = int(os.environ.get("SEED", "0"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    scale = float(os.environ.get("SCALE", "1.0"))
    shuffle_labels = bool(cfg.get("_shuffle_labels", False))
    full_steps = int(cfg["training_steps"])
    eval_every = int(cfg["eval_every_steps"])
    if smoke:
        steps, eval_every = 600, 100
    else:
        steps = max(int(round(full_steps * scale)), 100)
        eval_every = max(int(round(eval_every * scale)), 20)
    data = load_mnist(cfg)
    Xtr, ytr, Xte, yte = data
    if smoke:
        data = (Xtr[:6000], ytr[:6000], Xte[:2000], yte[:2000])

    res = {}
    for name, bn in [("with_bn", True), ("baseline_no_bn", False)]:
        curve, pcts = train_one(bn, cfg, seed, steps, eval_every, data, shuffle_labels)
        res[name] = dict(curve=curve, pcts=pcts)

    def final_acc(curve):
        mode = cfg["accuracy_comparison_point"]
        accs = [a for _, a in curve]
        if mode == "max_over_training":
            return max(accs)
        if mode == "mean_of_last_k_evals":
            return float(np.mean(accs[-5:]))
        return accs[-1]

    bn_curve, nb_curve = res["with_bn"]["curve"], res["baseline_no_bn"]["curve"]
    bn_final, nb_final = final_acc(bn_curve), final_acc(nb_curve)
    # steps at which BN first reaches baseline's final accuracy, rescaled to the 50k-step budget
    hit = next((s for s, a in bn_curve if a >= nb_final), steps)
    steps_to = float(hit) * (full_steps / steps)
    # accuracy at 10k steps (same fraction 0.2 of the budget when scaled)
    target = 0.2 * steps
    bn_at_10k = min(bn_curve, key=lambda sa: abs(sa[0] - target))[1]
    d_bn = drift(res["with_bn"]["pcts"], cfg["drift_statistic"])
    d_nb = drift(res["baseline_no_bn"]["pcts"], cfg["drift_statistic"])

    metrics = {
        "accuracy": bn_final,
        "accuracy_gap": bn_final - nb_final,
        "steps_to_baseline_final_accuracy": steps_to,
        "activation_distribution_drift_reduction": d_nb - d_bn,
        "mnist_bn_gt_nobn_final": bn_final - nb_final,
        "mnist_bn_final_acc": bn_final,
        "mnist_nobn_final_acc": nb_final,
        "mnist_bn_acc_at_10k": bn_at_10k,
        "mnist_bn_faster_to_baseline_acc": steps_to,
        "mnist_bn_activation_stability": d_nb - d_bn,
        "chance_level": 0.1,
        "accuracy_gap_baseline": 0.0,
        "steps_to_baseline_final_accuracy_baseline": float(full_steps),
        "activation_distribution_drift_reduction_baseline": 0.0,
        "_intermediates": {
            "scale": {"scale": scale, "smoke": smoke, "steps_run": steps,
                      "paper_steps": full_steps, "batch_size": int(cfg["batch_size"]),
                      "n_train_used": int(len(data[0])), "n_test_used": int(len(data[2]))},
            "bn_drift": d_bn, "baseline_drift": d_nb,
            "bn_curve_tail": bn_curve[-3:], "nobn_curve_tail": nb_curve[-3:],
            "runtime_s": time.time() - t0,
        },
    }
    out = {
        "seed": seed, "split": cfg["split"], "n_examples": int(len(data[2])),
        "scale": scale, "shuffled": shuffle_labels,
        "matched_scale": (not smoke) and scale >= 1.0,
        "metrics": metrics,
    }
    path = os.environ.get("METRICS_OUT", "metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "_intermediates"}, indent=1))
    print("runtime", time.time() - t0)


if __name__ == "__main__":
    main()
