import json, os, time, math
import numpy as np
import torch, torch.nn as nn
from torchvision import datasets

DATA_ROOT = os.environ.get("MNIST_ROOT", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_cache", "mnist")))

DEFAULTS = dict(lr_schedule="constant", learning_rate=0.1, augmentation="none",
                batch_vs_steps="steps", mixed_precision="fp32", split="test",
                mnist_splits="train60000_test10000", optimizer="sgd", init_std=0.05,
                input_binarization="threshold_0.5", bn_epsilon=1e-5,
                bn_param_init="gamma1_beta0", bn_inference_stats="moving_average",
                bn_momentum=0.9, bn_layers="hidden_only", use_bias_with_bn="drop_bias",
                weight_decay=0.0, eval_steps=[10000, 20000, 30000, 40000, 50000],
                n_seeds=3, loss="softmax_cross_entropy", shuffle="shuffle_each_epoch",
                steps=50000, batch_size=60)


def load_cfg():
    cfg = dict(DEFAULTS)
    raw = os.environ.get("REPLICATOR_CONFIG", "") or "{}"
    user = json.loads(raw)
    for k, v in user.items():
        if v is not None:
            cfg[k] = v
    for k in ("learning_rate", "init_std", "bn_epsilon", "bn_momentum", "weight_decay"):
        cfg[k] = float(cfg[k])
    return cfg


def binarize(x, mode):
    if mode == "threshold_0.5":
        return (x > 0.5).float()
    if mode == "threshold_0.3":
        return (x > 0.3).float()
    if mode == "bernoulli_sample":
        return torch.bernoulli(x)
    return x


def get_data(cfg, smoke):
    tr = datasets.MNIST(DATA_ROOT, train=True, download=False)
    te = datasets.MNIST(DATA_ROOT, train=False, download=False)
    Xtr = tr.data.float().div(255.).view(-1, 784)
    ytr = tr.targets.clone()
    Xte = te.data.float().div(255.).view(-1, 784)
    yte = te.targets.clone()
    if cfg["mnist_splits"] == "train50000_val10000_test10000":
        if cfg["split"] == "validation":
            Xte, yte = Xtr[50000:], ytr[50000:]
        Xtr, ytr = Xtr[:50000], ytr[:50000]
    if smoke:
        Xtr, ytr = Xtr[:6000], ytr[:6000]
        Xte, yte = Xte[:2000], yte[:2000]
    Xtr = binarize(Xtr, cfg["input_binarization"])
    Xte = binarize(Xte, cfg["input_binarization"])
    return Xtr, ytr, Xte, yte


class MLP(nn.Module):
    def __init__(self, cfg, bn):
        super().__init__()
        self.bn = bn
        dims = [784, 100, 100, 100]
        bias = (not bn) or cfg["use_bias_with_bn"] == "keep_bias"
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(3)])
        self.out = nn.Linear(100, 10)
        self.bns = nn.ModuleList([nn.BatchNorm1d(100, eps=cfg["bn_epsilon"],
                                                 momentum=1.0 - cfg["bn_momentum"])
                                  for _ in range(3)]) if bn else None
        self.out_bn = (nn.BatchNorm1d(10, eps=cfg["bn_epsilon"], momentum=1.0 - cfg["bn_momentum"])
                       if bn and cfg["bn_layers"] == "hidden_plus_output" else None)
        std = cfg["init_std"]
        for m in list(self.lins) + [self.out]:
            if std == "glorot" or cfg["init_std"] == 0:
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.normal_(m.weight, 0.0, float(std))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if bn and cfg["bn_param_init"] == "gamma_random_small_beta0":
            for b in self.bns:
                nn.init.normal_(b.weight, 1.0, 0.1)

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if self.bn:
                x = self.bns[i](x)
            x = torch.sigmoid(x)
        x = self.out(x)
        if self.out_bn is not None:
            x = self.out_bn(x)
        return x


def evaluate(model, X, y, Xtr=None, cfg=None):
    model.eval()
    if cfg is not None and model.bn and cfg["bn_inference_stats"] == "population_unbiased":
        # recompute population stats from training minibatches
        m = cfg["batch_size"]
        nb = min(200, len(Xtr) // m)
        mus = {i: [] for i in range(len(model.bns))}
        vars_ = {i: [] for i in range(len(model.bns))}
        with torch.no_grad():
            for b in range(nb):
                h = Xtr[b * m:(b + 1) * m]
                for i, lin in enumerate(model.lins):
                    z = lin(h)
                    mus[i].append(z.mean(0))
                    vars_[i].append(z.var(0, unbiased=False))
                    h = torch.sigmoid(model.bns[i](z if False else z))
            for i, b in enumerate(model.bns):
                b.running_mean.copy_(torch.stack(mus[i]).mean(0))
                b.running_var.copy_(torch.stack(vars_[i]).mean(0) * m / (m - 1))
    with torch.no_grad():
        correct = 0
        for i in range(0, len(X), 1000):
            logits = model(X[i:i + 1000])
            correct += (logits.argmax(1) == y[i:i + 1000]).sum().item()
    model.train()
    return correct / len(X)


def make_opt(model, cfg):
    o = cfg["optimizer"]
    lr, wd = cfg["learning_rate"], cfg["weight_decay"]
    if o == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    if o == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)


def train(cfg, bn, seed, Xtr, ytr, Xte, yte, steps, eval_steps):
    torch.manual_seed(seed + (1000 if bn else 0))
    rng = np.random.default_rng(seed)
    model = MLP(cfg, bn)
    opt = make_opt(model, cfg)
    sched = None
    if cfg["lr_schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    elif cfg["lr_schedule"] in ("step", "as_paper"):
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 3), gamma=0.1)
    lossf = nn.CrossEntropyLoss() if cfg["loss"] == "softmax_cross_entropy" else nn.MSELoss()
    m = int(cfg["batch_size"])
    n = len(Xtr)
    perm = rng.permutation(n)
    pos = 0
    curve = {}
    model.train()
    for step in range(1, steps + 1):
        if cfg["shuffle"] == "random_with_replacement":
            idx = rng.integers(0, n, m)
        else:
            if pos + m > n:
                perm = rng.permutation(n) if cfg["shuffle"] == "shuffle_each_epoch" else np.arange(n)
                pos = 0
            idx = perm[pos:pos + m]
            pos += m
        xb, yb = Xtr[idx], ytr[idx]
        logits = model(xb)
        if cfg["loss"] == "mse":
            loss = lossf(torch.softmax(logits, 1), torch.nn.functional.one_hot(yb, 10).float())
        else:
            loss = lossf(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if step in eval_steps:
            curve[step] = evaluate(model, Xte, yte, Xtr, cfg)
    return curve


def main():
    t0 = time.time()
    seed = int(os.environ.get("SEED", "0"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    cfg = load_cfg()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    Xtr, ytr, Xte, yte = get_data(cfg, smoke)
    steps = int(cfg["steps"])
    scale = {}
    if smoke:
        steps = 3000
        scale = dict(smoke=True, steps=steps, train_subset=len(Xtr), eval_subset=len(Xte))
    es = cfg["eval_steps"]
    if isinstance(es, str):
        es = {"every_500_steps": list(range(500, steps + 1, 500)),
              "end_only": [steps],
              "{10000,20000,30000,40000,50000}": [10000, 20000, 30000, 40000, 50000]}[es]
    eval_steps = sorted({s for s in es if s <= steps} | {steps})
    curve_bn = train(cfg, True, seed, Xtr, ytr, Xte, yte, steps, eval_steps)
    curve_nb = train(cfg, False, seed, Xtr, ytr, Xte, yte, steps, eval_steps)
    metrics = {}
    for s in eval_steps:
        metrics[f"acc_bn_{s}"] = curve_bn[s]
        metrics[f"acc_nobn_{s}"] = curve_nb[s]
        metrics[f"gap_{s}"] = curve_bn[s] - curve_nb[s]
    final = eval_steps[-1]
    metrics["accuracy_gap"] = curve_bn[final] - curve_nb[final]
    metrics["accuracy"] = curve_bn[final]
    # steps for BN net to first reach the non-BN net's final accuracy (speed-of-training proxy);
    # ImageNet 'training_steps' claims are UNTESTED (ILSVRC2012 unavailable, CPU-only).
    target = curve_nb[final]
    reached = [s for s in eval_steps if curve_bn[s] >= target]
    metrics["training_steps"] = float(reached[0] if reached else final)
    metrics["_intermediates"] = {
        "scale": scale or {"full": True, "steps": steps},
        "n_train": int(len(Xtr)), "n_eval": int(len(Xte)),
        "eval_steps": eval_steps, "curve_bn": curve_bn, "curve_nobn": curve_nb,
        "config": {k: cfg[k] for k in cfg if not k.startswith("_")},
        "runtime_s": round(time.time() - t0, 1),
        "untested_claims": ["inception_*/bn_* ImageNet claims: ILSVRC2012 unavailable, CPU-only"],
    }
    out = {"seed": seed, "split": cfg["split"], "n_examples": int(len(Xte)), "metrics": metrics}
    path = os.environ.get("METRICS_OUT", "metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "_intermediates"}, indent=1))


if __name__ == "__main__":
    main()
