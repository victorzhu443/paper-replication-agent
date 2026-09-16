"""Core models/utilities for the Layer Normalization (Ba et al., 2016) reduced-scale replication.

Everything ambiguous is driven by a config dict (keys = spec ambiguity config_key list).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- config


DEFAULTS = {
    # ambiguity list (spec defaults)
    "lr_schedule": "constant",
    "learning_rate": 0.001,
    "adam_betas_eps": "beta1=0.9,beta2=0.999,eps=1e-8",
    "augmentation": "none",
    "batch_vs_steps": "epochs",
    "mixed_precision": "fp32",
    "split": "test",
    "ln_eps": 1e-5,
    "ln_param_init": "gain=1,bias=0",
    "ln_bias_handling": "ln_beta_only",
    "activation": "relu",
    "weight_init": "glorot_uniform",
    "n_seeds": 1,
    "max_epochs": 60,
    "bn_scope": "all_layers",
    "variance_estimator": "biased_LN_unbiased_BN_small_batch",
    "bn_inference_stats": "running_average_momentum_0.9",
    "val_split": "last_5000",
    "regularization": "none",
    "grad_clip": "none",
    "replication_task": "mnist_mlp",
    "speedup_metric": "iterations_for_LN_to_reach_its_own_best_vs_baseline_to_reach_its_own_best",
    "checkpoint_criterion": "sum_of_R@1_R@5_R@10",
    "reporting_statistic": "final_epoch",
    "hidden_width": 1000,
    # compute-tier-2 scaling knobs (NOT paper choices; recorded in metrics._intermediates.scale)
    "scale_mlp_train_subset": 55000,
    "scale_mlp_epochs_bz128": 20,
    "scale_mlp_epochs_bz4": 5,
    "scale_mlp_train_subset_bz4": 8000,
    "scale_val_subset": 5000,
    "scale_eval_every_steps": 100,
    "run_bz4": True,
    "run_gru_aux": True,
    "run_draw_lite": True,
    "scale_gru_hidden": 128,
    "scale_gru_train_subset": 8000,
    "scale_gru_epochs": 4,
    "scale_gru_batch": 64,
    "scale_draw_glimpses": 16,
    "scale_draw_hidden": 128,
    "scale_draw_latent": 32,
    "scale_draw_train_subset": 12000,
    "scale_draw_epochs": 8,
    "scale_draw_batch": 128,
    "conv_smooth_window": 3,
    "train_nll_threshold": 0.1,
}


def resolve_config(user: dict | None) -> dict:
    cfg = dict(DEFAULTS)
    for k, v in (user or {}).items():
        if k.startswith("_"):
            cfg[k] = v
        elif k in cfg:
            # keep numeric types
            if isinstance(cfg[k], bool):
                cfg[k] = bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "yes")
            elif isinstance(cfg[k], int) and not isinstance(cfg[k], bool):
                cfg[k] = int(float(v))
            elif isinstance(cfg[k], float):
                cfg[k] = float(v)
            else:
                cfg[k] = v
        else:
            cfg[k] = v
    return cfg


def parse_adam(cfg) -> tuple[tuple[float, float], float]:
    s = str(cfg["adam_betas_eps"])
    d = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=")
            d[k.strip()] = float(v)
    return (d.get("beta1", 0.9), d.get("beta2", 0.999)), d.get("eps", 1e-8)


# ----------------------------------------------------------------------------- data

MNIST_ROOTS = [
    os.environ.get("MNIST_ROOT", ""),
    "../../../data_cache/mnist",
    "../../data_cache/mnist",
    "../data_cache/mnist",
    "data_cache/mnist",
    os.path.expanduser("~/Developer/research-copy/runs/data_cache/mnist"),
]


def load_mnist():
    from torchvision import datasets

    root = None
    for cand in MNIST_ROOTS:
        if cand and os.path.isdir(os.path.join(cand, "MNIST", "raw")):
            root = cand
            break
    if root is None:
        root = MNIST_ROOTS[1]
        os.makedirs(root, exist_ok=True)
        download = True
    else:
        download = False
    tr = datasets.MNIST(root, train=True, download=download)
    te = datasets.MNIST(root, train=False, download=download)
    Xtr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    ytr = tr.targets.numpy().astype(np.int64)
    Xte = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    yte = te.targets.numpy().astype(np.int64)
    return Xtr, ytr, Xte, yte, os.path.abspath(root)


def make_splits(cfg, smoke: bool):
    """MNIST: 55k train / 5k val (last_5000 convention) / 10k test, then tier-2 subsetting."""
    Xtr, ytr, Xte, yte, root = load_mnist()
    n_all = len(Xtr)
    if cfg["val_split"] == "none_held_out":
        tr_idx = np.arange(n_all)
        val_idx = np.arange(n_all - 5000, n_all)
    elif cfg["val_split"] == "random_5000_fixed_seed":
        rng = np.random.RandomState(1234)
        perm = rng.permutation(n_all)
        val_idx, tr_idx = perm[:5000], perm[5000:]
    else:  # last_5000
        tr_idx, val_idx = np.arange(0, n_all - 5000), np.arange(n_all - 5000, n_all)
    return dict(X=Xtr, y=ytr, tr_idx=tr_idx, val_idx=val_idx, Xte=Xte, yte=yte, root=root)


# ----------------------------------------------------------------------------- modules


def init_weight_(w: torch.Tensor, scheme: str, recurrent: bool = False):
    if scheme == "he_normal":
        nn.init.kaiming_normal_(w, nonlinearity="relu")
    elif scheme == "gaussian_0.01":
        nn.init.normal_(w, 0.0, 0.01)
    elif scheme == "orthogonal_recurrent" and recurrent:
        # orthogonal per HxH block
        with torch.no_grad():
            out, inp = w.shape
            if out % inp == 0:
                for i in range(out // inp):
                    blk = torch.empty(inp, inp)
                    nn.init.orthogonal_(blk)
                    w[i * inp:(i + 1) * inp].copy_(blk)
            else:
                nn.init.orthogonal_(w)
    else:
        nn.init.xavier_uniform_(w)


class LayerNorm(nn.Module):
    """LN per Eq. 15/16: biased 1/H variance, learned gain+bias (default gain=1, bias=0)."""

    def __init__(self, dim: int, eps: float, param_init: str = "gain=1,bias=0"):
        super().__init__()
        g, b = (1.0, 0.0) if param_init != "gain=0,bias=1" else (0.0, 1.0)
        self.gain = nn.Parameter(torch.full((dim,), g))
        self.bias = nn.Parameter(torch.full((dim,), b))
        self.eps = float(eps)
        self.dim = dim

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.gain + self.bias


class BatchNorm(nn.Module):
    """BN with explicit control of the normalization variance estimator and running-avg decay."""

    def __init__(self, dim: int, unbiased: bool, decay: float = 0.9, eps: float = 1e-5,
                 use_batch_stats_at_test: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.unbiased = unbiased
        self.decay = decay
        self.eps = eps
        self.use_batch_stats_at_test = use_batch_stats_at_test

    def forward(self, x):
        if self.training or self.use_batch_stats_at_test:
            mean = x.mean(0)
            n = x.shape[0]
            var_b = x.var(0, unbiased=False)
            var_u = x.var(0, unbiased=True) if n > 1 else var_b
            var = var_u if self.unbiased else var_b
            with torch.no_grad():
                self.running_mean.mul_(self.decay).add_((1 - self.decay) * mean)
                self.running_var.mul_(self.decay).add_((1 - self.decay) * var_u)
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


ACTS = {"relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


class MLP(nn.Module):
    """784-W-W-10 permutation-invariant MLP; norm in {none, ln, bn}."""

    def __init__(self, cfg, norm: str, width: int, batch_size: int, in_dim=784, out_dim=10):
        super().__init__()
        self.norm = norm
        self.act = ACTS[cfg["activation"]]
        keep_bias = (norm == "none") or (norm == "ln" and cfg["ln_bias_handling"] == "keep_layer_bias_and_ln_beta")
        dims = [in_dim, width, width]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=keep_bias) for i in range(2)])
        self.out = nn.Linear(width, out_dim, bias=(norm != "bn" or cfg["bn_scope"] != "all_layers"))
        for lin in list(self.lins) + [self.out]:
            init_weight_(lin.weight, cfg["weight_init"])
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        ve = cfg["variance_estimator"]
        bn_unbiased = (ve == "unbiased_all" or ve == "biased_LN_unbiased_BN_all_batches"
                       or (ve == "biased_LN_unbiased_BN_small_batch" and batch_size < 32))
        decay = 0.99 if cfg["bn_inference_stats"] == "running_average_momentum_0.99" else 0.9
        use_batch_test = cfg["bn_inference_stats"] == "batch_statistics_at_test"
        if norm == "ln":
            self.norms = nn.ModuleList([LayerNorm(width, cfg["ln_eps"], cfg["ln_param_init"]) for _ in range(2)])
            self.out_norm = None  # LN excludes the softmax layer (Sec 6.6)
        elif norm == "bn":
            self.norms = nn.ModuleList([BatchNorm(width, bn_unbiased, decay, use_batch_stats_at_test=use_batch_test)
                                        for _ in range(2)])
            self.out_norm = (BatchNorm(out_dim, bn_unbiased, decay, use_batch_stats_at_test=use_batch_test)
                             if cfg["bn_scope"] == "all_layers" else None)
        else:
            self.norms = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.out_norm = None

    def forward(self, x):
        for lin, nrm in zip(self.lins, self.norms):
            x = self.act(nrm(lin(x)))
        x = self.out(x)
        if self.out_norm is not None:
            x = self.out_norm(x)
        return x


class GRUCell(nn.Module):
    """GRU, optionally with LN inside per Appendix Eqs. 26-28."""

    def __init__(self, cfg, in_dim, hid, ln: bool):
        super().__init__()
        self.ln = ln
        self.hid = hid
        self.W_x_zr = nn.Parameter(torch.empty(2 * hid, in_dim))
        self.W_h_zr = nn.Parameter(torch.empty(2 * hid, hid))
        self.W_x_h = nn.Parameter(torch.empty(hid, in_dim))
        self.U_h = nn.Parameter(torch.empty(hid, hid))
        for w, rec in [(self.W_x_zr, False), (self.W_h_zr, True), (self.W_x_h, False), (self.U_h, True)]:
            init_weight_(w, cfg["weight_init"], recurrent=rec)
        if not ln:
            self.b_zr = nn.Parameter(torch.zeros(2 * hid))
            self.b_h = nn.Parameter(torch.zeros(hid))
        else:
            e, pi = cfg["ln_eps"], cfg["ln_param_init"]
            self.ln1 = LayerNorm(2 * hid, e, pi)
            self.ln2 = LayerNorm(2 * hid, e, pi)
            self.ln3 = LayerNorm(hid, e, pi)
            self.ln4 = LayerNorm(hid, e, pi)

    def forward(self, x, h):
        if self.ln:
            zr = self.ln1(h @ self.W_h_zr.T) + self.ln2(x @ self.W_x_zr.T)
            z, r = zr.chunk(2, -1)
            hh = torch.tanh(self.ln3(x @ self.W_x_h.T) + torch.sigmoid(r) * self.ln4(h @ self.U_h.T))
        else:
            zr = h @ self.W_h_zr.T + x @ self.W_x_zr.T + self.b_zr
            z, r = zr.chunk(2, -1)
            hh = torch.tanh(x @ self.W_x_h.T + torch.sigmoid(r) * (h @ self.U_h.T) + self.b_h)
        sz = torch.sigmoid(z)
        return (1 - sz) * h + sz * hh


class GRUClassifier(nn.Module):
    def __init__(self, cfg, in_dim, hid, n_cls, ln: bool):
        super().__init__()
        self.cell = GRUCell(cfg, in_dim, hid, ln)
        self.out = nn.Linear(hid, n_cls)
        init_weight_(self.out.weight, cfg["weight_init"])
        nn.init.zeros_(self.out.bias)
        self.hid = hid

    def forward(self, x):  # x: (B, T, in_dim)
        h = x.new_zeros(x.shape[0], self.hid)
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h)
        return self.out(h)


class LSTMCellLN(nn.Module):
    """LSTM cell; DRAW-style LN on the cell state only (Appendix Eq. 31)."""

    def __init__(self, cfg, in_dim, hid, ln_cell: bool):
        super().__init__()
        self.hid = hid
        self.W_x = nn.Parameter(torch.empty(4 * hid, in_dim))
        self.W_h = nn.Parameter(torch.empty(4 * hid, hid))
        self.b = nn.Parameter(torch.zeros(4 * hid))
        init_weight_(self.W_x, cfg["weight_init"])
        init_weight_(self.W_h, cfg["weight_init"], recurrent=True)
        self.ln = LayerNorm(hid, cfg["ln_eps"], cfg["ln_param_init"]) if ln_cell else None

    def forward(self, x, state):
        h, c = state
        g = x @ self.W_x.T + h @ self.W_h.T + self.b
        i, f, u, o = g.chunk(4, -1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u)
        cc = self.ln(c) if self.ln is not None else c
        h = torch.sigmoid(o) * torch.tanh(cc)
        return h, c


class DrawLite(nn.Module):
    """Scaled-down DRAW (no attention: full-canvas read/write) with T glimpses."""

    def __init__(self, cfg, ln: bool, T: int, hid: int, z: int, x_dim: int = 784):
        super().__init__()
        self.T, self.hid, self.z, self.x_dim = T, hid, z, x_dim
        self.enc = LSTMCellLN(cfg, 2 * x_dim + hid, hid, ln)
        self.dec = LSTMCellLN(cfg, z, hid, ln)
        self.mu = nn.Linear(hid, z)
        self.logvar = nn.Linear(hid, z)
        self.write = nn.Linear(hid, x_dim)
        for lin in [self.mu, self.logvar, self.write]:
            init_weight_(lin.weight, cfg["weight_init"])
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        B = x.shape[0]
        h_e = x.new_zeros(B, self.hid); c_e = x.new_zeros(B, self.hid)
        h_d = x.new_zeros(B, self.hid); c_d = x.new_zeros(B, self.hid)
        canvas = x.new_zeros(B, self.x_dim)
        kl = x.new_zeros(())
        for _ in range(self.T):
            err = x - torch.sigmoid(canvas)
            r = torch.cat([x, err, h_d], dim=-1)
            h_e, c_e = self.enc(r, (h_e, c_e))
            mu = self.mu(h_e)
            logvar = torch.clamp(self.logvar(h_e), -8.0, 8.0)
            zz = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            kl = kl + 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1).mean()
            h_d, c_d = self.dec(zz, (h_d, c_d))
            canvas = canvas + self.write(h_d)
        rec = F.binary_cross_entropy_with_logits(canvas, x, reduction="none").sum(-1).mean()
        return rec + kl, rec.detach(), kl.detach()


# ----------------------------------------------------------------------------- training helpers


def make_optimizer(model, cfg, total_steps: int):
    betas, eps = parse_adam(cfg)
    wd = 1e-4 if cfg["regularization"] == "weight_decay_1e-4" else 0.0
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), betas=betas, eps=eps, weight_decay=wd)
    sch = None
    if cfg["lr_schedule"] == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
    elif cfg["lr_schedule"] == "step":
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(total_steps // 3, 1), gamma=0.1)
    return opt, sch


def clip_grads(model, cfg):
    if cfg["grad_clip"] == "clip_norm_5":
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    elif cfg["grad_clip"] == "clip_norm_10":
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)


@torch.no_grad()
def eval_classifier(model, X, y, batch=1000, seq=False):
    model.eval()
    n, correct, nll = len(X), 0, 0.0
    for i in range(0, n, batch):
        xb = X[i:i + batch]
        logits = model(xb)
        yb = y[i:i + batch]
        nll += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(-1) == yb).sum().item()
    model.train()
    return 1.0 - correct / n, nll / n


def smooth_curve(curve, window: int):
    """Moving average over the value channel (robust 'best' detection on noisy val curves)."""
    if window <= 1:
        return list(curve)
    out = []
    vals = [c[1] for c in curve]
    for i in range(len(curve)):
        lo = max(0, i - window + 1)
        out.append((curve[i][0], float(np.mean(vals[lo:i + 1])), curve[i][2]))
    return out


def steps_to_own_best(curve, higher_is_better=False):
    """curve: list of (step, value, wallclock). Returns (step, wallclock, best_value)."""
    vals = [c[1] for c in curve]
    best = max(vals) if higher_is_better else min(vals)
    idx = vals.index(best)
    return curve[idx][0], curve[idx][2], best


def steps_to_threshold(curve, thresh, higher_is_better=False):
    for step, val, wc in curve:
        if (val >= thresh) if higher_is_better else (val <= thresh):
            return step, wc
    return curve[-1][0], curve[-1][2]
