"""Toy Models of Superposition - replication core."""
from __future__ import annotations
import json, math, os, time
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", "15")))


def init_W(R, m, n, how, gen):
    W = torch.empty(R, m, n)
    for r in range(R):
        if how == "xavier_normal":
            torch.nn.init.xavier_normal_(W[r], generator=gen) if False else None
            std = math.sqrt(2.0 / (m + n))
            W[r] = torch.randn(m, n, generator=gen) * std
        elif how == "kaiming_uniform":
            bound = math.sqrt(6.0 / n)
            W[r] = (torch.rand(m, n, generator=gen) * 2 - 1) * bound
        else:  # normal_0.1 / zeros_bias_only
            W[r] = torch.randn(m, n, generator=gen) * 0.1
    return W


def train(n, m, model_type, importance, sparsity, steps, batch, cfg, gen,
          restarts=1, shuffle_labels=False, feat_density=None):
    """sparsity: float or array of length R (one per restart-slot)."""
    R = restarts
    S = torch.as_tensor(np.broadcast_to(np.asarray(sparsity, dtype=np.float32), (R,)).copy())
    dens = (1.0 - S).view(R, 1, 1)
    if feat_density is not None:
        dens = dens * torch.as_tensor(feat_density, dtype=torch.float32).view(1, 1, n)
    W = init_W(R, m, n, cfg.get("weight_init", "xavier_normal"), gen).requires_grad_(True)
    b = torch.zeros(R, n, requires_grad=True)
    lr = float(cfg.get("learning_rate", 1e-3))
    opt_name = cfg.get("optimizer", "adam")
    params = [W, b]
    if opt_name == "sgd":
        opt = torch.optim.SGD(params, lr=lr)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(params, lr=lr)
    else:
        opt = torch.optim.Adam(params, lr=lr)
    sched = None
    if cfg.get("lr_schedule") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    elif cfg.get("lr_schedule") == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 3), gamma=0.3)
    I = torch.as_tensor(importance, dtype=torch.float32).view(1, 1, n)
    sign = 1.0 if cfg.get("bias_sign", "plus_b") == "plus_b" else -1.0
    fixed = None
    if cfg.get("data_resampling", "fresh_each_step") == "fixed_dataset":
        fixed = sample(R, batch * 8, n, dens, gen)

    def fwd(x):
        h = torch.einsum("rmn,rbn->rbm", W, x)
        if model_type == "relu_hidden":
            h = torch.relu(h)
        out = torch.einsum("rmn,rbm->rbn", W, h) + sign * b.unsqueeze(1)
        if model_type != "linear":
            out = torch.relu(out)
        return out

    for t in range(steps):
        if fixed is None:
            x = sample(R, batch, n, dens, gen)
        else:
            idx = torch.randint(0, fixed.shape[1], (batch,), generator=gen)
            x = fixed[:, idx]
        tgt = x
        if shuffle_labels:
            perm = torch.randperm(batch, generator=gen)
            tgt = x[:, perm]
        out = fwd(x)
        err = I * (tgt - out) ** 2
        red = cfg.get("loss_reduction", "sum_over_features_mean_over_batch")
        if red == "mean_over_both":
            loss_r = err.mean(dim=(1, 2))
        elif red == "sum_over_both":
            loss_r = err.sum(dim=(1, 2))
        else:
            loss_r = err.sum(dim=2).mean(dim=1)
        loss = loss_r.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
    # final loss estimate
    with torch.no_grad():
        x = sample(R, max(batch, 2048), n, dens, gen)
        out = fwd(x)
        final = (I * (x - out) ** 2).sum(dim=2).mean(dim=1)
    return W.detach(), b.detach(), final


def sample(R, B, n, dens, gen):
    u = torch.rand(R, B, n, generator=gen)
    mask = (torch.rand(R, B, n, generator=gen) < dens).float()
    return u * mask


def stats(W, cfg):
    """W: (m,n) -> dict of metrics."""
    norms = W.norm(dim=0)                     # (n,)
    Wn = W / norms.clamp_min(1e-9)
    G = (Wn.T @ W)                            # (n,n) : row i = W_hat_i . W_j
    sup = (G ** 2).sum(dim=1) - (G.diagonal() ** 2)   # sum_{j!=i}
    dim_i = norms ** 2 / (G ** 2).sum(dim=1).clamp_min(1e-9)
    rule = cfg.get("feature_count_rule", "frobenius_norm_sq")
    if rule == "count_norm_sq_gt_0.5":
        count = float((norms ** 2 > 0.5).sum())
    elif rule == "count_norm_gt_0.5":
        count = float((norms > 0.5).sum())
    else:
        count = float((norms ** 2).sum())
    return dict(count=count, norms=norms.numpy(), sup=sup.numpy(), dim=dim_i.numpy(),
                frob2=float((norms ** 2).sum()))


def importance_vec(n, base, cfg):
    off = 0 if cfg.get("importance_index_base", "zero_based") == "zero_based" else 1
    return np.array([base ** (i + off) for i in range(n)], dtype=np.float32)


def pick_best(W, b, losses):
    i = int(torch.argmin(losses))
    return W[i], b[i], float(losses[i])
