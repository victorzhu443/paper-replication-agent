"""Toy Models of Superposition (Elhage et al. 2022) -- re-implementation.

Trains ReLU-output tied-weight toy autoencoders on synthetic sparse features and measures
  D*  = m / ||W||_F^2   (dimensions per feature) across a sparsity sweep,
  D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2 (per-feature dimensionality) for small geometries,
  adversarial vulnerability ratio under the analytic optimal L2 attack.
Labels here are reconstruction targets; _shuffle_labels permutes the targets within each batch.
"""
import json, math, os, time
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", "15")))

CFG_DEFAULTS = {
    "lr_schedule": "constant", "augmentation": "none", "batch_vs_steps": "steps",
    "mixed_precision": "fp32", "split": "test", "optimizer": "adam", "learning_rate": 1e-3,
    "training_steps": 10000, "batch_size": 1024, "data_resampling": "fresh_each_step",
    "bias_sign": "plus_b", "bias_init": "zeros", "weight_init": "pytorch_default_kaiming_uniform",
    "n_restarts": "as_paper_where_stated", "n3m2_restarts": "same_as_n2m1_ten_drop_worst",
    "phase_grid_resolution": "40x40",
    "phase_classification_thresholds": "continuous_2d_colormap_no_thresholds",
    "pentagon_threshold_direction": "relative_density_0.4x_varied_feature_sparser",
    "adv_baseline_model": "dense_end_model_of_same_sweep",
    "adv_attack_scale": "0.1_of_average_input_l2_norm",
    "sparsity_grid": "20_log_spaced_1_to_10", "eval_samples": 100000,
    "importance_indexing": "zero_indexed", "dstar_aggregation": "lowest_loss_of_restarts",
    "energy_jump_sparsity": "any_sparsity_where_all_features_become_digons", "n_seeds": 5,
}


def make_opt(params, cfg):
    lr = float(cfg["learning_rate"])
    o = cfg["optimizer"]
    if o == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if o == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


def init_W(K, m, n, cfg, gen):
    wi = cfg["weight_init"]
    if wi == "small_gaussian_0.01":
        W = torch.randn(K, m, n, generator=gen) * 0.01
    elif wi == "xavier_uniform":
        a = math.sqrt(6.0 / (n + m))
        W = (torch.rand(K, m, n, generator=gen) * 2 - 1) * a
    else:  # pytorch default kaiming uniform (fan_in = n)
        a = 1.0 / math.sqrt(n)
        W = (torch.rand(K, m, n, generator=gen) * 2 - 1) * a
    return W


def sample(K, B, n, dens, gen):
    """dens: tensor [K,1,1] feature density (1-S)."""
    mask = (torch.rand(K, B, n, generator=gen) < dens).float()
    return mask * torch.rand(K, B, n, generator=gen)


def forward(x, W, b, cfg):
    h = torch.bmm(x, W.transpose(1, 2))
    pre = torch.bmm(h, W)
    pre = pre + b if cfg["bias_sign"] == "plus_b" else pre - b
    return torch.relu(pre)


def train(n, m, dens, cfg, seed, steps, shuffle=False, imp=None):
    """dens: list of feature densities, one per parallel model. Returns W,b,losses."""
    gen = torch.Generator().manual_seed(seed)
    K = len(dens)
    d = torch.tensor(dens, dtype=torch.float32).view(K, 1, 1)
    W = init_W(K, m, n, cfg, gen).requires_grad_(True)
    b0 = torch.zeros(K, 1, n) if cfg["bias_init"] != "small_random" else 0.01 * torch.randn(K, 1, n, generator=gen)
    b = b0.clone().requires_grad_(True)
    if cfg["bias_init"] == "no_bias":
        params = [W]
    else:
        params = [W, b]
    opt = make_opt(params, cfg)
    B = int(cfg["batch_size"])
    I = torch.ones(1, 1, n) if imp is None else torch.tensor(imp, dtype=torch.float32).view(1, 1, n)
    sch = None
    if cfg["lr_schedule"] == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    elif cfg["lr_schedule"] == "step":
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 3), gamma=0.3)
    for t in range(steps):
        x = sample(K, B, n, d, gen)
        y = x
        if shuffle:  # shuffle targets within each batch (per parallel model)
            perm = torch.randperm(B, generator=gen)
            y = x[:, perm, :]
        out = forward(x, W, b, cfg)
        loss = (I * (out - y) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if sch is not None:
            sch.step()
    with torch.no_grad():
        losses = eval_loss(W, b, d, cfg, I, n, gen, shuffle)
    return W.detach(), b.detach(), losses


def eval_loss(W, b, d, cfg, I, n, gen, shuffle=False, nsamp=8192):
    K = W.shape[0]
    tot = torch.zeros(K)
    done = 0
    B = min(2048, nsamp)
    while done < nsamp:
        x = sample(K, B, n, d, gen)
        y = x
        if shuffle:
            y = x[:, torch.randperm(B, generator=gen), :]
        out = forward(x, W, b, cfg)
        tot += (I * (out - y) ** 2).sum(-1).mean(1) * B
        done += B
    return (tot / done).numpy()


def feature_dimensionality(W):
    """W: [m,n]. Returns per-feature D_i."""
    Wn = W / (W.norm(dim=0, keepdim=True) + 1e-9)
    G = (Wn.T @ W) ** 2           # [n,n]  (W_i_hat . W_j)^2
    num = (W.norm(dim=0)) ** 2
    return (num / (G.sum(1) + 1e-12)).numpy()


def attack_delta(W, b, dens_k, cfg, n, gen, nsamp=4096):
    """Loss increase (L' - L) of models W,b when inputs are drawn at density dens_k
    and perturbed by the analytic optimal per-feature L2 attack (worst feature)."""
    K = W.shape[0]
    d = torch.tensor(dens_k, dtype=torch.float32).view(K, 1, 1)
    I = torch.ones(1, 1, n)
    WtW = torch.bmm(W.transpose(1, 2), W)
    dirs = WtW / (WtW.norm(dim=2, keepdim=True) + 1e-9)   # row i = attack direction for feature i
    x0 = sample(K, 2048, n, d, gen)
    avg_norm = x0.norm(dim=2).mean(1)                     # average input L2 norm at this density
    scale = 0.1 * avg_norm
    if cfg["adv_attack_scale"] == "fixed_absolute_0.1":
        scale = torch.full((K,), 0.1)
    x = sample(K, nsamp, n, d, gen)
    with torch.no_grad():
        clean = (I * (forward(x, W, b, cfg) - x) ** 2).sum(-1).mean(1)
        worst = clean.clone()
        for i in range(n):
            delta = scale.view(K, 1, 1) * dirs[:, i, :].unsqueeze(1)
            l = (I * (forward(x + delta, W, b, cfg) - x) ** 2).sum(-1).mean(1)
            worst = torch.maximum(worst, l)
    return clean.numpy(), (worst - clean).numpy()


def baseline_model(W, b, cfg, n, m):
    """Non-superposition reference model for the adversarial ratio."""
    opt = cfg["adv_baseline_model"]
    if opt == "dense_end_model_of_same_sweep":
        return W[0:1].clone(), b[0:1].clone()
    # hand-constructed / linear: identity on the top-m features, zero elsewhere
    W0 = torch.zeros(1, m, n)
    for i in range(m):
        W0[0, i, i] = 1.0
    return W0, torch.zeros(1, 1, n)


def adversarial_curve(W, b, dens, cfg, n, m, seed):
    gen = torch.Generator().manual_seed(seed + 777)
    _, delta = attack_delta(W, b, dens, cfg, n, gen)
    Wb, bb = baseline_model(W, b, cfg, n, m)
    K = W.shape[0]
    Wb = Wb.expand(K, -1, -1).contiguous(); bb = bb.expand(K, -1, -1).contiguous()
    _, delta0 = attack_delta(Wb, bb, dens, cfg, n, gen)
    return delta / np.maximum(delta0, 1e-12)


def cluster_mode(vals, tol=0.03):
    """Value of the densest cluster (the 'sticky' plateau)."""
    vals = np.asarray(vals)
    best, bestc = float(vals[0]), -1
    for v in vals:
        sel = vals[np.abs(vals - v) <= tol]
        if len(sel) > bestc:
            bestc, best = len(sel), float(sel.mean())
    return best, bestc


def geometry_experiments(cfg, seed, steps, restarts, shuffle):
    """Small uniform ReLU-output models whose optimal geometry is a known polytope."""
    specs = {"digon": (2, 1), "triangle": (3, 2), "pentagon": (5, 2),
             "tetrahedron": (4, 3), "square_antiprism": (8, 3)}
    out = {}
    for j, (name, (n, m)) in enumerate(specs.items()):
        dens = [0.05] * restarts
        W, b, losses = train(n, m, dens, cfg, seed + 101 * (j + 1), steps, shuffle=shuffle)
        k = int(np.argmin(losses))
        Ds = feature_dimensionality(W[k])
        out[name] = float(np.mean(Ds))
    return out


def run(cfg, seed, scale, smoke, shuffle, timeout_s):
    t0 = time.time()
    n, m = 400, 30
    steps = int(cfg["training_steps"])
    ngrid = {"20_log_spaced_1_to_10": 20, "40_log_spaced_1_to_10": 40,
             "as_paper_discrete_7_points": 7}[cfg["sparsity_grid"]]
    geo_steps, geo_restarts = 4000, 8
    if smoke:
        steps, ngrid, geo_steps, geo_restarts = 200, 6, 300, 3
        cfg = dict(cfg); cfg["batch_size"] = 256
    else:
        steps = max(200, int(round(steps * scale)))
        ngrid = max(4, int(round(ngrid * scale)))
        geo_steps = max(300, int(round(geo_steps * scale)))
        geo_restarts = max(2, int(round(geo_restarts * scale)))

    inv = np.geomspace(1.0, 10.0, ngrid)          # 1/(1-S)
    dens = (1.0 / inv).tolist()                   # feature density 1-S
    W, b, losses = train(n, m, dens, cfg, seed, steps, shuffle=shuffle)
    frob2 = (W ** 2).sum(dim=(1, 2)).numpy()
    dstar = m / frob2                              # dimensions per feature
    dense_end = float(dstar[0])                    # 1/(1-S) = 1
    sticky, count = cluster_mode(dstar[inv >= 1.5], tol=0.03)

    # untrained (no-skill) reference
    g = torch.Generator().manual_seed(seed + 31337)
    W0 = init_W(1, m, n, cfg, g)
    dstar_rand = float(m / (W0 ** 2).sum().item())

    # adversarial vulnerability
    ratio = adversarial_curve(W, b, dens, cfg, n, m, seed)
    adv_ratio = float(np.max(ratio))

    geo = geometry_experiments(cfg, seed, geo_steps, geo_restarts, shuffle)

    metrics = {
        "dimensions_per_feature": sticky,
        "dimensions_per_feature_sticky_half": sticky,
        "dimensions_per_feature_dense_end": dense_end,
        "dimensions_per_feature_random_init_baseline": dstar_rand,
        "adversarial_vulnerability_ratio": adv_ratio,
        "feature_dimensionality": geo["digon"],
        "chance_level": dstar_rand,
    }
    for k, v in geo.items():
        metrics["feature_dimensionality_" + k] = v
    inter = {
        "scale": {"n": n, "m": m, "training_steps": steps, "batch_size": int(cfg["batch_size"]),
                  "sparsity_grid_points": ngrid, "restarts_dstar": 1,
                  "geometry_steps": geo_steps, "geometry_restarts": geo_restarts,
                  "note": "paper n=400,m=30 uniform ReLU-output model; restarts reduced to 1 for "
                          "the D* sweep (paper states none), geometry fits reduced"},
        "dstar_curve": [[float(a), float(c)] for a, c in zip(inv, dstar)],
        "sweep_loss": [float(x) for x in losses],
        "adv_ratio_curve": [float(x) for x in ratio],
        "sticky_cluster_size": count,
        "wall_s": time.time() - t0,
    }
    n_examples = int(steps * int(cfg["batch_size"]) * ngrid)
    return metrics, inter, n_examples


def main():
    seed = int(os.environ.get("SEED", "0"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    scale = float(os.environ.get("SCALE", "1.0"))
    timeout_s = float(os.environ.get("RUN_TIMEOUT_S", "1080"))
    user = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}")
    cfg = dict(CFG_DEFAULTS)
    for k, v in user.items():
        if k in cfg:
            cfg[k] = v
    shuffle = bool(user.get("_shuffle_labels", False))
    torch.manual_seed(seed); np.random.seed(seed)
    metrics, inter, n_ex = run(cfg, seed, scale, smoke, shuffle, timeout_s)
    metrics["_intermediates"] = inter
    out = {"seed": seed, "split": cfg["split"], "n_examples": n_ex,
           "scale": scale,
           "shuffled": shuffle,
           "matched_scale": (not smoke) and scale == 1.0 and int(cfg["training_steps"]) >= 10000,
           "metrics": metrics}
    path = os.environ.get("METRICS_OUT", "metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "_intermediates"}, indent=1))


if __name__ == "__main__":
    main()
