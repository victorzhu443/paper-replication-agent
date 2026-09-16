"""Reduced-scale CIFAR-10 replication of He et al. (2015) Section 4.2.

Trains depth-matched ResNet / plain nets on a class-balanced subset and reports
top-1 test error plus the direction (residual vs plain) contrasts.
"""
import json, os, time, math
import numpy as np
import torch, torch.nn as nn
from resnet_cifar import CifarNet

DATA = os.environ.get("CIFAR_NPZ",
                      "/Users/vzhu/Developer/research-copy/data_cache/cifar10/cifar10.npz")

DEFAULTS = {
    # ambiguity keys (spec defaults)
    "lr_schedule": "as_paper",
    "augmentation": "as_paper",
    "batch_vs_steps": "steps",
    "mixed_precision": "fp32",
    "split": "test",
    "train_split_size": "full_50k",
    "train_subset_selection": "class_balanced_random_seeded",
    "n_seeds": "3",
    "shortcut_option_a_impl": "stride2_slice_then_zero_pad",
    "shortcut_zero_pad_placement": "append_end",
    "init": "he_normal_fan_out",
    "conv_bias": "no_bias",
    "input_normalization": "per_pixel_mean_only",
    "bn_config": "framework_default",
    "lr_warmup": "depth>=110_only",
    "plain_lr_warmup": "no_warmup",
    "plain110_error_metric": "test_error",
    "checkpoint_selection": "final",
    "wd_scope": "all_params",
    "optimizer_nesterov": "classical",
    "determinism": "seeded_deterministic",
    # reduced-scale knobs (replication-only)
    "train_subset_size": 10000,
    "batch_size": 128,
    "steps": 400,
    "base_lr": 0.1,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "depths": [20, 56],
    "eval_subset": 10000,
    "eval_batch": 500,
}


def get_cfg():
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(os.environ.get("REPLICATOR_CONFIG") or "{}"))
    return cfg


def load_data(cfg, seed):
    z = np.load(DATA)
    Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    rng = np.random.default_rng(1234)  # fixed split rng (independent of run seed)
    if cfg["split"] == "validation" or cfg["train_split_size"] == "45k_only":
        perm = rng.permutation(len(Xtr))
        tr_idx, val_idx = perm[:45000], perm[45000:]
        if cfg["split"] == "validation":
            Xte, yte = Xtr[val_idx], ytr[val_idx]
        Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]
    n = int(cfg["train_subset_size"])
    srng = np.random.default_rng(1000 + seed)
    if n < len(Xtr):
        sel = cfg["train_subset_selection"]
        if sel == "class_balanced_random_seeded":
            per = n // 10
            idx = np.concatenate([srng.permutation(np.where(ytr == c)[0])[:per] for c in range(10)])
            idx = srng.permutation(idx)
        elif sel == "first_n_of_shuffled":
            idx = srng.permutation(len(Xtr))[:n]
        else:
            idx = srng.integers(0, len(Xtr), size=n)
        Xtr, ytr = Xtr[idx], ytr[idx]
    m = int(cfg["eval_subset"])
    if m < len(Xte):
        Xte, yte = Xte[:m], yte[:m]
    return Xtr, ytr, Xte, yte


def normalize(Xtr, Xte, mode):
    a = Xtr.astype(np.float32) / 255.0
    b = Xte.astype(np.float32) / 255.0
    if mode == "per_channel_mean_only":
        mu = a.mean(axis=(0, 2, 3), keepdims=True)
        a, b = a - mu, b - mu
    elif mode == "per_channel_mean_and_std":
        mu = a.mean(axis=(0, 2, 3), keepdims=True)
        sd = a.std(axis=(0, 2, 3), keepdims=True)
        a, b = (a - mu) / sd, (b - mu) / sd
    else:  # per_pixel_mean_only (paper)
        mu = a.mean(axis=0, keepdims=True)
        a, b = a - mu, b - mu
    return torch.from_numpy(a), torch.from_numpy(b)


def augment(batch, gen, mode):
    if mode == "none":
        return batch
    N = batch.shape[0]
    x = torch.nn.functional.pad(batch, (4, 4, 4, 4))  # zero padding, 4 px each side
    ij = torch.randint(0, 9, (N, 2), generator=gen)
    out = torch.empty_like(batch)
    for k in range(N):
        i, j = int(ij[k, 0]), int(ij[k, 1])
        out[k] = x[k, :, i:i + 32, j:j + 32]
    flip = torch.rand(N, generator=gen) < 0.5
    out[flip] = torch.flip(out[flip], dims=[3])
    return out


def lr_at(step, total, cfg, warmup_active):
    base = float(cfg["base_lr"])
    if warmup_active:
        return 0.01
    sched = cfg["lr_schedule"]
    if sched == "constant":
        return base
    if sched == "cosine":
        return base * 0.5 * (1 + math.cos(math.pi * step / total))
    # as_paper / step: /10 at 50% and 75% of the budget (32k/48k of 64k)
    if step >= 0.75 * total:
        return base / 100
    if step >= 0.5 * total:
        return base / 10
    return base


@torch.no_grad()
def evaluate(model, X, y, bs):
    model.eval()
    wrong = 0
    for i in range(0, len(X), bs):
        logits = model(X[i:i + bs])
        wrong += int((logits.argmax(1) != y[i:i + bs]).sum())
    model.train()
    return 100.0 * wrong / len(X)


def train_one(name, depth, residual, cfg, seed, Xtr, ytr, Xte, yte, log):
    torch.manual_seed(seed + 7 * depth + (0 if residual else 1))
    gen = torch.Generator().manual_seed(seed * 31 + depth + (0 if residual else 1))
    mcfg = {k: cfg[k] for k in ["conv_bias", "shortcut_option_a_impl",
                                "shortcut_zero_pad_placement", "init"]}
    bnc = cfg["bn_config"]
    mcfg["bn_kw"] = ({"momentum": 0.1, "eps": 1e-5} if bnc == "momentum_0.9_eps_1e-5_affine"
                     else {"momentum": 0.01, "eps": 1e-3} if bnc == "momentum_0.99_eps_1e-3_affine"
                     else {})
    model = CifarNet(depth, residual, mcfg)
    nparams = sum(p.numel() for p in model.parameters())
    if cfg["wd_scope"] == "conv_fc_weights_only":
        decay = [p for n_, p in model.named_parameters() if p.dim() > 1]
        nodecay = [p for n_, p in model.named_parameters() if p.dim() <= 1]
        groups = [{"params": decay, "weight_decay": float(cfg["weight_decay"])},
                  {"params": nodecay, "weight_decay": 0.0}]
    else:
        groups = [{"params": list(model.parameters()), "weight_decay": float(cfg["weight_decay"])}]
    opt = torch.optim.SGD(groups, lr=float(cfg["base_lr"]), momentum=float(cfg["momentum"]),
                          nesterov=(cfg["optimizer_nesterov"] == "nesterov"))
    lossf = nn.CrossEntropyLoss()
    bs = int(cfg["batch_size"])
    total = int(cfg["steps"])
    if cfg["batch_vs_steps"] == "epochs":
        total = max(1, int(cfg.get("epochs", 5)) * len(Xtr) // bs)
    warmup = ((cfg["lr_warmup"] == "all_depths" or
               (cfg["lr_warmup"] == "depth>=110_only" and depth >= 110)) if residual
              else (cfg["plain_lr_warmup"] == "same_as_resnet110" and depth >= 110))
    warm_done = not warmup
    order = torch.randperm(len(Xtr), generator=gen)
    pos = 0
    t0 = time.time()
    best = None
    train_err_ema = None
    curve = []
    for step in range(total):
        if pos + bs > len(Xtr):
            order = torch.randperm(len(Xtr), generator=gen); pos = 0
        idx = order[pos:pos + bs]; pos += bs
        xb = augment(Xtr[idx], gen, cfg["augmentation"])
        yb = ytr[idx]
        lr = lr_at(step, total, cfg, warmup and not warm_done)
        for g in opt.param_groups:
            g["lr"] = lr
        logits = model(xb)
        loss = lossf(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        berr = 100.0 * float((logits.argmax(1) != yb).float().mean())
        train_err_ema = berr if train_err_ema is None else 0.9 * train_err_ema + 0.1 * berr
        if warmup and not warm_done and train_err_ema < 80.0:
            warm_done = True
        if (step + 1) % max(1, total // 5) == 0 or step == total - 1:
            e = evaluate(model, Xte, yte, int(cfg["eval_batch"]))
            curve.append({"step": step + 1, "test_error": e, "train_err_ema": train_err_ema})
            best = e if best is None else min(best, e)
            log(f"  {name} step {step+1}/{total} lr {lr:.3f} train_err_ema {train_err_ema:.1f} "
                f"test_err {e:.2f} ({time.time()-t0:.0f}s)")
    final = evaluate(model, Xte, yte, int(cfg["eval_batch"]))
    # training-set error (no augmentation) on the training subset
    tr_err = evaluate(model, Xtr[:5000], ytr[:5000], int(cfg["eval_batch"]))
    rep = final if cfg["checkpoint_selection"] == "final" else min(best, final)
    return {"test_error": rep, "final_test_error": final, "best_test_error": min(best, final),
            "train_error": tr_err, "params": nparams, "steps": total,
            "seconds": time.time() - t0, "curve": curve}


def main():
    seed = int(os.environ.get("SEED", "0"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    cfg = get_cfg()
    if smoke:
        cfg.setdefault("_smoke", True)
        cfg["train_subset_size"] = min(int(cfg["train_subset_size"]), 2000)
        cfg["steps"] = min(int(cfg["steps"]), 20)
        cfg["eval_subset"] = min(int(cfg["eval_subset"]), 2000)
        cfg["depths"] = [20]
    torch.manual_seed(seed); np.random.seed(seed)
    if cfg["determinism"] == "seeded_deterministic":
        torch.use_deterministic_algorithms(False)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "15")))

    def log(m):
        print(m, flush=True)

    Xtr_raw, ytr, Xte_raw, yte = load_data(cfg, seed)
    Xtr, Xte = normalize(Xtr_raw, Xte_raw, cfg["input_normalization"])
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
    log(f"train {tuple(Xtr.shape)} eval {tuple(Xte.shape)} cfg steps={cfg['steps']}")

    results = {}
    for depth in [int(d) for d in cfg["depths"]]:
        for residual in [True, False]:
            name = f"{'resnet' if residual else 'plain'}{depth}"
            results[name] = train_one(name, depth, residual, cfg, seed, Xtr, ytr_t, Xte, yte_t, log)

    m = {}
    for name, r in results.items():
        m[f"top1_error_{name}"] = r["test_error"]
        m[f"train_error_{name}"] = r["train_error"]
    d0 = int(cfg["depths"][0])
    m["top1_error"] = results[f"resnet{d0}"]["test_error"]
    for depth in [int(d) for d in cfg["depths"]]:
        m[f"resnet_minus_plain_{depth}"] = (results[f"resnet{depth}"]["test_error"]
                                            - results[f"plain{depth}"]["test_error"])
    depths = [int(d) for d in cfg["depths"]]
    if len(depths) > 1:
        a, b = depths[0], depths[-1]
        m[f"plain_deep_minus_shallow_test"] = (results[f"plain{b}"]["test_error"]
                                               - results[f"plain{a}"]["test_error"])
        m[f"plain_deep_minus_shallow_train"] = (results[f"plain{b}"]["train_error"]
                                                - results[f"plain{a}"]["train_error"])
        m[f"resnet_deep_minus_shallow_test"] = (results[f"resnet{b}"]["test_error"]
                                                - results[f"resnet{a}"]["test_error"])
    m["_intermediates"] = {
        "scale": {
            "train_subset_size": int(cfg["train_subset_size"]),
            "steps": int(cfg["steps"]),
            "batch_size": int(cfg["batch_size"]),
            "depths": depths,
            "eval_subset": int(cfg["eval_subset"]),
            "paper_budget": "50000 images, 64000 iterations, batch 128, depths 20/32/44/56/110/1202",
            "note": "compute tier 2, CPU only: subset + short step budget; absolute errors are "
                    "far above the paper's 8.75%; only the residual-vs-plain direction is tested",
        },
        "n_train": int(Xtr.shape[0]), "n_eval": int(Xte.shape[0]),
        "per_model": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    out = {"seed": seed, "split": cfg["split"], "n_examples": int(Xte.shape[0]), "metrics": m}
    path = os.environ.get("METRICS_OUT", "metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {path}: " + json.dumps({k: v for k, v in m.items() if k != '_intermediates'}))


if __name__ == "__main__":
    main()
