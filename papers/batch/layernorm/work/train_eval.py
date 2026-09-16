"""Reduced-scale re-implementation of the Layer Normalization (Ba, Kiros & Hinton, 2016) experiments.

CPU-only, compute tier 2. Runs, per seed:
  (A) MNIST MLP 784-W-W-10 (Sec 6.6) with {none, LayerNorm, BatchNorm}, batch 128 and batch 4,
      tracking train NLL and validation/test error -> headline convergence-speed ratio.
  (B) small GRU (rows-of-MNIST sequence task) with/without LN inside the GRU (Appendix Eqs 26-28).
  (C) DRAW-lite (scaled-down DRAW, Sec 6.4) with/without LN on the LSTM cell state
      -> test variational bound (nats/image) on binarized MNIST.
Writes METRICS_OUT json: {seed, split, n_examples, metrics{...}}.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import lncore as L

t_start = time.time()


def log(*a):
    print(f"[{time.time() - t_start:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ MLP experiment


def train_mlp(cfg, norm, batch_size, epochs, data, seed, eval_every, width, shuffle_labels=False):
    torch.manual_seed(seed * 1000 + {"none": 1, "ln": 2, "bn": 3}[norm])
    Xtr, ytr = data["Xtr"], data["ytr"]
    if shuffle_labels:
        perm = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed + 99))
        ytr = ytr[perm]
    n = len(Xtr)
    steps_per_epoch = max(n // batch_size, 1)
    total_steps = steps_per_epoch * epochs
    model = L.MLP(cfg, norm, width, batch_size)
    opt, sch = L.make_optimizer(model, cfg, total_steps)
    model.train()
    curve = []        # (step, val_error, wallclock)
    train_curve = []  # (step, running mean train NLL since last eval, wallclock)
    epoch_train_nll = []
    step, t0 = 0, time.time()
    g = torch.Generator().manual_seed(seed + 7)
    win_loss, win_cnt = 0.0, 0
    for ep in range(epochs):
        order = torch.randperm(n, generator=g)
        run_loss, run_cnt = 0.0, 0
        for b in range(steps_per_epoch):
            idx = order[b * batch_size:(b + 1) * batch_size]
            logits = model(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            L.clip_grads(model, cfg)
            opt.step()
            if sch is not None:
                sch.step()
            li = loss.item()
            run_loss += li * len(idx); run_cnt += len(idx)
            win_loss += li * len(idx); win_cnt += len(idx)
            step += 1
            if step % eval_every == 0 or step == total_steps:
                verr, _ = L.eval_classifier(model, data["Xval"], data["yval"])
                wc = time.time() - t0
                curve.append((step, verr, wc))
                train_curve.append((step, win_loss / max(win_cnt, 1), wc))
                win_loss, win_cnt = 0.0, 0
        epoch_train_nll.append(run_loss / max(run_cnt, 1))
    test_err, test_nll = L.eval_classifier(model, data["Xte"], data["yte"])
    train_err, train_nll_full = L.eval_classifier(model, Xtr, ytr)
    final_train_nll = epoch_train_nll[-1] if cfg["reporting_statistic"] == "final_epoch" else min(epoch_train_nll)
    return dict(curve=curve, train_curve=train_curve, train_nll_epoch_mean=final_train_nll,
                train_nll_full=train_nll_full, train_err=train_err, test_err=test_err, test_nll=test_nll,
                steps=total_steps, epochs=epochs, wallclock=time.time() - t0, epoch_train_nll=epoch_train_nll)


# ------------------------------------------------------------------ GRU experiment


def train_gru(cfg, ln, data, seed, hid, epochs, batch_size, eval_every, shuffle_labels=False):
    torch.manual_seed(seed * 991 + (5 if ln else 3))
    Xtr = data["Xtr"].reshape(-1, 28, 28)
    ytr = data["ytr"]
    if shuffle_labels:
        perm = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed + 99))
        ytr = ytr[perm]
    Xval = data["Xval"].reshape(-1, 28, 28)
    Xte = data["Xte"].reshape(-1, 28, 28)
    model = L.GRUClassifier(cfg, 28, hid, 10, ln)
    n = len(Xtr)
    spe = max(n // batch_size, 1)
    opt, sch = L.make_optimizer(model, cfg, spe * epochs)
    model.train()
    curve, train_curve, step, t0 = [], [], 0, time.time()
    g = torch.Generator().manual_seed(seed + 3)
    ep_losses, win_loss, win_cnt = [], 0.0, 0
    for ep in range(epochs):
        order = torch.randperm(n, generator=g)
        run, cnt = 0.0, 0
        for b in range(spe):
            idx = order[b * batch_size:(b + 1) * batch_size]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            L.clip_grads(model, cfg)
            opt.step()
            if sch is not None:
                sch.step()
            li = loss.item()
            run += li * len(idx); cnt += len(idx)
            win_loss += li * len(idx); win_cnt += len(idx)
            step += 1
            if step % eval_every == 0 or (ep == epochs - 1 and b == spe - 1):
                verr, _ = L.eval_classifier(model, Xval, data["yval"], batch=500)
                wc = time.time() - t0
                curve.append((step, verr, wc))
                train_curve.append((step, win_loss / max(win_cnt, 1), wc))
                win_loss, win_cnt = 0.0, 0
        ep_losses.append(run / max(cnt, 1))
    test_err, _ = L.eval_classifier(model, Xte, data["yte"], batch=500)
    return dict(curve=curve, train_curve=train_curve, train_nll_epoch_mean=ep_losses[-1], test_err=test_err,
                steps=step, wallclock=time.time() - t0, epoch_train_nll=ep_losses)


# ------------------------------------------------------------------ DRAW-lite experiment


def binarize(X, seed=12345):
    """Fixed binarization stand-in for Larochelle & Murray (2011): one fixed Bernoulli draw."""
    rng = np.random.RandomState(seed)
    return (rng.rand(*X.shape) < X).astype(np.float32)


def train_draw(cfg, ln, bx, seed):
    torch.manual_seed(seed * 77 + (11 if ln else 13))
    T, hid, z = int(cfg["scale_draw_glimpses"]), int(cfg["scale_draw_hidden"]), int(cfg["scale_draw_latent"])
    bs, epochs = int(cfg["scale_draw_batch"]), int(cfg["scale_draw_epochs"])
    model = L.DrawLite(cfg, ln, T, hid, z)
    Xtr, Xte = bx["train"], bx["test"]
    n = len(Xtr)
    spe = max(n // bs, 1)
    opt, sch = L.make_optimizer(model, cfg, spe * epochs)
    model.train()
    t0 = time.time()
    g = torch.Generator().manual_seed(seed + 5)
    ep_losses = []
    for ep in range(epochs):
        order = torch.randperm(n, generator=g)
        run, cnt = 0.0, 0
        for b in range(spe):
            xb = Xtr[order[b * bs:(b + 1) * bs]]
            loss, rec, kl = model(xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # DRAW-style clipping for stability
            opt.step()
            if sch is not None:
                sch.step()
            run += loss.item() * len(xb); cnt += len(xb)
        ep_losses.append(run / max(cnt, 1))
        log(f"  draw ln={ln} epoch {ep + 1}/{epochs} train_bound={ep_losses[-1]:.2f}")
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(Xte), 250):
            xb = Xte[i:i + 250]
            loss, rec, kl = model(xb)
            tot += loss.item() * len(xb); cnt += len(xb)
    return dict(test_bound=tot / cnt, train_bound=ep_losses[-1], epoch_bounds=ep_losses,
                wallclock=time.time() - t0)


# ------------------------------------------------------------------ convergence metrics


def conv_ratios(cfg, curve_ln, curve_base, tr_ln, tr_base):
    """Operationalizations of Sec 6.1's '60% of the time to its best validation model'."""
    w = int(cfg["conv_smooth_window"])
    s_ln_c, s_b_c = L.smooth_curve(curve_ln, w), L.smooth_curve(curve_base, w)
    s_ln, w_ln, best_ln = L.steps_to_own_best(s_ln_c)
    s_b, w_b, best_b = L.steps_to_own_best(s_b_c)
    hit_step, hit_wc = L.steps_to_threshold(s_ln_c, best_b)
    thr = float(cfg["train_nll_threshold"])
    tr_ln_step, _ = L.steps_to_threshold(tr_ln, thr)
    tr_b_step, _ = L.steps_to_threshold(tr_base, thr)
    out = {
        "own_best_iterations": 100.0 * s_ln / max(s_b, 1e-9),
        "own_best_wallclock": 100.0 * w_ln / max(w_b, 1e-9),
        "reach_baseline_best_iterations": 100.0 * hit_step / max(s_b, 1e-9),
        "train_nll_threshold_iterations": 100.0 * tr_ln_step / max(tr_b_step, 1e-9),
    }
    detail = dict(steps_ln_best=s_ln, steps_base_best=s_b, best_val_err_ln=best_ln, best_val_err_base=best_b,
                  wallclock_ln_best=w_ln, wallclock_base_best=w_b, steps_ln_reach_base_best=hit_step,
                  train_nll_threshold=thr, steps_ln_train_thresh=tr_ln_step, steps_base_train_thresh=tr_b_step,
                  smooth_window=w)
    mode = cfg["speedup_metric"]
    if mode == "wallclock_for_LN_to_reach_its_own_best_vs_baseline":
        primary = out["own_best_wallclock"]
    elif mode == "iterations_for_LN_to_reach_baseline_best":
        primary = out["reach_baseline_best_iterations"]
    else:
        primary = out["own_best_iterations"]
    return primary, out, detail


# ------------------------------------------------------------------ main


def main():
    seed = int(os.environ.get("SEED", "0"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    raw_cfg = os.environ.get("REPLICATOR_CONFIG", "") or "{}"
    try:
        cfg_user = json.loads(raw_cfg)
    except Exception:
        cfg_user = {}
    if not isinstance(cfg_user, dict):
        cfg_user = {}
    out_path = os.environ.get("METRICS_OUT", "metrics.json")
    cfg = L.resolve_config(cfg_user)
    shuffle_labels = bool(cfg.get("_shuffle_labels", False))

    torch.set_num_threads(min(15, os.cpu_count() or 4))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- tier-2 scaling (recorded in _intermediates.scale)
    width = int(cfg["hidden_width"])
    n_train = int(cfg["scale_mlp_train_subset"])
    ep128 = min(int(cfg["max_epochs"]), int(cfg["scale_mlp_epochs_bz128"]))
    ep4 = min(int(cfg["max_epochs"]), int(cfg["scale_mlp_epochs_bz4"]))
    n_train4 = int(cfg["scale_mlp_train_subset_bz4"])
    n_val = int(cfg["scale_val_subset"])
    eval_every = int(cfg["scale_eval_every_steps"])
    run_bz4 = bool(cfg["run_bz4"])
    run_gru = bool(cfg["run_gru_aux"]) or cfg["replication_task"] in ("small_gru_sequence", "both")
    run_draw = bool(cfg["run_draw_lite"])
    if smoke:
        width, n_train, ep128, ep4, n_train4, n_val, eval_every = 256, 2000, 1, 1, 1000, 500, 10
        cfg["scale_gru_train_subset"], cfg["scale_gru_epochs"], cfg["scale_gru_hidden"] = 1000, 1, 64
        cfg["scale_draw_train_subset"], cfg["scale_draw_epochs"] = 1000, 1
        cfg["scale_draw_glimpses"], cfg["scale_draw_hidden"] = 4, 64

    sp = L.make_splits(cfg, smoke)
    X, y = torch.from_numpy(sp["X"]), torch.from_numpy(sp["y"])
    tr_idx, val_idx = sp["tr_idx"], sp["val_idx"]
    n_test = 2000 if smoke else len(sp["Xte"])
    data = dict(
        Xtr=X[tr_idx[:n_train]], ytr=y[tr_idx[:n_train]],
        Xval=X[val_idx[:n_val]], yval=y[val_idx[:n_val]],
        Xte=torch.from_numpy(sp["Xte"][:n_test]), yte=torch.from_numpy(sp["yte"][:n_test]),
    )
    data4 = dict(data); data4["Xtr"] = data["Xtr"][:n_train4]; data4["ytr"] = data["ytr"][:n_train4]

    metrics = {}
    inter = {"curves": {}, "config_resolved": {k: v for k, v in cfg.items() if not k.startswith("_")}}
    inter["counts"] = dict(mnist_train_available=int(len(tr_idx)), mnist_train_used=int(len(data["Xtr"])),
                           mnist_val_used=int(len(data["Xval"])), mnist_test_used=int(len(data["Xte"])),
                           mnist_train_used_bz4=int(len(data4["Xtr"])), data_root=sp["root"])
    inter["scale"] = {
        "reason": "compute tier 2, CPU only, <=15 min per reproduce.sh run",
        "paper_mnist_train_size": 55000, "used_mnist_train_size": int(len(data["Xtr"])),
        "paper_max_epochs_figure6": int(cfg["max_epochs"]), "used_epochs_bz128": ep128, "used_epochs_bz4": ep4,
        "used_mnist_train_size_bz4": int(len(data4["Xtr"])),
        "paper_hidden_width": 1000, "used_hidden_width": width,
        "val_subset_for_curves": int(len(data["Xval"])), "val_eval_interval_steps": eval_every,
        "paper_draw": "64 glimpses, 256 LSTM units, attention read/write, 200 epochs, 50k train imgs",
        "used_draw": (f"{cfg['scale_draw_glimpses']} glimpses, {cfg['scale_draw_hidden']} LSTM units, "
                      f"full-canvas read/write (no attention), latent {cfg['scale_draw_latent']}, "
                      f"{cfg['scale_draw_epochs']} epochs, {cfg['scale_draw_train_subset']} train imgs, "
                      f"binarization = single fixed Bernoulli draw (Larochelle-Murray file unavailable)"
                      if run_draw else "skipped"),
        "gru_aux": (f"rows-of-MNIST GRU (28 steps x 28 inputs) hidden={cfg['scale_gru_hidden']}, "
                    f"{cfg['scale_gru_epochs']} epochs, {cfg['scale_gru_train_subset']} train imgs; "
                    f"stand-in for the paper's GRU/LSTM tasks" if run_gru else "skipped"),
        "untested_paper_tasks": ["order-embeddings/MSCOCO (data unavailable)",
                                 "skip-thoughts/BookCorpus (data unavailable)",
                                 "attentive reader/CNN QA (data unavailable)",
                                 "handwriting/IAM-OnDB (data unavailable)"],
        "shuffle_labels_control": shuffle_labels,
    }

    # ---- (A) MNIST MLP, batch 128
    res128 = {}
    for norm in ["none", "ln", "bn"]:
        r = train_mlp(cfg, norm, 128, ep128, data, seed, eval_every, width, shuffle_labels)
        res128[norm] = r
        log(f"MLP bz128 norm={norm}: train_nll={r['train_nll_epoch_mean']:.5f} "
            f"test_err={r['test_err']:.4f} steps={r['steps']} {r['wallclock']:.0f}s")
    for norm, tag in [("none", "baseline"), ("ln", "ln"), ("bn", "bn")]:
        r = res128[norm]
        metrics[f"mnist_mlp_bz128_{tag}_train_nll"] = float(r["train_nll_epoch_mean"])
        metrics[f"mnist_mlp_bz128_{tag}_test_accuracy_pct"] = float(100.0 * (1 - r["test_err"]))
        metrics[f"mnist_mlp_bz128_{tag}_test_error_pct"] = float(100.0 * r["test_err"])
        metrics[f"mnist_mlp_bz128_{tag}_val_error_best_pct"] = float(100.0 * min(v for _, v, _ in r["curve"]))
        for e in (1, 3, 5):
            if len(r["epoch_train_nll"]) >= e:
                metrics[f"mnist_mlp_bz128_{tag}_train_nll_at_epoch{e}"] = float(r["epoch_train_nll"][e - 1])
        inter["curves"][f"mlp_bz128_{tag}_val_error"] = [[int(s), round(float(v), 5)] for s, v, _ in r["curve"]]
        inter["curves"][f"mlp_bz128_{tag}_train_nll_window"] = [[int(s), round(float(v), 6)]
                                                                for s, v, _ in r["train_curve"]]
        inter["curves"][f"mlp_bz128_{tag}_train_nll_per_epoch"] = [round(float(v), 6) for v in r["epoch_train_nll"]]

    ratio_mlp, all_mlp, det_mlp = conv_ratios(cfg, res128["ln"]["curve"], res128["none"]["curve"],
                                              res128["ln"]["train_curve"], res128["none"]["train_curve"])
    metrics["convergence_time_ratio_vs_baseline_mnist_mlp"] = float(ratio_mlp)
    for k, v in all_mlp.items():
        metrics[f"convergence_time_ratio_mnist_mlp_{k}"] = float(v)
    inter["convergence_detail_mnist_mlp"] = {k: float(v) for k, v in det_mlp.items()}
    metrics["mnist_mlp_bz128_val_error_gap_ln_minus_baseline_pct"] = float(
        100.0 * (det_mlp["best_val_err_ln"] - det_mlp["best_val_err_base"]))

    # ---- (A2) MNIST MLP, batch 4 (LN vs BN at small batch, Sec 6.6)
    if run_bz4:
        res4 = {}
        for norm, tag in [("none", "baseline"), ("ln", "ln"), ("bn", "bn")]:
            r = train_mlp(cfg, norm, 4, ep4, data4, seed, max(eval_every * 4, 1), width, shuffle_labels)
            res4[norm] = r
            metrics[f"mnist_mlp_bz4_{tag}_train_nll"] = float(r["train_nll_epoch_mean"])
            metrics[f"mnist_mlp_bz4_{tag}_test_accuracy_pct"] = float(100.0 * (1 - r["test_err"]))
            metrics[f"mnist_mlp_bz4_{tag}_test_error_pct"] = float(100.0 * r["test_err"])
            inter["curves"][f"mlp_bz4_{tag}_val_error"] = [[int(s), round(float(v), 5)] for s, v, _ in r["curve"]]
            inter["curves"][f"mlp_bz4_{tag}_train_nll_per_epoch"] = [round(float(v), 6) for v in r["epoch_train_nll"]]
            log(f"MLP bz4 norm={norm}: train_nll={r['train_nll_epoch_mean']:.5f} "
                f"test_err={r['test_err']:.4f} steps={r['steps']} {r['wallclock']:.0f}s")
        ratio4, all4, det4 = conv_ratios(cfg, res4["ln"]["curve"], res4["none"]["curve"],
                                         res4["ln"]["train_curve"], res4["none"]["train_curve"])
        metrics["convergence_time_ratio_vs_baseline_mnist_mlp_bz4"] = float(ratio4)
        inter["convergence_detail_mnist_mlp_bz4"] = {k: float(v) for k, v in det4.items()}

    # ---- (B) GRU stand-in for the paper's recurrent tasks
    ratio_gru = None
    if run_gru:
        gdata = dict(data)
        ns = min(int(cfg["scale_gru_train_subset"]), len(data["Xtr"]))
        gdata["Xtr"], gdata["ytr"] = data["Xtr"][:ns], data["ytr"][:ns]
        gres = {}
        for ln in [False, True]:
            r = train_gru(cfg, ln, gdata, seed, int(cfg["scale_gru_hidden"]), int(cfg["scale_gru_epochs"]),
                          int(cfg["scale_gru_batch"]), max(eval_every // 2, 1), shuffle_labels)
            gres[ln] = r
            tag = "ln" if ln else "baseline"
            metrics[f"gru_seq_{tag}_train_nll"] = float(r["train_nll_epoch_mean"])
            metrics[f"gru_seq_{tag}_test_accuracy_pct"] = float(100.0 * (1 - r["test_err"]))
            metrics[f"gru_seq_{tag}_test_error_pct"] = float(100.0 * r["test_err"])
            inter["curves"][f"gru_{tag}_val_error"] = [[int(s), round(float(v), 5)] for s, v, _ in r["curve"]]
            inter["curves"][f"gru_{tag}_train_nll_per_epoch"] = [round(float(v), 6) for v in r["epoch_train_nll"]]
            log(f"GRU ln={ln}: train_nll={r['train_nll_epoch_mean']:.4f} test_err={r['test_err']:.4f} "
                f"{r['wallclock']:.0f}s")
        ratio_gru, all_gru, det_gru = conv_ratios(cfg, gres[True]["curve"], gres[False]["curve"],
                                                  gres[True]["train_curve"], gres[False]["train_curve"])
        metrics["convergence_time_ratio_vs_baseline_gru"] = float(ratio_gru)
        for k, v in all_gru.items():
            metrics[f"convergence_time_ratio_gru_{k}"] = float(v)
        inter["convergence_detail_gru"] = {k: float(v) for k, v in det_gru.items()}

    # headline metric key (claim oe_ln_speedup_60pct), primary task per config
    primary_task = cfg["replication_task"]
    if primary_task == "small_gru_sequence" and ratio_gru is not None:
        metrics["convergence_time_ratio_vs_baseline"] = float(ratio_gru)
    elif primary_task == "both" and ratio_gru is not None:
        metrics["convergence_time_ratio_vs_baseline"] = float(0.5 * (ratio_mlp + ratio_gru))
    else:
        metrics["convergence_time_ratio_vs_baseline"] = float(ratio_mlp)
    inter["headline_metric_source"] = dict(task=primary_task, speedup_metric=cfg["speedup_metric"],
                                           note="substitution: MSCOCO order-embeddings unavailable; "
                                                "convergence-speed claim tested on the paper's MNIST MLP "
                                                "(Sec 6.6) and a small GRU stand-in")

    # ---- (C) DRAW-lite
    if run_draw:
        ntr = min(int(cfg["scale_draw_train_subset"]), len(data["Xtr"]))
        bx = dict(train=torch.from_numpy(binarize(data["Xtr"][:ntr].numpy())),
                  test=torch.from_numpy(binarize(data["Xte"][:min(2000, len(data["Xte"]))].numpy(), seed=999)))
        inter["counts"]["draw_train_used"] = int(len(bx["train"]))
        inter["counts"]["draw_test_used"] = int(len(bx["test"]))
        dres = {}
        for ln in [False, True]:
            r = train_draw(cfg, ln, bx, seed)
            dres[ln] = r
            tag = "draw_ln" if ln else "draw_baseline"
            metrics[f"test_variational_bound_negative_log_likelihood_nats_{tag}"] = float(r["test_bound"])
            metrics[f"{tag}_train_variational_bound_nats"] = float(r["train_bound"])
            inter["curves"][f"{tag}_train_bound_per_epoch"] = [round(float(v), 3) for v in r["epoch_bounds"]]
            log(f"DRAW-lite ln={ln}: test_bound={r['test_bound']:.2f} {r['wallclock']:.0f}s")
        # Both DRAW claims share this metric name; we expose the LN variant here and both variants
        # under suffixed keys. Absolute nats are NOT comparable to the paper's 82.09/82.36 because the
        # model is scaled down (see _intermediates.scale); only the LN-vs-baseline gap is informative.
        metrics["test_variational_bound_negative_log_likelihood_nats"] = float(dres[True]["test_bound"])
        metrics["draw_bound_gap_ln_minus_baseline_nats"] = float(dres[True]["test_bound"] - dres[False]["test_bound"])
        inter["draw_note"] = ("scaled-down DRAW; paper gap (LN - baseline) = 82.09 - 82.36 = -0.27 nats; "
                              "we report our own gap, absolute values not comparable")

    # ---- claims whose data source is unavailable (MSCOCO order-embeddings, Table 2 rows
    # 'Sym [Vendrov et al., 2016]' / 'OE ...'): reported as NaN = Untested, per the plan's
    # substitution table ("mscoco: UNAVAILABLE"). Keys are emitted so the contract is explicit.
    for k in ("recall_at_1", "recall_at_5", "recall_at_10", "mean_rank"):
        metrics[k] = float("nan")
    inter["untested_metrics"] = {
        "keys": ["recall_at_1", "recall_at_5", "recall_at_10", "mean_rank"],
        "reason": ("MSCOCO images/captions and the pre-trained VGG 10-crop features are not available "
                   "offline; the Table 2 retrieval numbers (including the copied Vendrov et al. rows) "
                   "cannot be re-implemented. Reported as NaN (Untested) rather than substituted."),
    }
    metrics["_intermediates"] = inter
    payload = {"seed": seed, "split": cfg["split"], "n_examples": int(len(data["Xte"])), "metrics": metrics}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    log("wrote", out_path, f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
