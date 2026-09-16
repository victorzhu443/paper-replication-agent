"""World Models (Ha & Schmidhuber 2018) — reduced-scale mechanics replication.

Three staged checks in a tiny pixel env (CarRacing is out of reach on CPU):
  1. ConvVAE reconstruction loss falls on held-out frames.
  2. MDN-RNN next-latent NLL falls below a reference marginal-Gaussian predictor.
  3. CMA-ES linear controller on [z, h] beats a random policy.
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

from wm_env import BatchCatcher, all_state_frames, N_STATES

torch.set_num_threads(max(1, min(15, os.cpu_count() or 1)))


# ---------------------------------------------------------------- models
class ConvVAE(nn.Module):
    """Paper Fig. 22 topology (4 stride-2 convs / 4 deconvs), channels scaled down."""

    def __init__(self, z_dim=32, ch=(8, 16, 32, 64)):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.enc = nn.Sequential(
            nn.Conv2d(3, c1, 4, 2), nn.ReLU(),
            nn.Conv2d(c1, c2, 4, 2), nn.ReLU(),
            nn.Conv2d(c2, c3, 4, 2), nn.ReLU(),
            nn.Conv2d(c3, c4, 4, 2), nn.ReLU())
        self.fc_mu = nn.Linear(c4 * 2 * 2, z_dim)
        self.fc_lv = nn.Linear(c4 * 2 * 2, z_dim)
        self.fc_dec = nn.Linear(z_dim, c4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c4 * 4, c3, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(c3, c2, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(c2, c1, 6, 2), nn.ReLU(),
            nn.ConvTranspose2d(c1, 3, 6, 2), nn.Sigmoid())
        self.z_dim = z_dim

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        h = self.fc_dec(z).unsqueeze(-1).unsqueeze(-1)
        return self.dec(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.decode(z), mu, lv


class MDNRNN(nn.Module):
    def __init__(self, z_dim=32, a_dim=1, hidden=64, k=5):
        super().__init__()
        self.lstm = nn.LSTM(z_dim + a_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3 * k * z_dim)
        self.z_dim, self.k, self.hidden = z_dim, k, hidden

    def forward(self, z, a, state=None):
        out, state = self.lstm(torch.cat([z, a], -1), state)
        p = self.head(out)
        B, T, _ = p.shape
        p = p.view(B, T, self.k, 3, self.z_dim)
        logpi, mu, logsig = p[..., 0, :], p[..., 1, :], p[..., 2, :]
        logpi = logpi - torch.logsumexp(logpi, dim=2, keepdim=True)
        logsig = torch.clamp(logsig, -7.0, 3.0)
        return logpi, mu, logsig, state


def mdn_nll(logpi, mu, logsig, target):
    t = target.unsqueeze(2)
    lp = -0.5 * ((t - mu) / torch.exp(logsig)) ** 2 - logsig - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(logpi + lp, dim=2).sum(-1).mean()


# ---------------------------------------------------------------- CMA-ES (separable)
class SepCMAES:
    def __init__(self, n, sigma=0.1, popsize=32, seed=0):
        self.n, self.sigma, self.lam = n, sigma, popsize
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros(n)
        self.mu = popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.ds = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = 4.0 / (n + 4)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.D = np.ones(n)          # diagonal std
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    def ask(self):
        self.zs = self.rng.standard_normal((self.lam, self.n))
        self.x = self.mean + self.sigma * self.zs * self.D
        return self.x

    def tell(self, fitness):
        order = np.argsort(-np.asarray(fitness))   # maximise
        xs = self.x[order[:self.mu]]
        zs = self.zs[order[:self.mu]]
        old = self.mean
        self.mean = (self.w[:, None] * xs).sum(0)
        zmean = (self.w[:, None] * zs).sum(0)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * zmean
        hs = np.linalg.norm(self.ps) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.mean - old) / self.sigma
        C = self.D ** 2
        rank_mu = (self.w[:, None] * ((xs - old) / self.sigma) ** 2).sum(0)
        C = (1 - self.c1 - self.cmu) * C + self.c1 * self.pc ** 2 + self.cmu * rank_mu
        self.D = np.sqrt(np.maximum(C, 1e-12))
        self.sigma *= math.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-4, 10.0))


# ---------------------------------------------------------------- helpers
def to_torch(frames):
    x = torch.from_numpy(frames).float()
    if x.max() > 1.5:
        x = x / 255.0
    return x.permute(0, 3, 1, 2).contiguous()


def collect_rollouts(n_roll, T, seed, batch=100, frame_skip=1):
    """Random-policy rollouts; returns frames uint8 (n_roll,T,64,64,3), actions (n_roll,T)."""
    rng = np.random.default_rng(seed)
    frames, acts = [], []
    done_n = 0
    while done_n < n_roll:
        b = min(batch, n_roll - done_n)
        env = BatchCatcher(b, seed=int(rng.integers(1 << 30)), max_steps=T)
        obs = env.reset()
        fb = np.zeros((b, T, 64, 64, 3), dtype=np.uint8)
        ab = np.zeros((b, T), dtype=np.int64)
        a = rng.integers(0, 3, size=b)
        for t in range(T):
            fb[:, t] = (obs * 255).astype(np.uint8)
            if t % frame_skip == 0:
                a = rng.integers(0, 3, size=b)   # uniform over action space
            ab[:, t] = a
            obs, r, d = env.step(a)
        frames.append(fb)
        acts.append(ab)
        done_n += b
    return np.concatenate(frames), np.concatenate(acts)


def act_to_cont(a):
    return (np.asarray(a).astype(np.float32) - 1.0)   # {-1,0,1}


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    seed = int(os.environ.get("SEED", "0"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    scale = float(os.environ.get("SCALE", "1.0"))
    cfg = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}")
    out_path = os.environ.get("METRICS_OUT", "metrics.json")
    shuffle = bool(cfg.get("_shuffle_labels", False))

    # ---- ambiguity config keys (spec defaults)
    C = dict(
        lr_schedule=cfg.get("lr_schedule", "constant"),
        augmentation=cfg.get("augmentation", "none"),
        batch_vs_steps=cfg.get("batch_vs_steps", "steps"),
        mixed_precision=cfg.get("mixed_precision", "fp32"),
        split=cfg.get("split", "test"),
        optimizer=cfg.get("optimizer", "adam_1e-4"),
        batch_size=int(cfg.get("batch_size", 100)),
        kl_weight=cfg.get("kl_weight", "kl_tolerance_0.5_per_dim"),
        recon_reduction=cfg.get("recon_reduction", "sum_over_pixels"),
        rnn_seq_len=cfg.get("rnn_seq_len", 1000),
        temperature_carracing=cfg.get("temperature_carracing", "not_applicable_deterministic_h"),
        cma_sigma_init=float(cfg.get("cma_sigma_init", 0.1)),
        n_generations=int(cfg.get("n_generations", 100)),
        n_random_rollouts=int(cfg.get("n_random_rollouts", 500)),
        frame_skip=int(cfg.get("frame_skip", 1)),
        n_seeds=int(cfg.get("n_seeds", 3)),
        environment=cfg.get("environment", "tiny_gridworld_pixels"),
        controller_state_input=cfg.get("controller_state_input", "h_only"),
        controller_bias=cfg.get("controller_bias", "with_bias"),
        n_eval_episodes=int(cfg.get("n_eval_episodes", 100)),
        mdnrnn_baseline=cfg.get("mdnrnn_baseline", "marginal_gaussian_fit_to_z"),
        dispersion_measure=cfg.get("dispersion_measure", "std_across_trials"),
        baseline_n_trials=cfg.get("baseline_n_trials", "unknown_as_quoted"),
        n_virtual_rollouts=int(cfg.get("n_virtual_rollouts", 100)),
        random_policy=cfg.get("random_policy", "uniform_over_action_space"),
        z_encode_mode=cfg.get("z_encode_mode", "sample"),
        action_mapping=cfg.get("action_mapping", "clip_negative_to_zero"),
        dream_init=cfg.get("dream_init", "random_real_frame_latent"),
        dream_max_steps=int(cfg.get("dream_max_steps", 2100)),
        fitness_seed_policy=cfg.get("fitness_seed_policy", "resample_each_generation"),
    )

    # ---- reduced-scale compute knobs (tier 2)
    T = 60                                    # env episode length
    z_dim, hidden = 32, 64
    pop = int(cfg.get("_cma_popsize", 64))
    rolls_per_cand = int(cfg.get("_rollouts_per_candidate", 16))
    n_roll = C["n_random_rollouts"]
    vae_steps = int(cfg.get("_vae_steps", 1500))
    mdn_steps = int(cfg.get("_mdn_steps", 300))
    gens = C["n_generations"]
    n_eval = C["n_eval_episodes"]
    if smoke:
        n_roll, vae_steps, mdn_steps, gens, pop, rolls_per_cand, n_eval = 40, 20, 15, 4, 8, 2, 20
    else:
        n_roll = max(40, int(round(n_roll * scale)))
        vae_steps = max(20, int(round(vae_steps * scale)))
        mdn_steps = max(15, int(round(mdn_steps * scale)))
        gens = max(3, int(round(gens * scale)))

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    lr = 1e-3 if C["optimizer"] == "adam_1e-3" else 1e-4

    # ---------------- data
    frames, acts = collect_rollouts(n_roll, T, seed, frame_skip=C["frame_skip"])
    n_train = max(1, int(0.9 * n_roll))
    tr_f, te_f = frames[:n_train], frames[n_train:]
    tr_a, te_a = acts[:n_train], acts[n_train:]
    n_frames = int(frames.shape[0] * frames.shape[1])

    # ---------------- stage 1: ConvVAE
    vae = ConvVAE(z_dim=z_dim)
    opt = torch.optim.Adam(vae.parameters(), lr=max(lr, 1e-3))
    bs = C["batch_size"]
    flat_tr = tr_f.reshape(-1, 64, 64, 3)
    flat_te = te_f.reshape(-1, 64, 64, 3)
    te_idx = rng.choice(len(flat_te), size=min(200, len(flat_te)), replace=False)
    x_te = to_torch(flat_te[te_idx])

    def vae_recon(x):
        with torch.no_grad():
            mu, lv = vae.encode(x)
            xr = vae.decode(mu)
            per = ((xr - x) ** 2).flatten(1).sum(1)
        return float(per.mean())

    vae_recon_initial = vae_recon(x_te)
    kl_tol = 0.5 * z_dim if C["kl_weight"] == "kl_tolerance_0.5_per_dim" else 0.0
    for step in range(vae_steps):
        idx = rng.integers(0, len(flat_tr), size=bs)
        x = to_torch(flat_tr[idx])
        tgt = x
        if shuffle:
            tgt = x[torch.randperm(x.shape[0])]      # shuffle targets within batch
        xr, mu, lv = vae(x)
        if C["recon_reduction"] == "sum_over_pixels":
            rec = ((xr - tgt) ** 2).flatten(1).sum(1).mean()
        else:
            rec = ((xr - tgt) ** 2).mean()
        kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).clamp(min=kl_tol).mean()
        loss = rec + kl
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
    vae_recon_final = vae_recon(x_te)
    vae.eval()

    # Exact memoisation of the (deterministic) renderer -> encoder map over the
    # finite state set of the substitute env; identical numbers, far cheaper.
    with torch.no_grad():
        mu_tab, lv_tab = [], []
        af = all_state_frames()
        for i in range(0, N_STATES, 256):
            m, l = vae.encode(to_torch(af[i:i + 256]))
            mu_tab.append(m); lv_tab.append(l)
        mu_tab = torch.cat(mu_tab); lv_tab = torch.cat(lv_tab)

    # ---------------- encode rollouts to latents
    def encode_all(fs):
        mus, lvs = [], []
        with torch.no_grad():
            flat = fs.reshape(-1, 64, 64, 3)
            for i in range(0, len(flat), 500):
                m, l = vae.encode(to_torch(flat[i:i + 500]))
                mus.append(m); lvs.append(l)
        mu = torch.cat(mus).view(fs.shape[0], fs.shape[1], z_dim)
        lv = torch.cat(lvs).view(fs.shape[0], fs.shape[1], z_dim)
        return mu, lv

    mu_tr, lv_tr = encode_all(tr_f)
    mu_te, lv_te = encode_all(te_f)
    a_tr = torch.from_numpy(act_to_cont(tr_a)).unsqueeze(-1)
    a_te = torch.from_numpy(act_to_cont(te_a)).unsqueeze(-1)

    # ---------------- stage 2: MDN-RNN
    rnn = MDNRNN(z_dim=z_dim, a_dim=1, hidden=hidden, k=5)
    opt_r = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    seq_len = min(int(C["rnn_seq_len"]) if str(C["rnn_seq_len"]).isdigit() else T, T)
    nb = min(C["batch_size"], mu_tr.shape[0])

    def sample_z(mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def rnn_nll(mu_s, lv_s, a_s):
        with torch.no_grad():
            z = sample_z(mu_s, lv_s)
            lp, m, ls, _ = rnn(z[:, :-1], a_s[:, :-1])
            return float(mdn_nll(lp, m, ls, z[:, 1:]))

    mdn_nll_initial = rnn_nll(mu_te, lv_te, a_te)
    for step in range(mdn_steps):
        idx = torch.from_numpy(rng.integers(0, mu_tr.shape[0], size=nb))
        z = sample_z(mu_tr[idx], lv_tr[idx])          # fresh z each batch (A.2)
        a = a_tr[idx]
        tgt = z[:, 1:]
        if shuffle:
            tgt = tgt[torch.randperm(tgt.shape[0])]
        lp, m, ls, _ = rnn(z[:, :-1], a[:, :-1])
        loss = mdn_nll(lp, m, ls, tgt)
        opt_r.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
        opt_r.step()
    mdn_nll_final = rnn_nll(mu_te, lv_te, a_te)
    rnn.eval()

    # reference-distribution baseline: marginal Gaussian fit to training z
    zt = sample_z(mu_tr, lv_tr).reshape(-1, z_dim)
    m0, s0 = zt.mean(0), zt.std(0).clamp(min=1e-3)
    zte = sample_z(mu_te, lv_te)[:, 1:].reshape(-1, z_dim)
    base_nll = float((0.5 * ((zte - m0) / s0) ** 2 + torch.log(s0) + 0.5 * math.log(2 * math.pi)).sum(-1).mean())

    # ---------------- stage 3: CMA-ES controller on [z, h]
    in_dim = z_dim + (hidden if C["controller_state_input"] == "h_only" else 2 * hidden)
    n_params = in_dim + (1 if C["controller_bias"] == "with_bias" else 0)

    def rollout_batch(params, n_rollouts, ep_seed):
        """params: (K, n_params). Runs K*n_rollouts episodes in parallel. Returns (K,) mean return."""
        K = params.shape[0]
        N = K * n_rollouts
        P = torch.from_numpy(np.repeat(params, n_rollouts, axis=0)).float()
        W, b = P[:, :in_dim], (P[:, in_dim] if C["controller_bias"] == "with_bias" else torch.zeros(N))
        env = BatchCatcher(N, seed=ep_seed, max_steps=T, tile_k=K)
        h = torch.zeros(1, N, hidden); c = torch.zeros(1, N, hidden)
        total = np.zeros(N, dtype=np.float64)
        with torch.no_grad():
            for t in range(T):
                si = torch.from_numpy(env.state_index())
                mu, lv = mu_tab[si], lv_tab[si]
                z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if C["z_encode_mode"] == "sample" else mu
                feat = torch.cat([z, h[0]], -1) if C["controller_state_input"] == "h_only" else torch.cat([z, h[0], c[0]], -1)
                a_cont = torch.tanh((W * feat).sum(-1) + b)
                a_disc = np.digitize(a_cont.numpy(), [-1 / 3, 1 / 3])   # tanh output split into thirds
                _, (h, c) = rnn.lstm(torch.cat([z, a_cont.unsqueeze(-1)], -1).unsqueeze(1), (h, c))
                _, r, d = env.step(a_disc)
                total += r
        return total.reshape(K, n_rollouts).mean(1), total

    es = SepCMAES(n_params, sigma=C["cma_sigma_init"], popsize=pop, seed=seed)
    fit_hist = []
    for g in range(gens):
        X = es.ask()
        gs = seed * 1000 + g if C["fitness_seed_policy"] == "resample_each_generation" else seed
        fit, _ = rollout_batch(X, rolls_per_cand, int(gs))
        if shuffle:
            fit = fit[np.random.default_rng(seed + g).permutation(len(fit))]
        es.tell(fit)
        fit_hist.append(float(np.mean(fit)))
        if os.environ.get("WM_VERBOSE") == "1" and g % 5 == 0:
            print(f"gen {g} mean {np.mean(fit):.3f} max {np.max(fit):.3f} "
                  f"std {np.std(fit):.3f} sigma {es.sigma:.3f} |mean| {np.abs(es.mean).mean():.3f}", flush=True)
    best = es.mean.reshape(1, -1)

    _, rets = rollout_batch(best, n_eval, ep_seed=seed + 77777)
    rand_params = np.zeros((1, n_params))
    # random policy: uniform over the 3 actions
    env = BatchCatcher(n_eval, seed=seed + 88888, max_steps=T)
    obs = env.reset()
    rrng = np.random.default_rng(seed + 999)
    rand_rets = np.zeros(n_eval)
    for t in range(T):
        obs, r, d = env.step(rrng.integers(0, 3, size=n_eval))
        rand_rets += r

    mean_return = float(rets.mean())
    mean_random = float(rand_rets.mean())
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sdr = float(rand_rets.std(ddof=1)) if len(rand_rets) > 1 else 0.0
    se = math.sqrt(sd ** 2 / max(len(rets), 1) + sdr ** 2 / max(len(rand_rets), 1))
    tstat = (mean_return - mean_random) / se if se > 0 else 0.0

    metrics = {
        "mean_return": mean_return,
        "mean_return_random_policy": mean_random,
        "return_std": sd,
        "return_std_random_policy": sdr,
        "controller_vs_random_tstat": float(tstat),
        "vae_recon_loss_initial": vae_recon_initial,
        "vae_recon_loss_final": vae_recon_final,
        "vae_recon_loss_drop": vae_recon_initial - vae_recon_final,
        "mdnrnn_nll_initial": mdn_nll_initial,
        "mdnrnn_nll_final": mdn_nll_final,
        "mdnrnn_baseline_nll": base_nll,
        "mdnrnn_nll_gap_vs_baseline": base_nll - mdn_nll_final,
        "_intermediates": {
            "n_rollouts": int(n_roll),
            "n_frames": n_frames,
            "episode_length": T,
            "n_train_rollouts": int(n_train),
            "n_heldout_rollouts": int(n_roll - n_train),
            "controller_n_params": int(n_params),
            "cma_generations": int(gens),
            "cma_popsize": int(pop),
            "rollouts_per_candidate": int(rolls_per_cand),
            "vae_steps": int(vae_steps),
            "mdn_steps": int(mdn_steps),
            "fitness_history_first": fit_hist[0] if fit_hist else None,
            "fitness_history_last": fit_hist[-1] if fit_hist else None,
            "runtime_s": time.time() - t0,
            "config_used": C,
            "scale": {
                "environment": "tiny 8x8 pixel Catcher rendered to 64x64x3 (CarRacing-v0/VizDoom out of reach on CPU)",
                "scale_factor": scale,
                "smoke": smoke,
                "vae_channels": "8/16/32/64 (paper 32/64/128/256), z=32 as paper",
                "lstm_hidden": hidden,
                "mixtures": 5,
                "rollouts": int(n_roll),
                "paper_rollouts": 10000,
                "cma_generations": int(gens),
                "paper_generations": 1800,
                "cma_popsize": int(pop),
                "paper_popsize": 64,
                "rollouts_per_candidate": int(rolls_per_cand),
                "paper_rollouts_per_candidate": 16,
                "episode_length": T,
            },
        },
    }
    payload = {"seed": seed, "split": C["split"], "n_examples": n_frames,
               "shuffled": bool(shuffle), "metrics": metrics}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "_intermediates"}, indent=1))
    print("runtime", time.time() - t0)


if __name__ == "__main__":
    main()
