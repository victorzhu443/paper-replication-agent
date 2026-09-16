"""PPO-Clip vs no-clip on CartPole-v1 (CPU-only substitute for the paper's MuJoCo suite)."""
from __future__ import annotations

import json, os, time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

torch.set_num_threads(4)

DEFAULTS = {
    "env_suite": "gym_cartpole_v1_200k_steps",
    "lr_schedule": "constant",
    "augmentation": "none",
    "batch_vs_steps": "steps",
    "mixed_precision": "fp32",
    "split": "test",
    "eval_return_source": "train_episode_returns_last_100",
    "episode_buffer_rule": "global_last_100_finished_episodes",
    "seeds": "3",
    "score_normalization_reference": "max_over_all_settings_and_seeds_per_env",
    "advantage_normalization": "per_batch",
    "obs_normalization": "none",
    "entropy_coef": "0.0",
    "shared_trunk": "separate",
    "value_net_training": "same_mlp_same_adam_same_epochs_weight1",
    "kl_estimator": "analytic_per_state",
    "max_grad_norm": "none",
    "clip_vloss": "none",
    "rollout_length_and_actors": "T=2048,N=1",
    "ppo_epochs_minibatch": "K=10,M=64",
    "total_timesteps": "200000",
    "noclip_update_rule": "same_K_epochs_ratio_objective",
    "terminal_bootstrap": "bootstrap_on_truncation_zero_on_terminal",
    "policy_std_parameterization": "state_independent_learned_logstd_init0",
    "init_scheme": "orthogonal_sqrt2_policy0.01",
    "adam_eps": "1e-5",
    "atari_wrappers": "mnih2016_standard_deepmind_wrappers",
    "atari_reward_clipping": "clip_train_report_raw",
}

GAMMA, LAM = 0.99, 0.95
LR = 3e-4


def mlp(inp, out, init, out_gain):
    layers, last = [], inp
    for _ in range(2):
        layers.append(nn.Linear(last, 64)); layers.append(nn.Tanh()); last = 64
    layers.append(nn.Linear(last, out))
    net = nn.Sequential(*layers)
    lin = [m for m in net if isinstance(m, nn.Linear)]
    for i, m in enumerate(lin):
        gain = out_gain if i == len(lin) - 1 else np.sqrt(2)
        if init.startswith("orthogonal"):
            nn.init.orthogonal_(m.weight, gain)
        elif init == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif init.startswith("normc"):
            w = torch.randn_like(m.weight)
            m.weight.data = gain * w / w.pow(2).sum(0, keepdim=True).sqrt()
        nn.init.zeros_(m.bias)
    return net


class RunningMS:
    def __init__(self, d):
        self.m = np.zeros(d); self.v = np.ones(d); self.n = 1e-4

    def update(self, x):
        b = x.reshape(1, -1) if x.ndim == 1 else x
        bm, bv, bn = b.mean(0), b.var(0), b.shape[0]
        d = bm - self.m; tot = self.n + bn
        self.m += d * bn / tot
        self.v = (self.v * self.n + bv * bn + d ** 2 * self.n * bn / tot) / tot
        self.n = tot

    def norm(self, x):
        return np.clip((x - self.m) / np.sqrt(self.v + 1e-8), -10, 10)


def train(variant, cfg, seed, total_steps):
    T = int(cfg["rollout_length_and_actors"].split(",")[0].split("=")[1])
    K = int(cfg["ppo_epochs_minibatch"].split(",")[0].split("=")[1])
    M = int(cfg["ppo_epochs_minibatch"].split(",")[1].split("=")[1])
    ent_c = float(cfg["entropy_coef"])
    adam_eps = float(cfg["adam_eps"])
    mgn = None if cfg["max_grad_norm"] == "none" else float(cfg["max_grad_norm"])
    obs_norm = cfg["obs_normalization"] == "running_mean_std"
    adv_norm = cfg["advantage_normalization"]

    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    od, ad = env.observation_space.shape[0], env.action_space.n
    pi = mlp(od, ad, cfg["init_scheme"], 0.01)
    vf = mlp(od, 1, cfg["init_scheme"], 1.0)
    params = list(pi.parameters()) + list(vf.parameters())
    opt = torch.optim.Adam(params, lr=LR, eps=adam_eps)
    rms = RunningMS(od) if obs_norm else None

    ep_returns, ep_ret = [], 0.0
    beta = 1.0
    n_updates = max(1, total_steps // T)
    steps_done = 0
    for upd in range(n_updates):
        if cfg["lr_schedule"] != "constant":
            frac = 1.0 - upd / n_updates
            for g in opt.param_groups:
                g["lr"] = LR * (frac if cfg["lr_schedule"] == "step" else 0.5 * (1 + np.cos(np.pi * (1 - frac))))
        O = np.zeros((T, od), np.float32); A = np.zeros(T, np.int64)
        LP = np.zeros(T, np.float32); R = np.zeros(T, np.float32)
        D = np.zeros(T, np.float32); TR = np.zeros(T, np.float32); V = np.zeros(T, np.float32)
        for t in range(T):
            if rms is not None:
                rms.update(obs.astype(np.float64))
                o_in = rms.norm(obs.astype(np.float64)).astype(np.float32)
            else:
                o_in = obs.astype(np.float32)
            O[t] = o_in
            with torch.no_grad():
                ot = torch.as_tensor(o_in)
                logits = pi(ot); v = vf(ot).item()
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                LP[t] = dist.log_prob(a).item()
            A[t] = int(a); V[t] = v
            obs, r, term, trunc, _ = env.step(int(a))
            R[t] = r; ep_ret += r
            D[t] = float(term); TR[t] = float(trunc)
            steps_done += 1
            if term or trunc:
                ep_returns.append(ep_ret); ep_ret = 0.0
                obs, _ = env.reset()
        if rms is not None:
            last_in = rms.norm(obs.astype(np.float64)).astype(np.float32)
        else:
            last_in = obs.astype(np.float32)
        with torch.no_grad():
            last_v = vf(torch.as_tensor(last_in)).item()
        # GAE
        adv = np.zeros(T, np.float32); lastgae = 0.0
        tb = cfg["terminal_bootstrap"]
        for t in reversed(range(T)):
            if t == T - 1:
                nextv = last_v; nonterm = 1.0
            else:
                nextv = V[t + 1]; nonterm = 1.0
            done_t = D[t] > 0; trunc_t = TR[t] > 0
            if done_t or trunc_t:
                if done_t and tb != "bootstrap_on_both":
                    nextv = 0.0
                elif trunc_t and tb == "zero_on_both":
                    nextv = 0.0
                else:
                    with torch.no_grad():
                        nextv = nextv if t < T - 1 else last_v
                nonterm = 0.0
            delta = R[t] + GAMMA * nextv - V[t]
            lastgae = delta + GAMMA * LAM * nonterm * lastgae
            adv[t] = lastgae
        ret = adv + V
        ob_t = torch.as_tensor(O); ac_t = torch.as_tensor(A)
        lp_t = torch.as_tensor(LP); adv_t = torch.as_tensor(adv); ret_t = torch.as_tensor(ret)
        with torch.no_grad():
            old_logits = pi(ob_t)
        if adv_norm == "per_batch":
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        idx = np.arange(T)
        approx_kl = 0.0
        for _ in range(K):
            np.random.shuffle(idx)
            for s in range(0, T, M):
                mb = idx[s:s + M]
                mbt = torch.as_tensor(mb)
                logits = pi(ob_t[mbt])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(ac_t[mbt])
                ratio = torch.exp(lp - lp_t[mbt])
                a_mb = adv_t[mbt]
                if adv_norm == "per_minibatch":
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                if variant.startswith("clip"):
                    eps = float(variant.split("_")[-1])
                    pl = -torch.min(ratio * a_mb,
                                    torch.clamp(ratio, 1 - eps, 1 + eps) * a_mb).mean()
                elif variant == "no_clip":
                    pl = -(ratio * a_mb).mean()
                else:  # kl variants
                    old_d = torch.distributions.Categorical(logits=old_logits[mbt])
                    kl = torch.distributions.kl_divergence(old_d, dist).mean()
                    pl = -(ratio * a_mb).mean() + beta * kl
                vloss = ((vf(ob_t[mbt]).squeeze(-1) - ret_t[mbt]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pl + vloss - ent_c * ent
                opt.zero_grad(); loss.backward()
                if mgn is not None:
                    torch.nn.utils.clip_grad_norm_(params, mgn)
                opt.step()
        if variant.startswith("adaptive_kl"):
            d_targ = float(variant.split("_")[-1])
            with torch.no_grad():
                new_logits = pi(ob_t)
                d = torch.distributions.kl_divergence(
                    torch.distributions.Categorical(logits=old_logits),
                    torch.distributions.Categorical(logits=new_logits)).mean().item()
            if d < d_targ / 1.5:
                beta /= 2
            elif d > d_targ * 1.5:
                beta *= 2
            approx_kl = d
        elif variant.startswith("fixed_kl"):
            beta = float(variant.split("_")[-1])
    env.close()
    last100 = ep_returns[-100:] if ep_returns else [0.0]
    return {"mean_return": float(np.mean(last100)),
            "n_episodes": len(ep_returns),
            "steps": steps_done,
            "final_kl": float(approx_kl)}


def random_policy_score(seed, n_ep=100):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed + 12345); env.action_space.seed(seed + 12345)
    rs = []
    for _ in range(n_ep):
        env.reset(); done = False; tot = 0.0
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            tot += r; done = term or trunc
        rs.append(tot)
    env.close()
    return float(np.mean(rs))


def main():
    seed = int(os.environ.get("SEED", "0"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}"))
    cfg = {k: (str(v) if not isinstance(v, str) else v) for k, v in cfg.items()}
    out_path = os.environ.get("METRICS_OUT", "metrics.json")

    total = int(float(cfg["total_timesteps"]))
    variants = ["clip_0.2", "no_clip"]
    if smoke:
        total = min(total, 12288)
    else:
        variants = ["clip_0.2", "no_clip", "clip_0.1", "clip_0.3",
                    "adaptive_kl_0.01", "fixed_kl_1"]

    t0 = time.time()
    results = {}
    for v in variants:
        results[v] = train(v, cfg, seed, total)
    rnd = random_policy_score(seed, 20 if smoke else 100)
    best = max(r["mean_return"] for r in results.values())
    norm = {v: (r["mean_return"] - rnd) / max(best - rnd, 1e-8) for v, r in results.items()}

    metrics = {
        "mean_return": results["clip_0.2"]["mean_return"],
        "mean_return_no_clip": results["no_clip"]["mean_return"],
        "avg_normalized_score": norm["clip_0.2"],
        "avg_normalized_score_no_clip": norm["no_clip"],
        "clip_minus_noclip": results["clip_0.2"]["mean_return"] - results["no_clip"]["mean_return"],
        "random_policy_return": rnd,
    }
    for v, r in results.items():
        metrics[f"mean_return__{v}"] = r["mean_return"]
        metrics[f"avg_normalized_score__{v}"] = norm[v]
    metrics["_intermediates"] = {
        "scale": {
            "env_substitute": "CartPole-v1 instead of 7 MuJoCo -v1 tasks (no MuJoCo, CPU only)",
            "total_timesteps": total,
            "paper_timesteps": 1000000,
            "variants_run": variants,
            "smoke": smoke,
        },
        "episodes": {v: r["n_episodes"] for v, r in results.items()},
        "steps": {v: r["steps"] for v, r in results.items()},
        "wall_time_s": time.time() - t0,
        "config": cfg,
    }
    n_examples = int(sum(r["steps"] for r in results.values()))
    with open(out_path, "w") as f:
        json.dump({"seed": seed, "split": cfg["split"], "n_examples": n_examples,
                   "metrics": metrics}, f, indent=1)
    print("wrote", out_path, metrics["mean_return"], metrics["mean_return_no_clip"])


if __name__ == "__main__":
    main()
