"""CartPole-v1 DQN surrogate for 'Playing Atari with Deep RL' (mechanism-level check)."""
from __future__ import annotations

import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym

torch.set_num_threads(min(8, os.cpu_count() or 1))


def cfg_get(cfg, key, default):
    v = cfg.get(key, default)
    return default if v is None else v


class MLP(nn.Module):
    def __init__(self, obs_dim, n_act, hidden, weight_init="kaiming"):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, n_act)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                if weight_init == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif weight_init == "glorot":
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, cap, obs_dim):
        self.cap = cap
        self.s = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, obs_dim), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.i = 0
        self.n = 0

    def add(self, s, a, r, s2, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs, rng):
        idx = rng.integers(0, self.n, size=bs)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.d[idx]))


def random_policy_return(env_id, episodes, seed):
    env = gym.make(env_id)
    rets = []
    for e in range(episodes):
        obs, _ = env.reset(seed=seed + 10000 + e)
        done, R = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            R += r
            done = term or trunc
        rets.append(R)
    env.close()
    return rets


def greedy_eval(net, env_id, episodes, seed, eps=0.05):
    env = gym.make(env_id)
    rets = []
    rng = np.random.default_rng(seed + 777)
    for e in range(episodes):
        obs, _ = env.reset(seed=seed + 20000 + e)
        done, R = False, 0.0
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(env.action_space.n))
            else:
                with torch.no_grad():
                    a = int(net(torch.from_numpy(np.asarray(obs, np.float32))).argmax())
            obs, r, term, trunc, _ = env.step(a)
            R += r
            done = term or trunc
    	    
        rets.append(R)
    env.close()
    return rets


def main():
    t0 = time.time()
    seed = int(os.environ.get("SEED", "0"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    cfg = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}")
    out_path = os.environ.get("METRICS_OUT", "metrics.json")

    env_id = "CartPole-v1"
    total_steps = int(cfg_get(cfg, "total_steps", 150000))
    if smoke:
        total_steps = int(cfg_get(cfg, "smoke_total_steps", 6000))
    n_seeds = int(cfg_get(cfg, "n_seeds", 3))
    if smoke:
        n_seeds = 1
    gamma = float(cfg_get(cfg, "gamma", 0.99))
    hidden = cfg_get(cfg, "hidden_sizes", "[128,128]")
    if isinstance(hidden, str):
        hidden = json.loads(hidden)
    replay_capacity = int(cfg_get(cfg, "replay_capacity", 50000))
    learning_starts = int(cfg_get(cfg, "learning_starts", 1000))
    target_network = cfg_get(cfg, "target_network", "frozen_periodic_sync")
    target_sync = int(cfg_get(cfg, "target_sync_interval", 500))
    train_freq_opt = cfg_get(cfg, "train_freq", "1_per_step")
    train_freq = {"1_per_step": 1, "1_per_4_steps": 4, "multiple_per_step": 1}[train_freq_opt]
    updates_per = 2 if train_freq_opt == "multiple_per_step" else 1
    eps_sched = cfg_get(cfg, "epsilon_schedule", "linear_1_to_0.1_over_10pct_steps")
    eps_final, eps_frac = {"linear_1_to_0.1_over_10pct_steps": (0.1, 0.10),
                           "linear_1_to_0.05_over_10pct_steps": (0.05, 0.10),
                           "linear_1_to_0.01_over_20pct_steps": (0.01, 0.20)}[eps_sched]
    loss_fn = cfg_get(cfg, "loss_fn", "mse")
    optimizer_opt = cfg_get(cfg, "optimizer", "rmsprop_0.95_decay")
    lr = float(cfg_get(cfg, "learning_rate", 5e-4))
    lr_schedule = cfg_get(cfg, "lr_schedule", "constant")
    batch_size = int(cfg_get(cfg, "batch_size", 32))
    weight_init = cfg_get(cfg, "weight_init", "kaiming")
    eval_eps_count = 20 if smoke else int(cfg_get(cfg, "eval_greedy_episodes", 50))
    rand_eps_count = 10 if smoke else 50
    split = cfg_get(cfg, "split", "test")

    per_seed_last50, per_seed_eval, per_seed_best, all_curves = [], [], [], []
    total_episodes = 0
    for s_i in range(n_seeds):
        sd = seed + s_i
        random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
        rng = np.random.default_rng(sd)
        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        n_act = env.action_space.n
        net = MLP(obs_dim, n_act, hidden, weight_init)
        tgt = MLP(obs_dim, n_act, hidden, weight_init)
        tgt.load_state_dict(net.state_dict())
        if optimizer_opt == "adam":
            opt = torch.optim.Adam(net.parameters(), lr=lr)
        elif optimizer_opt == "rmsprop_paper_defaults":
            opt = torch.optim.RMSprop(net.parameters(), lr=lr)
        else:
            opt = torch.optim.RMSprop(net.parameters(), lr=lr, alpha=0.95, eps=1e-2)
        sched = None
        if lr_schedule == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
        elif lr_schedule == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, total_steps // 3), gamma=0.5)
        buf = Replay(replay_capacity, obs_dim)
        crit = nn.MSELoss() if loss_fn == "mse" else nn.SmoothL1Loss()

        obs, _ = env.reset(seed=sd)
        ep_ret, ep_returns = 0.0, []
        n_updates = 0
        for step in range(total_steps):
            frac = min(1.0, step / max(1.0, eps_frac * total_steps))
            eps = 1.0 + frac * (eps_final - 1.0)
            if rng.random() < eps or buf.n < learning_starts:
                a = int(rng.integers(n_act))
            else:
                with torch.no_grad():
                    a = int(net(torch.from_numpy(np.asarray(obs, np.float32))).argmax())
            obs2, r, term, trunc, _ = env.step(a)
            ep_ret += r
            buf.add(obs, a, r, obs2, float(term))
            obs = obs2
            if term or trunc:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()
            if buf.n >= learning_starts and step % train_freq == 0:
                for _ in range(updates_per):
                    bs, ba, br, bs2, bd = buf.sample(batch_size, rng)
                    with torch.no_grad():
                        qn = (tgt if target_network == "frozen_periodic_sync" else net)(bs2).max(1).values
                        y = br + gamma * (1.0 - bd) * qn
                    q = net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                    loss = crit(q, y)
                    opt.zero_grad(); loss.backward(); opt.step()
                    n_updates += 1
                    if sched is not None:
                        sched.step()
                    if target_network == "frozen_periodic_sync" and n_updates % target_sync == 0:
                        tgt.load_state_dict(net.state_dict())
        env.close()
        last50 = ep_returns[-50:] if ep_returns else [0.0]
        per_seed_last50.append(float(np.mean(last50)))
        per_seed_best.append(float(np.max(ep_returns)) if ep_returns else 0.0)
        ev = greedy_eval(net, env_id, eval_eps_count, sd)
        per_seed_eval.append(float(np.mean(ev)))
        total_episodes += len(ep_returns)
        all_curves.append([float(np.mean(ep_returns[i:i + 20])) for i in range(0, len(ep_returns), 20)][-10:])

    rand = random_policy_return(env_id, rand_eps_count, seed)
    rand_mean = float(np.mean(rand))
    mean_return = float(np.mean(per_seed_last50))

    metrics = {
        "mean_return": mean_return,
        "mean_return_std": float(np.std(per_seed_last50, ddof=1)) if len(per_seed_last50) > 1 else 0.0,
        "eval_mean_return_eps005": float(np.mean(per_seed_eval)),
        "best_episode_return": float(np.max(per_seed_best)),
        "random_policy_return": rand_mean,
        "median_return": float(np.median(per_seed_last50)),
        "learning_gap": mean_return - rand_mean,
        "mechanism_check_passed": float(mean_return > 5.0 * rand_mean),
        "_intermediates": {
            "scale": {
                "substitute_env": "CartPole-v1 (ALE/Atari out of CPU reach)",
                "total_env_steps_per_seed": total_steps,
                "n_seeds": n_seeds,
                "smoke": smoke,
                "network": f"MLP hidden={hidden} (paper: conv 16/32 + fc256)",
                "atari_claims": "Untested",
            },
            "total_train_episodes": total_episodes,
            "per_seed_last50": per_seed_last50,
            "per_seed_eval_eps005": per_seed_eval,
            "random_policy_episodes": rand_eps_count,
            "curve_tail_mean20": all_curves,
            "runtime_s": round(time.time() - t0, 1),
            "config_used": {
                "gamma": gamma, "hidden_sizes": hidden, "replay_capacity": replay_capacity,
                "learning_starts": learning_starts, "target_network": target_network,
                "target_sync_interval": target_sync, "train_freq": train_freq_opt,
                "epsilon_schedule": eps_sched, "loss_fn": loss_fn, "optimizer": optimizer_opt,
                "lr_schedule": lr_schedule, "learning_rate": lr, "weight_init": weight_init,
                "split": split, "batch_vs_steps": cfg_get(cfg, "batch_vs_steps", "steps"),
                "mixed_precision": cfg_get(cfg, "mixed_precision", "fp32"),
                "augmentation": cfg_get(cfg, "augmentation", "none"),
                "eval_episodes": cfg_get(cfg, "eval_episodes", "unspecified"),
                "eval_checkpoint": cfg_get(cfg, "eval_checkpoint", "final_network"),
                "eval_partial_episodes": cfg_get(cfg, "eval_partial_episodes", "exclude"),
                "episodic_life": cfg_get(cfg, "episodic_life", "game_over_only"),
                "downsample_method": cfg_get(cfg, "downsample_method", "bilinear_bottom_crop"),
                "frame_stack_selection": cfg_get(cfg, "frame_stack_selection", "last_4_acted_frames"),
                "frame_count_definition": cfg_get(cfg, "frame_count_definition", "agent_steps"),
                "replay_capacity_unit": cfg_get(cfg, "replay_capacity_unit", "agent_steps_transitions"),
                "random_baseline_protocol": cfg_get(cfg, "random_baseline_protocol", "same_steps_and_frameskip_as_dqn"),
                "random_starts": cfg_get(cfg, "random_starts", "none"),
                "n_seeds": n_seeds,
            },
        },
    }
    with open(out_path, "w") as f:
        json.dump({"seed": seed, "split": split, "n_examples": total_episodes, "metrics": metrics}, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "_intermediates"}, indent=2))


if __name__ == "__main__":
    main()
