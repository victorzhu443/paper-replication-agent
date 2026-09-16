import re
s = open("wm_main.py").read()

s = s.replace("from wm_env import BatchCatcher",
              "from wm_env import BatchCatcher, all_state_frames, N_STATES")

s = s.replace('    T = 50                                    # env episode length',
              '    T = 60                                    # env episode length')
s = s.replace('rolls_per_cand = int(cfg.get("_rollouts_per_candidate", 4))',
              'rolls_per_cand = int(cfg.get("_rollouts_per_candidate", 16))')

# memoised encoder table, inserted after vae.eval()
s = s.replace("""    vae_recon_final = vae_recon(x_te)
    vae.eval()
""",
"""    vae_recon_final = vae_recon(x_te)
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
""")

old = s[s.index("        env = BatchCatcher(N, seed=ep_seed, max_steps=T)"):s.index("        return total.reshape(K, n_rollouts).mean(1), total")]
new = """        env = BatchCatcher(N, seed=ep_seed, max_steps=T)
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
"""
s = s.replace(old, new)
open("wm_main.py", "w").write(s)
print("ok")
