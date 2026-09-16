import numpy as np, torch
from wm_env import BatchCatcher, all_state_frames, N_STATES
import wm_main as W

z_dim, hidden = 32, 64
vae = W.ConvVAE(z_dim); vae.eval()
rnn = W.MDNRNN(z_dim, 1, hidden, 5); rnn.eval()
with torch.no_grad():
    af = all_state_frames()
    mu_tab, lv_tab = [], []
    for i in range(0, N_STATES, 256):
        m, l = vae.encode(W.to_torch(af[i:i+256])); mu_tab.append(m); lv_tab.append(l)
    mu_tab = torch.cat(mu_tab); lv_tab = torch.cat(lv_tab)
print("mu_tab std across states", mu_tab.std(0).mean().item(), "lv mean", lv_tab.mean().item())

in_dim = z_dim + hidden
T = 60
def roll(params, R, seed):
    K = params.shape[0]; N = K*R
    P = torch.from_numpy(np.repeat(params, R, 0)).float()
    Wm, b = P[:, :in_dim], P[:, in_dim]
    env = BatchCatcher(N, seed=seed, max_steps=T)
    h = torch.zeros(1,N,hidden); c = torch.zeros(1,N,hidden)
    tot = np.zeros(N); acts = []
    with torch.no_grad():
        for t in range(T):
            si = torch.from_numpy(env.state_index())
            mu, lv = mu_tab[si], lv_tab[si]
            z = mu + torch.exp(0.5*lv)*torch.randn_like(mu)
            feat = torch.cat([z, h[0]], -1)
            a_cont = torch.tanh((Wm*feat).sum(-1)+b)
            a_disc = np.digitize(a_cont.numpy(), [-1/3, 1/3])
            acts.append(a_disc)
            _, (h,c) = rnn.lstm(torch.cat([z, a_cont.unsqueeze(-1)],-1).unsqueeze(1), (h,c))
            _, r, d = env.step(a_disc)
            tot += r
    return tot.reshape(K,R).mean(1), np.array(acts)

rng = np.random.default_rng(0)
X = rng.standard_normal((32, in_dim+1))*0.1
f, acts = roll(X, 16, 1)
print("fitness", np.round(f,3))
print("action hist", np.bincount(acts.ravel(), minlength=3))
X2 = rng.standard_normal((32, in_dim+1))*1.0
f2, acts2 = roll(X2, 16, 1)
print("fitness sigma=1", np.round(f2,3), np.bincount(acts2.ravel(), minlength=3))
