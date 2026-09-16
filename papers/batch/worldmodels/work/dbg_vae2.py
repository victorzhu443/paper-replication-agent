import numpy as np, torch, time
import wm_main as W
from wm_env import all_state_frames, N_STATES, G

frames, acts = W.collect_rollouts(200, 60, 0)
flat = frames.reshape(-1, 64, 64, 3)
rng = np.random.default_rng(0)
torch.manual_seed(0)
vae = W.ConvVAE(32, ch=(8, 16, 32, 64))
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
t = time.time()
for step in range(801):
    x = W.to_torch(flat[rng.integers(0, len(flat), 100)])
    xr, mu, lv = vae(x)
    rec = ((xr - x) ** 2).flatten(1).sum(1).mean()
    klraw = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
    (rec + klraw.clamp(min=16.0)).backward(); opt.step(); opt.zero_grad()
    if step % 100 == 0:
        print(f"step {step} rec {rec.item():.1f} kl {klraw.item():.1f} t {time.time()-t:.0f}s", flush=True)

# linear probe: can a linear map of z recover the sign of (ball_c - pad)?
vae.eval()
af = all_state_frames()
with torch.no_grad():
    mus, lvs = [], []
    for i in range(0, N_STATES, 256):
        m, l = vae.encode(W.to_torch(af[i:i+256])); mus.append(m); lvs.append(l)
    mu = torch.cat(mus); lv = torch.cat(lvs)
r, c, p = np.meshgrid(np.arange(G), np.arange(G), np.arange(G), indexing="ij")
y = np.sign(c.ravel() - p.ravel())
X = mu.numpy()
Xa = np.hstack([X, np.ones((len(X), 1))])
w, *_ = np.linalg.lstsq(Xa, y, rcond=None)
pred = np.sign(Xa @ w)
print("probe acc", (pred == y).mean(), "posterior sigma mean", lv.exp().sqrt().mean().item())
print("mu std across states", X.std(0).mean())
