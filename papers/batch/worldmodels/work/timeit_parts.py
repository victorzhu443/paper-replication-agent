import time, numpy as np, torch
from wm_env import BatchCatcher
import wm_main as W

vae = W.ConvVAE(32); vae.eval()
t = time.time()
f, a = W.collect_rollouts(50, 50, 0)
print("collect 50 rollouts", time.time() - t)
flat = f.reshape(-1, 64, 64, 3)
opt = torch.optim.Adam(vae.parameters(), 1e-3)
t = time.time()
for i in range(20):
    x = W.to_torch(flat[np.random.randint(0, len(flat), 100)])
    xr, mu, lv = vae(x)
    loss = ((xr - x) ** 2).flatten(1).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print("vae 20 steps", time.time() - t)
t = time.time()
with torch.no_grad():
    for i in range(50):
        x = W.to_torch(flat[:128])
        vae.encode(x)
print("128-batch encode x50", time.time() - t)
