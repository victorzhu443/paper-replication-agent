import numpy as np, torch, time
import wm_main as W

frames, acts = W.collect_rollouts(200, 60, 0)
flat = frames.reshape(-1, 64, 64, 3)
rng = np.random.default_rng(0)
for lr in (1e-3, 3e-3):
    torch.manual_seed(0)
    vae = W.ConvVAE(32)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    t = time.time()
    for step in range(1501):
        x = W.to_torch(flat[rng.integers(0, len(flat), 100)])
        xr, mu, lv = vae(x)
        rec = ((xr - x) ** 2).flatten(1).sum(1).mean()
        kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).clamp(min=16.0).mean()
        (rec + kl).backward(); opt.step(); opt.zero_grad()
        if step % 250 == 0:
            print(f"lr {lr} step {step} rec {rec.item():.1f} kl {(-0.5*(1+lv-mu**2-lv.exp()).sum(1)).mean().item():.1f} t {time.time()-t:.0f}s", flush=True)
