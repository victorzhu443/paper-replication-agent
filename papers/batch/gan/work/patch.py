import re
s = open("gan_mnist.py").read()
old_start = s.index("# ---------------- optional Parzen ----------------")
old_end = s.index('metrics["_intermediates"]')
new = '''# ---------------- Parzen-window log-likelihood ----------------
pc = C["parzen_config"]
nsamp = 1000 if pc.startswith("1000") else 10000
if SMOKE:
    nsamp = 1000
with torch.no_grad():
    S = g(noise(nsamp))
sigmas = np.logspace(-1.3, 0.0, 12) if not pc.startswith("1000") else np.logspace(-1.3, 0.0, 5)
if SMOKE:
    sigmas = np.logspace(-1.3, 0.0, 5)


def parzen_ll(X, S, sigma, chunk=500):
    s2 = (S ** 2).sum(1)
    outs = []
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        dist = (x ** 2).sum(1, keepdim=True) + s2.unsqueeze(0) - 2.0 * (x @ S.T)
        ll = torch.logsumexp(-dist.clamp_min(0) / (2 * sigma ** 2), dim=1) - np.log(len(S)) \\
            - 784 * np.log(sigma * np.sqrt(2 * np.pi))
        outs.append(ll)
    return torch.cat(outs)


nval = 1000 if not SMOKE else 200
Xv = torch.from_numpy(np.ascontiguousarray((Xval if len(Xval) else Xtr)[:nval]))
with torch.no_grad():
    cv = [(float(parzen_ll(Xv, S, sg).mean()), float(sg)) for sg in sigmas]
    best_s = max(cv)[1]
    target = Xte_t if C["split"] == "test" else Xv
    lls = parzen_ll(target, S, best_s)
pll = float(lls.mean())
if C["parzen_log_base"] == "log2_bits":
    pll /= np.log(2)
metrics["parzen_log_likelihood"] = pll
metrics["parzen_sem"] = float(lls.std(unbiased=True) / np.sqrt(len(lls)))
metrics["parzen_sigma"] = float(best_s)

'''
s = s[:old_start] + new + s[old_end:]
s = s.replace('TIME_BUDGET = 60 if SMOKE else 660',
              'TIME_BUDGET = float(os.environ.get("TIME_BUDGET_S", 60 if SMOKE else 660))')
s = s.replace('"parzen": C["parzen_config"]',
              '"parzen": pc, "parzen_n_samples": nsamp, "parzen_cv_val_n": nval')
open("gan_mnist.py", "w").write(s)
print("ok")
