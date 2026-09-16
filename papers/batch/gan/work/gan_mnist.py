import json, os, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = int(os.environ.get("SEED", "0"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
CFG = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}")
OUT = os.environ.get("METRICS_OUT", "metrics.json")

D_ = dict(lr_schedule="constant", augmentation="none", input_scaling="unit_interval_0_1",
          validation_split="50000_train_10000_val", batch_vs_steps="steps", mixed_precision="fp32",
          split="test", generator_objective="non_saturating", latent_dim="100",
          noise_prior="uniform_-1_1", architecture="reduced_cpu_512_512_G_512_D",
          n_hidden_layers="2_hidden_layers_each", batch_size="100",
          optimizer="sgd_momentum_lr0.1_mom0.5", dropout_rate="0.5", k="1",
          train_steps="fit_15min_cpu_budget", weight_init="glorot_uniform",
          equilibrium_metric="sum_bce_real_plus_fake_last_window",
          eval_classifier="small_cnn_trained_in_script", parzen_config="skip_parzen",
          parzen_log_base="natural_log_nats", n_seeds="3", output_activation="sigmoid_0_1")
C = {**D_, **{k: v for k, v in CFG.items() if not k.startswith("_")}}

torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(int(os.environ.get("NTHREADS", "15")))

# ---------------- data ----------------
ROOTS = ["/Users/vzhu/Developer/research-copy/runs/data_cache/mnist",
         os.path.expanduser("~/Developer/research-copy/runs/data_cache/mnist"),
         "data_cache/mnist"]


def load_idx(path):
    with open(path, "rb") as f:
        b = f.read()
    magic = int.from_bytes(b[:4], "big")
    nd = magic & 0xFF
    dims = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * nd).reshape(dims)


def mnist():
    for r in ROOTS:
        raw = os.path.join(r, "MNIST", "raw")
        if os.path.isdir(raw):
            return (load_idx(os.path.join(raw, "train-images-idx3-ubyte")),
                    load_idx(os.path.join(raw, "train-labels-idx1-ubyte")),
                    load_idx(os.path.join(raw, "t10k-images-idx3-ubyte")),
                    load_idx(os.path.join(raw, "t10k-labels-idx1-ubyte")))
    raise RuntimeError("MNIST not found in cache")


Xtr_u8, ytr, Xte_u8, yte = mnist()


def scale(x):
    x = x.astype(np.float32).reshape(len(x), -1)
    if C["input_scaling"] == "raw_0_255":
        return x
    if C["input_scaling"] == "minus1_to_1":
        return x / 127.5 - 1.0
    return x / 255.0


Xall = scale(Xtr_u8)
if C["validation_split"] == "60000_train_no_val":
    ntr = 60000
elif C["validation_split"] == "59000_train_1000_val":
    ntr = 59000
else:
    ntr = 50000
Xtr, Xval = Xall[:ntr], Xall[ntr:]
ytr_t = torch.from_numpy(ytr[:ntr].astype(np.int64))
Xte = scale(Xte_u8)
yte_t = torch.from_numpy(yte.astype(np.int64))
if SMOKE:
    Xtr, ytr_t = Xtr[:4000], ytr_t[:4000]
    Xte, yte_t = Xte[:1000], yte_t[:1000]
Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr))
Xte_t = torch.from_numpy(np.ascontiguousarray(Xte))

# ---------------- models ----------------
zdim = int(C["latent_dim"])
arch = C["architecture"]
if arch.startswith("paper_code"):
    gw, dw, pieces = 1200, 240, 5
elif arch.startswith("reduced_cpu_256"):
    gw, dw, pieces = 256, 256, 5
else:
    gw, dw, pieces = 512, 512, 5
nlayers = {"1_hidden_layer_each": 1, "2_hidden_layers_each": 2, "3_hidden_layers_each": 3}[C["n_hidden_layers"]]
drop = float(C["dropout_rate"])


def init(m):
    if isinstance(m, nn.Linear):
        if C["weight_init"] == "normal_0.02":
            nn.init.normal_(m.weight, 0, 0.02)
        elif C["weight_init"] == "uniform_irange_0.005":
            nn.init.uniform_(m.weight, -0.005, 0.005)
        else:
            nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class G(nn.Module):
    def __init__(self):
        super().__init__()
        layers, d = [], zdim
        for i in range(nlayers):
            layers += [nn.Linear(d, gw), nn.ReLU() if i % 2 == 0 else nn.Sigmoid()]
            d = gw
        layers += [nn.Linear(d, 784), nn.Sigmoid() if C["output_activation"] == "sigmoid_0_1" else nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(init)

    def forward(self, z):
        return self.net(z)


class Maxout(nn.Module):
    def __init__(self, din, dout, pieces):
        super().__init__()
        self.lin = nn.Linear(din, dout * pieces)
        self.dout, self.pieces = dout, pieces

    def forward(self, x):
        return self.lin(x).view(x.shape[0], self.dout, self.pieces).max(dim=2).values


class D(nn.Module):
    def __init__(self):
        super().__init__()
        mods, d = [], 784
        for _ in range(nlayers):
            mods += [Maxout(d, dw, pieces), nn.Dropout(drop)]
            d = dw
        mods += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*mods)
        self.apply(init)

    def forward(self, x):
        return self.net(x).squeeze(1)


g, d = G(), D()


def make_opt(params):
    o = C["optimizer"]
    if o.startswith("adam"):
        return torch.optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))
    if o == "sgd_momentum_lr0.01_mom0.9":
        return torch.optim.SGD(params, lr=0.01, momentum=0.9)
    return torch.optim.SGD(params, lr=0.1, momentum=0.5)


optG, optD = make_opt(g.parameters()), make_opt(d.parameters())
bs = int(C["batch_size"])
K = int(C["k"])

if C["train_steps"] == "fit_15min_cpu_budget":
    steps = 200 if SMOKE else 20000
elif C["train_steps"].endswith("epochs"):
    ep = int(C["train_steps"].split("_")[0])
    steps = max(1, ep * len(Xtr_t) // bs)
else:
    steps = 200 if SMOKE else 20000
TIME_BUDGET = float(os.environ.get("TIME_BUDGET_S", 60 if SMOKE else 660))


def noise(n):
    if C["noise_prior"] == "gaussian_0_1":
        return torch.randn(n, zdim)
    if C["noise_prior"] == "uniform_0_1":
        return torch.rand(n, zdim)
    return torch.rand(n, zdim) * 2 - 1


def lr_at(frac):
    base = 2e-4 if C["optimizer"].startswith("adam") else (0.01 if "0.01" in C["optimizer"] else 0.1)
    if C["lr_schedule"] == "cosine":
        return base * 0.5 * (1 + math.cos(math.pi * frac))
    if C["lr_schedule"] == "step":
        return base * (0.1 ** int(frac * 3))
    return base


bce = nn.BCEWithLogitsLoss()
t0 = time.time()
hist = []
N = len(Xtr_t)
done = 0
for step in range(steps):
    if time.time() - t0 > TIME_BUDGET:
        break
    lr = lr_at(step / steps)
    for og in (optG, optD):
        for pg_ in og.param_groups:
            pg_["lr"] = lr
    for _ in range(K):
        idx = torch.randint(0, N, (bs,))
        real = Xtr_t[idx]
        fake = g(noise(bs)).detach()
        d.train()
        lr_real = bce(d(real), torch.ones(bs))
        lr_fake = bce(d(fake), torch.zeros(bs))
        lossD = lr_real + lr_fake
        optD.zero_grad(); lossD.backward(); optD.step()
    z = noise(bs)
    fake = g(z)
    out = d(fake)
    if C["generator_objective"] == "minimax":
        lossG = -bce(out, torch.zeros(bs))
    else:
        lossG = bce(out, torch.ones(bs))
    optG.zero_grad(); lossG.backward(); optG.step()
    hist.append(float(lossD))
    done = step + 1

# ---------------- equilibrium diagnostics (eval mode, no dropout) ----------------
d.eval(); g.eval()
with torch.no_grad():
    dr, df, ls = [], [], []
    for _ in range(50):
        idx = torch.randint(0, len(Xtr_t), (bs,))
        real = Xtr_t[idx]
        fake = g(noise(bs))
        lo_r, lo_f = d(real), d(fake)
        ls.append(float(bce(lo_r, torch.ones(bs)) + bce(lo_f, torch.zeros(bs))))
        dr.append(float(torch.sigmoid(lo_r).mean())); df.append(float(torch.sigmoid(lo_f).mean()))
if C["equilibrium_metric"] == "mean_bce_real_plus_fake_full_epoch":
    d_loss = float(np.mean(hist)) if hist else float("nan")
elif C["equilibrium_metric"] == "value_function_on_heldout_batch":
    d_loss = float(np.mean(ls))
else:  # sum_bce_real_plus_fake_last_window
    w = min(len(hist), 200)
    d_loss = float(np.mean(hist[-w:])) if hist else float("nan")
mean_d_out = float((np.mean(dr) + np.mean(df)) / 2)

# ---------------- sample-quality classifier ----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.f = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f(x.flatten(1))


class MLPc(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.n(x)


clf = MLPc() if C["eval_classifier"] == "small_mlp_trained_in_script" else CNN()
optC = torch.optim.Adam(clf.parameters(), lr=1e-3)
csteps = 200 if SMOKE else 2000
for s in range(csteps):
    idx = torch.randint(0, len(Xtr_t), (128,))
    loss = F.cross_entropy(clf(Xtr_t[idx]), ytr_t[idx])
    optC.zero_grad(); loss.backward(); optC.step()
clf.eval()
with torch.no_grad():
    acc = float((clf(Xte_t).argmax(1) == yte_t).float().mean())
    samples = g(noise(1000))
    p = F.softmax(clf(samples), 1)
    conf = float(p.max(1).values.mean())
    frac_conf = float((p.max(1).values > 0.9).float().mean())
    cls_mean = p.mean(0)
    cls_entropy = float(-(cls_mean * torch.log(cls_mean + 1e-12)).sum())

metrics = {
    "discriminator_loss": d_loss,
    "value_function_at_optimum": -d_loss,
    "mean_discriminator_output": mean_d_out,
    "mean_d_output_real": float(np.mean(dr)),
    "mean_d_output_fake": float(np.mean(df)),
    "sample_mean_confidence": conf,
    "sample_frac_confident_0.9": frac_conf,
    "sample_class_entropy": cls_entropy,
    "classifier_test_accuracy": acc,
}

# ---------------- Parzen-window log-likelihood ----------------
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
        ll = torch.logsumexp(-dist.clamp_min(0) / (2 * sigma ** 2), dim=1) - np.log(len(S)) \
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

metrics["_intermediates"] = {
    "scale": {"g_hidden": gw, "d_hidden": dw, "maxout_pieces": pieces, "train_steps_run": done,
              "train_steps_requested": steps, "time_budget_s": TIME_BUDGET,
              "clf_steps": csteps, "smoke": SMOKE, "parzen": pc, "parzen_n_samples": nsamp, "parzen_cv_val_n": nval},
    "n_train": int(len(Xtr_t)), "n_val": int(len(Xval)), "n_test": int(len(Xte_t)),
    "train_wall_s": round(time.time() - t0, 1),
    "d_loss_first50": float(np.mean(hist[:50])) if hist else None,
}
json.dump({"seed": SEED, "split": C["split"], "n_examples": int(len(Xte_t)), "metrics": metrics},
          open(OUT, "w"), indent=1)
print("wrote", OUT, {k: v for k, v in metrics.items() if k != "_intermediates"})
