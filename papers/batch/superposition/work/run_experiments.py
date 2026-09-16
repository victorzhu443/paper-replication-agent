import json, os, time
import numpy as np
import torch
from toy import train, stats, importance_vec, pick_best

SEED = int(os.environ.get("SEED", "0"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
SCALE = float(os.environ.get("SCALE", "1.0"))
CFG = json.loads(os.environ.get("REPLICATOR_CONFIG") or "{}")
OUT = os.environ.get("METRICS_OUT", "metrics.json")
SHUFFLE = bool(CFG.get("_shuffle_labels", False))

steps_full = int(CFG.get("training_steps", 10000))
batch = int(CFG.get("batch_size", 1024))
nrest_opt = str(CFG.get("n_restarts", "10_best_loss"))
n_restarts = 1 if nrest_opt == "1" else int(nrest_opt.split("_")[0])
ugrid_opt = str(CFG.get("uniform_sparsity_grid", "40_log_spaced"))

# ---- compute-tier-2 scaling (recorded) ----
scale_note = {}
steps = max(100, int(round(steps_full * SCALE)))
if SMOKE:
    steps = max(60, int(round(400 * SCALE)))
    batch = 256
    n_restarts = min(n_restarts, 2)
scale_note["steps"] = steps
scale_note["steps_spec"] = steps_full
scale_note["batch_size"] = batch
scale_note["SCALE"] = SCALE
scale_note["smoke"] = SMOKE

g = torch.Generator().manual_seed(SEED)


def run_group(n, m, model_type, base_imp, sparsities, restarts, st, bs, uniform_imp=False):
    """sparsities: list; returns list of (stats, loss) per sparsity, best-of-restarts."""
    R = len(sparsities) * restarts
    sp = np.repeat(np.asarray(sparsities, dtype=np.float32), restarts)
    imp = np.ones(n, dtype=np.float32) if uniform_imp else importance_vec(n, base_imp, CFG)
    W, b, loss = train(n, m, model_type, imp, sp, st, bs, CFG, g,
                       restarts=R, shuffle_labels=SHUFFLE)
    out = []
    for k in range(len(sparsities)):
        sl = slice(k * restarts, (k + 1) * restarts)
        Wb, bb, lb = pick_best(W[sl], b[sl], loss[sl])
        s = stats(Wb, CFG)
        s["loss"] = lb
        s["bias"] = bb.numpy()
        out.append(s)
    return out


t0 = time.time()
metrics = {}
inter = {"scale": scale_note}

# ---------- 1. ReLU output n=20 m=5 sparsity sweep ----------
if str(CFG.get("sparsity_grid", "task_five")) == "paper_seven":
    dens20 = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
else:
    dens20 = [1.0, 0.3, 0.1, 0.03, 0.01]
r20 = run_group(20, 5, "relu", 0.7, [1 - d for d in dens20], n_restarts, steps, batch)
sweep = {}
for d, s in zip(dens20, r20):
    sweep[f"1-S={d}"] = dict(count=round(s["count"], 3), frob2=round(s["frob2"], 3),
                             mean_sup=round(float(np.mean(s["sup"])), 4),
                             loss=round(s["loss"], 5),
                             norms=[round(float(v), 3) for v in s["norms"]])
inter["relu_n20_m5_sweep"] = sweep
dense = r20[0]
sparse = r20[dens20.index(0.01)]
metrics["num_features_represented__relu_output_n20_m5_dense"] = dense["count"]
metrics["num_features_represented__relu_output_n20_m5_s099"] = sparse["count"]
metrics["superposition_metric__relu_output_n20_m5_dense"] = float(np.mean(
    dense["sup"][dense["norms"] ** 2 > 0.5])) if (dense["norms"] ** 2 > 0.5).any() else 0.0

# ---------- 2. Linear n=20 m=5 ----------
lin20 = run_group(20, 5, "linear", 0.7, [0.999], n_restarts, steps, batch)[0]
metrics["num_features_represented__linear_n20_m5"] = lin20["count"]
rep = lin20["norms"] ** 2 > 0.5
metrics["superposition_metric__linear_n20_m5"] = float(np.mean(lin20["sup"][rep])) if rep.any() else 0.0
inter["linear_n20_m5"] = dict(frob2=round(lin20["frob2"], 3),
                              norms=[round(float(v), 3) for v in lin20["norms"]],
                              sup=[round(float(v), 4) for v in lin20["sup"]])

# ---------- 3. Linear n=80 m=20 ----------
r80 = min(n_restarts, 3)
scale_note["restarts_n80"] = r80
lin80 = run_group(80, 20, "linear", 0.9, [0.999], r80, steps, batch)[0]
metrics["num_features_represented__linear_n80_m20"] = lin80["count"]
inter["linear_n80_m20"] = dict(frob2=round(lin80["frob2"], 3))

# ---------- 4. Uniform superposition n=400 m=30 ----------
ngrid = {"20_log_spaced": 20, "40_log_spaced": 40, "100_log_spaced": 100}.get(ugrid_opt, 40)
ngrid_used = 4 if SMOKE else 10
scale_note["uniform_grid_points"] = ngrid_used
scale_note["uniform_grid_points_spec"] = ngrid
inv = np.unique(np.round(np.concatenate([
    np.geomspace(1.0, 10.0, ngrid_used - 3), [3.0, 3.5, 4.0]]), 4))
dens_u = 1.0 / inv
u_steps = max(60, int(steps * (0.5 if not SMOKE else 1.0)))
u_batch = min(batch, 512)
scale_note["uniform_steps"] = u_steps
scale_note["uniform_batch"] = u_batch
ru = run_group(400, 30, "relu", 1.0, [1 - d for d in dens_u], 1, u_steps, u_batch, uniform_imp=True)
dstar = []
for iv, s in zip(inv, ru):
    dstar.append(dict(inv_density=float(iv), dstar=30.0 / max(s["frob2"], 1e-9),
                      loss=round(s["loss"], 5)))
inter["uniform_dstar_curve"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
                                for d in dstar]
metrics["dimensions_per_feature__uniform_dense"] = float(dstar[0]["dstar"])
sticky = [d["dstar"] for d in dstar if 2.9 <= d["inv_density"] <= 4.1]
metrics["dimensions_per_feature__uniform_sticky_half"] = float(np.mean(sticky)) if sticky else float("nan")
# per-feature dimensionality at the sticky point closest to 3.5
k = int(np.argmin(np.abs(inv - 3.5)))
di = ru[k]["dim"]
nk = ru[k]["norms"]
sel = di[(nk ** 2 > 0.3) & (di > 0.25) & (di < 0.75)]
if sel.size == 0:
    sel = di[nk ** 2 > 0.3]
if sel.size == 0:
    sel = di
metrics["feature_dimensionality__antipodal_pair"] = float(np.median(sel))
inter["dim_hist_sticky"] = dict(inv_density=float(inv[k]), n_sel=int(sel.size),
                                median=float(np.median(sel)) if sel.size else None)

# aliases for plain metric names (most-headline variant)
metrics["num_features_represented"] = metrics["num_features_represented__relu_output_n20_m5_s099"]
metrics["superposition_metric"] = metrics["superposition_metric__linear_n20_m5"]
metrics["dimensions_per_feature"] = metrics["dimensions_per_feature__uniform_sticky_half"]
metrics["feature_dimensionality"] = metrics["feature_dimensionality__antipodal_pair"]

inter["runtime_s"] = round(time.time() - t0, 1)
metrics["_intermediates"] = inter

res = {"seed": SEED, "split": str(CFG.get("split", "test")),
       "n_examples": int(steps * batch), "shuffled": SHUFFLE, "metrics": metrics}
with open(OUT, "w") as f:
    json.dump(res, f, indent=1, default=float)
print(json.dumps({k: v for k, v in metrics.items() if k != "_intermediates"}, indent=1))
print("runtime", inter["runtime_s"])
