from __future__ import annotations
import json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from dpo_core import (VOCAB, POS, NEG, PROMPT_LEN, COMP_LEN, TinyLM, seq_logprob, dpo_loss,
                      generate, seq_kl, attribute_score, sample_prompts)

torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))

SEED = int(os.environ.get("SEED", "0"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
SCALE = float(os.environ.get("SCALE", "1.0"))
CFG = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}")
OUT = os.environ.get("METRICS_OUT", "metrics.json")

def cfg(k, d):
    v = CFG.get(k, d)
    return d if v is None else v

# ----- ambiguity config keys (defaults = spec defaults) -----
C = dict(
    lr_schedule=cfg("lr_schedule", "constant"),
    augmentation=cfg("augmentation", "none"),
    batch_vs_steps=cfg("batch_vs_steps", "steps"),
    batch_unit=cfg("batch_unit", "pairs"),
    mixed_precision=cfg("mixed_precision", "fp32"),
    split=cfg("split", "test"),
    logprob_aggregation=cfg("logprob_aggregation", "sum"),
    prompt_masking=cfg("prompt_masking", "completion_only"),
    beta=cfg("beta", 0.1),
    optimizer=cfg("optimizer", "rmsprop_1e-6_as_paper"),
    max_steps=int(cfg("max_steps", 1000)),
    max_length=int(cfg("max_length", 128)),
    eval_temperature=float(cfg("eval_temperature", 0.0)),
    decoding_params=cfg("decoding_params", "pure_temperature_sampling"),
    n_eval_prompts=int(cfg("n_eval_prompts", 256)),
    gpt4_winrate_denominator=cfg("gpt4_winrate_denominator", "human_judgment_set"),
    sft_stopping=cfg("sft_stopping", "one_epoch_subset"),
    judge=cfg("judge", "ground_truth_attribute_classifier"),
    tie_handling=cfg("tie_handling", "discard"),
    n_seeds=int(cfg("n_seeds", 3)),
    preference_rule=cfg("preference_rule", "deterministic_attribute_rule"),
    grad_clip=cfg("grad_clip", "none"),
    regularization=cfg("regularization", "model_default"),
    beta_grid=cfg("beta_grid", [0.05, 0.1, 0.5]),
    lr_multiplier=float(cfg("lr_multiplier", 60.0)),  # CPU-scale override, recorded
)
SHUFFLE = bool(CFG.get("_shuffle_labels", False))

# ----- scale -----
batch_pairs = 64 if C["batch_unit"] == "pairs" else 32
n_prefix = 2000
steps = C["max_steps"]
sft_steps = 400
if SMOKE:
    steps, sft_steps, n_prefix = 40, 40, 200
    C["n_eval_prompts"] = 64
    C["beta_grid"] = [0.05, 0.1, 0.5]
else:
    steps = max(20, int(round(steps * SCALE)))
    sft_steps = max(20, int(round(sft_steps * SCALE)))
    n_prefix = max(200, int(round(n_prefix * SCALE)))
    if C["batch_vs_steps"] == "epochs":
        steps = max(20, int(round(steps * SCALE)))

# ================= unit tests of Eq. 7 =================
def unit_tests():
    beta = 0.3
    lpw, lpl, lrw, lrl = 1.2, -0.7, 0.4, 0.1
    losses, cr, rr = dpo_loss(*[torch.tensor([v]) for v in (lpw, lpl, lrw, lrl)], beta)
    closed = -math.log(1 / (1 + math.exp(-beta * ((lpw - lrw) - (lpl - lrl)))))
    ok_loss = abs(losses.item() - closed) < 1e-6
    ok_rew = abs(cr.item() - beta * (lpw - lrw)) < 1e-6 and abs(rr.item() - beta * (lpl - lrl)) < 1e-6
    # gradient test: d loss/d lp_w < 0 (increase logp(yw)), d loss/d lp_l > 0,
    # magnitude = beta * sigmoid(r_l - r_w)
    a = torch.tensor([lpw], requires_grad=True); b = torch.tensor([lpl], requires_grad=True)
    l, _, _ = dpo_loss(a, b, torch.tensor([lrw]), torch.tensor([lrl]), beta)
    l.sum().backward()
    w = beta * torch.sigmoid(torch.tensor(beta * ((lpl - lrl) - (lpw - lrw)))).item()
    ok_grad = (a.grad.item() < 0 < b.grad.item() and abs(a.grad.item() + w) < 1e-6
               and abs(b.grad.item() - w) < 1e-6)
    return dict(loss_closed_form=bool(ok_loss), implicit_reward=bool(ok_rew),
                gradient_direction_and_weight=bool(ok_grad))

UT = unit_tests()
assert all(UT.values()), UT

# ================= data =================
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
gen = torch.Generator().manual_seed(SEED)

def base_completions(n, p_pos=0.3):
    pick = rng.random((n, COMP_LEN)) < p_pos
    pos = rng.choice(POS, size=(n, COMP_LEN)); neg = rng.choice(NEG, size=(n, COMP_LEN))
    return np.where(pick, pos, neg)

# --- "SFT" reference model: fit tiny LM on base-distribution sequences (one pass) ---
ref = TinyLM()
opt = torch.optim.Adam(ref.parameters(), lr=3e-3)
for s in range(sft_steps):
    pr = sample_prompts(rng, 128); co = base_completions(128)
    seq = torch.tensor(np.concatenate([pr, co], 1), dtype=torch.long)
    logits = ref(seq[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)

# --- preference pairs: 4 completions per prefix from pi_ref, 6 pairs, label by scorer ---
prompts = torch.tensor(sample_prompts(rng, n_prefix), dtype=torch.long)
samples = []
with torch.no_grad():
    for k in range(4):
        samples.append(generate(ref, prompts, temperature=1.0, gen=gen))
S = torch.stack(samples, 1)  # (N,4,T)
scores = attribute_score(S[:, :, PROMPT_LEN:].numpy())
W, L = [], []
for i in range(n_prefix):
    for a in range(4):
        for b in range(a + 1, 4):
            sa, sb = scores[i, a], scores[i, b]
            if sa == sb:
                continue  # ties discarded (deterministic attribute rule)
            if sa > sb:
                W.append(S[i, a]); L.append(S[i, b])
            else:
                W.append(S[i, b]); L.append(S[i, a])
W = torch.stack(W); L = torch.stack(L)
n_pairs = W.shape[0]
# held-out preference pairs
split_at = int(0.9 * n_pairs)
Wtr, Ltr, Wte, Lte = W[:split_at], L[:split_at], W[split_at:], L[split_at:]

# ================= DPO training =================
def lr_for(step):
    base = 1e-6 * C["lr_multiplier"]
    if C["optimizer"] == "rmsprop_1e-5":
        base = 1e-5 * C["lr_multiplier"]
    warm = min(1.0, (step + 1) / 150.0)
    lr = base * warm
    if C["lr_schedule"] == "cosine":
        lr = lr * 0.5 * (1 + math.cos(math.pi * step / max(1, steps)))
    elif C["lr_schedule"] == "step":
        lr = lr * (0.1 ** (step // max(1, steps // 3)))
    return lr


def train_dpo(beta):
    import copy
    pol = copy.deepcopy(ref)
    for p in pol.parameters():
        p.requires_grad_(True)
    pol.train()
    if C["optimizer"].startswith("adamw"):
        opt = torch.optim.AdamW(pol.parameters(), lr=1e-5 * C["lr_multiplier"],
                                weight_decay=0.01 if C["regularization"] == "weight_decay_0.01" else 0.0)
    else:
        opt = torch.optim.RMSprop(pol.parameters(), lr=1e-6)
    g = torch.Generator().manual_seed(SEED + 7)
    margins = []
    ntr = Wtr.shape[0]
    for st in range(steps):
        for grp in opt.param_groups:
            grp["lr"] = lr_for(st)
        idx = torch.randint(0, ntr, (batch_pairs,), generator=g)
        w, l = Wtr[idx], Ltr[idx]
        if SHUFFLE:  # shuffle preference labels within the batch before training
            flip = torch.rand(batch_pairs, generator=g) < 0.5
            w2 = torch.where(flip[:, None], l, w); l2 = torch.where(flip[:, None], w, l)
            w, l = w2, l2
        kw = dict(aggregation=C["logprob_aggregation"], prompt_masking=C["prompt_masking"])
        plw = seq_logprob(pol, w, **kw); pll = seq_logprob(pol, l, **kw)
        with torch.no_grad():
            rlw = seq_logprob(ref, w, **kw); rll = seq_logprob(ref, l, **kw)
        losses, cr, rr = dpo_loss(plw, pll, rlw, rll, beta)
        loss = losses.mean()
        opt.zero_grad(); loss.backward()
        if C["grad_clip"] != "none":
            torch.nn.utils.clip_grad_norm_(pol.parameters(), float(C["grad_clip"]))
        opt.step()
        margins.append((cr - rr).mean().item())
    pol.eval()
    return pol, margins


def win_rate(pol, temp):
    ep = torch.tensor(sample_prompts(np.random.default_rng(SEED + 99), C["n_eval_prompts"]),
                      dtype=torch.long)
    g = torch.Generator().manual_seed(SEED + 5)
    a = generate(pol, ep, temperature=temp, decoding=C["decoding_params"], gen=g)
    b = generate(ref, ep, temperature=temp, decoding=C["decoding_params"], gen=g)
    sa = attribute_score(a[:, PROMPT_LEN:].numpy()); sb = attribute_score(b[:, PROMPT_LEN:].numpy())
    win = sa > sb; tie = sa == sb
    if C["tie_handling"] == "discard":
        d = int((~tie).sum())
        wr = 100.0 * float(win.sum()) / d if d else 50.0
    elif C["tie_handling"] == "count_half":
        wr = 100.0 * float(win.sum() + 0.5 * tie.sum()) / len(sa)
    else:
        wr = 100.0 * float(win.sum()) / len(sa)
    return wr, float(sa.mean()), float(sb.mean()), seq_kl(pol, ref, a), float(tie.mean())


def pref_acc(pol, beta):
    kw = dict(aggregation=C["logprob_aggregation"], prompt_masking=C["prompt_masking"])
    with torch.no_grad():
        m = (beta * (seq_logprob(pol, Wte, **kw) - seq_logprob(ref, Wte, **kw))
             - beta * (seq_logprob(pol, Lte, **kw) - seq_logprob(ref, Lte, **kw)))
    return float((m > 0).float().mean()), float(m.mean())


t0 = time.time()
grid = [float(b) for b in C["beta_grid"]]
main_beta = float(C["beta"])
if main_beta not in grid:
    grid = sorted(grid + [main_beta])
res = {}
for beta in grid:
    pol, margins = train_dpo(beta)
    wr, sp, sr, kl, tier = win_rate(pol, C["eval_temperature"])
    wr25, _, _, _, _ = win_rate(pol, 0.25)
    acc, mm = pref_acc(pol, beta)
    res[beta] = dict(win_rate=wr, win_rate_temp025=wr25, pref_acc=100.0 * acc,
                     margin_mean_test=mm, margin_first=float(np.mean(margins[:5])),
                     margin_last=float(np.mean(margins[-5:])), kl=kl,
                     policy_score=sp, ref_score=sr, tie_frac=tier)

m = res[main_beta]
metrics = {
    "win_rate": m["win_rate"],
    "win_rate_temp025": m["win_rate_temp025"],
    "agreement_rate": m["pref_acc"],           # scorer/implicit-reward agreement on held-out pairs
    "held_out_pref_accuracy": m["pref_acc"],
    "implicit_reward_margin_final": m["margin_last"],
    "implicit_reward_margin_delta": m["margin_last"] - m["margin_first"],
    "sequence_kl": m["kl"],
    "mean_attribute_score_policy": m["policy_score"],
    "mean_attribute_score_ref": m["ref_score"],
    "win_rate_grows_with_beta": float(
        res[min(grid)]["win_rate"] <= res[max(grid)]["win_rate"]),
}
for b in grid:
    metrics[f"win_rate_beta_{b}"] = res[b]["win_rate"]
    metrics[f"kl_beta_{b}"] = res[b]["kl"]
    metrics[f"margin_last_beta_{b}"] = res[b]["margin_last"]

metrics["_intermediates"] = {
    "unit_tests": UT,
    "n_pairs_total": int(n_pairs), "n_pairs_train": int(Wtr.shape[0]),
    "n_pairs_test": int(Wte.shape[0]), "n_prefixes": int(n_prefix),
    "config_used": C, "train_seconds": time.time() - t0,
    "scale": {"SCALE": SCALE, "smoke": SMOKE, "dpo_steps": steps, "sft_steps": sft_steps,
              "batch_pairs": batch_pairs, "model": "TinyLM d=64 L=2 (substitute for GPT-J-6B)",
              "vocab": VOCAB, "seq_len": PROMPT_LEN + COMP_LEN,
              "data": "synthetic attribute preference pairs (substitute for TL;DR/IMDb)",
              "judge": "ground-truth attribute scorer (substitute for GPT-4)",
              "lr_multiplier_vs_paper_1e-6": C["lr_multiplier"],
              "n_eval_prompts": C["n_eval_prompts"], "beta_grid": grid},
}

json.dump({"seed": SEED, "split": C["split"], "n_examples": int(C["n_eval_prompts"]),
           "shuffled": SHUFFLE, "metrics": metrics}, open(OUT, "w"), indent=1)
print("wrote", OUT, {k: v for k, v in metrics.items() if not k.startswith("_")})
