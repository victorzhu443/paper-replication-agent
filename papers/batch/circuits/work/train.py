import json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("NTHREADS", "15")))

SEED = int(os.environ.get("SEED", "0"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
SCALE = float(os.environ.get("SCALE", "1.0"))
CFG = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}")
OUT = os.environ.get("METRICS_OUT", "metrics.json")
SHUFFLE = bool(CFG.get("_shuffle_labels", False))


def cfg(k, d):
    return CFG.get(k, d)


# ---------------- config (ambiguities) ----------------
C = dict(
    lr_schedule=cfg("lr_schedule", "cosine"),
    augmentation=cfg("augmentation", "none"),
    batch_vs_steps=cfg("batch_vs_steps", "steps"),
    mixed_precision=cfg("mixed_precision", "fp32"),
    split=cfg("split", "not_applicable"),
    training_corpus=cfg("data.training_corpus", "synthetic_repeated_random_tokens"),
    tokenizer=cfg("data.tokenizer", "gpt2_bpe_50257"),
    model_size=cfg("model.size", "paper_12x64_d768"),
    n_ctx=int(cfg("model.n_ctx", 2048)),
    positional=cfg("model.positional", "sinusoidal_into_qk_only"),
    attention_scale=cfg("model.attention_scale", "one_over_sqrt_dhead"),
    layer_norm=cfg("model.layer_norm", "ln_as_trained"),
    ln_folding=cfg("eval.ln_folding", "fold_ln_into_embedding"),
    biases=cfg("model.biases", "no_biases"),
    optimizer=cfg("train.optimizer", "adamw_lr1e-3_wd0.1_clip1.0"),
    token_budget=cfg("train.token_budget", "fixed_20k_steps"),
    copying_claim_model=cfg("eval.copying_claim_model", "one_layer_12head_64"),
    two_layer_heads=int(cfg("model.two_layer_heads_per_layer", 12)),
    copying_score=cfg("eval.copying_score", "sum_positive_eig_over_sum_abs_eig"),
    copying_threshold=cfg("eval.copying_threshold", "score_gt_0.7"),
    prefix_matching_score=cfg("eval.prefix_matching_score",
                              "mean_attn_to_token_after_earlier_copy_on_repeated_random"),
    induction_head_criterion=cfg("eval.induction_head_criterion", "positive_qk_and_ov_eigenvalue_corner"),
    random_seq=cfg("eval.random_seq", "len50_rep3_uniform_with_start"),
    loss_gap=cfg("eval.loss_gap", "mean_loss_second_half_minus_first_half_of_repeated_seq"),
    composition_baseline=cfg("eval.composition_baseline", "mean_over_N_random_gaussian_pairs_same_shape"),
    composition_formula=cfg("eval.composition_formula", "both"),
    n_seeds=int(cfg("train.n_seeds", 1)),
    n_eval_sequences=int(cfg("eval.n_eval_sequences", 1)),
    checkpoint=cfg("eval.checkpoint", "final"),
    matrix_algebra=cfg("eval.matrix_algebra", "factored_lambda_AB_equals_BA"),
    # replicator knobs
    comp_z_threshold=float(cfg("eval.comp_z_threshold", 5.0)),
    prefix_match_threshold=float(cfg("eval.prefix_match_threshold", 0.2)),
)

# ---------------- scaled-down setup (compute tier 2) ----------------
VOCAB = int(cfg("scale.vocab", 512))
N_HEADS = 12
D_HEAD = int(cfg("scale.d_head", 16))
D_MODEL = N_HEADS * D_HEAD
REP_LEN = int(cfg("scale.rep_len", 50))
N_REPS = 3
SEQ = REP_LEN * N_REPS + 1  # <START> + 3 repeats
BATCH = int(cfg("scale.batch", 32))
BASE_STEPS = int(cfg("scale.steps", 2500))
STEPS = max(5, int(round(BASE_STEPS * SCALE)))
N_EVAL = int(cfg("scale.n_eval_seq", 64))
if SMOKE:
    STEPS = max(5, int(20 * max(SCALE, 0.1)))
    N_EVAL = 8

LR = 1e-3
START_TOK = VOCAB  # special <START>
V_TOT = VOCAB + 1


PERIOD_LO = int(cfg("scale.period_lo", 25))
PERIOD_HI = int(cfg("scale.period_hi", 50))


def make_batch(bs, gen):
    """Repeated-random-token sequences; the repeat period is resampled per sequence so
    that purely positional (1-layer) attention cannot solve the copying task."""
    out = torch.empty(bs, SEQ, dtype=torch.long)
    out[:, 0] = START_TOK
    for b in range(bs):
        L = int(torch.randint(PERIOD_LO, PERIOD_HI + 1, (1,), generator=gen))
        core = torch.randint(0, VOCAB, (L,), generator=gen)
        reps = -(-(SEQ - 1) // L)
        out[b, 1:] = core.repeat(reps)[: SEQ - 1]
    return out


class AttnOnly(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.H, self.dh, self.dm = N_HEADS, D_HEAD, D_MODEL
        self.W_E = nn.Parameter(torch.randn(V_TOT, D_MODEL) * 0.02)
        self.W_U = nn.Parameter(torch.randn(D_MODEL, V_TOT) * 0.02)
        s = 1.0 / math.sqrt(D_MODEL)
        self.W_Q = nn.ParameterList([nn.Parameter(torch.randn(N_HEADS, D_MODEL, D_HEAD) * s) for _ in range(n_layers)])
        self.W_K = nn.ParameterList([nn.Parameter(torch.randn(N_HEADS, D_MODEL, D_HEAD) * s) for _ in range(n_layers)])
        self.W_V = nn.ParameterList([nn.Parameter(torch.randn(N_HEADS, D_MODEL, D_HEAD) * s) for _ in range(n_layers)])
        self.W_O = nn.ParameterList([nn.Parameter(torch.randn(N_HEADS, D_HEAD, D_MODEL) * s) for _ in range(n_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        pos = torch.zeros(SEQ, D_MODEL)
        p = torch.arange(SEQ).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-math.log(10000.0) / D_MODEL))
        pos[:, 0::2] = torch.sin(p * div)
        pos[:, 1::2] = torch.cos(p * div)
        self.register_buffer("pos", pos)
        mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
        self.register_buffer("mask", mask)
        self.use_ln = C["layer_norm"] == "ln_as_trained"
        self.scale = 1.0 / math.sqrt(D_HEAD) if C["attention_scale"] == "one_over_sqrt_dhead" else 1.0

    def read(self, x, l):
        return self.lns[l](x) if self.use_ln else x

    def forward(self, toks, want_attn=False):
        B, T = toks.shape
        x = self.W_E[toks]
        attns = []
        for l in range(self.n_layers):
            xr = self.read(x, l)
            xp = xr + self.pos[:T]
            q = torch.einsum("btm,hmd->bhtd", xp, self.W_Q[l])
            k = torch.einsum("btm,hmd->bhtd", xp, self.W_K[l])
            v = torch.einsum("btm,hmd->bhtd", xr, self.W_V[l])
            sc = torch.einsum("bhtd,bhsd->bhts", q, k) * self.scale
            sc = sc.masked_fill(~self.mask[:T, :T], float("-inf"))
            a = torch.softmax(sc, dim=-1)
            if want_attn:
                attns.append(a.detach())
            z = torch.einsum("bhts,bhsd->bhtd", a, v)
            x = x + torch.einsum("bhtd,hdm->btm", z, self.W_O[l])
        xf = self.ln_f(x) if self.use_ln else x
        logits = xf @ self.W_U
        return (logits, attns) if want_attn else logits


def loss_fn(logits, toks):
    return F.cross_entropy(logits[:, :-1].reshape(-1, V_TOT), toks[:, 1:].reshape(-1))


def train(n_layers, seed):
    torch.manual_seed(seed + 1000 * n_layers)
    gen = torch.Generator().manual_seed(seed + 7 * n_layers)
    m = AttnOnly(n_layers)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.1)
    for step in range(STEPS):
        toks = make_batch(BATCH, gen)
        tgt = toks.clone()
        if SHUFFLE:
            # shuffle targets within each sequence (period) before training
            for b in range(tgt.shape[0]):
                perm = torch.randperm(tgt.shape[1] - 1, generator=gen) + 1
                tgt[b, 1:] = tgt[b, perm]
        if C["lr_schedule"] == "cosine":
            lr = LR * 0.5 * (1 + math.cos(math.pi * step / max(1, STEPS)))
            warm = min(1.0, (step + 1) / max(1, int(0.05 * STEPS)))
            lr *= warm
        else:
            lr = LR
        for g in opt.param_groups:
            g["lr"] = lr
        logits = m(toks)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V_TOT), tgt[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


# ---------------- diagnostics ----------------
def folded_E(m):
    E = m.W_E.detach()
    if C["ln_folding"] == "fold_ln_into_embedding" and m.use_ln and m.n_layers > 0:
        En = (E - E.mean(-1, keepdim=True)) / (E.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        E = En * m.lns[0].weight.detach()
    return E


def copying_scores(m, layer=0):
    E, U = folded_E(m), m.W_U.detach()
    out = []
    for h in range(N_HEADS):
        OV = m.W_V[layer][h].detach() @ m.W_O[layer][h].detach()  # dm x dm
        M = (OV @ U @ E).numpy()  # eig equals eig of E OV U (vocab x vocab), nonzero part
        ev = np.linalg.eigvals(M)
        r = ev.real
        s = float(np.sum(np.abs(ev)))
        if C["copying_score"] == "fraction_of_eigs_with_positive_real_part":
            sc = float(np.mean(r > 0))
        elif C["copying_score"] == "trace_over_sum_abs_eig":
            sc = float(np.sum(r) / (s + 1e-12))
        else:
            sc = float(np.sum(r[r > 0]) / (s + 1e-12))
        out.append(sc)
    return np.array(out)


def copy_threshold():
    t = C["copying_threshold"]
    return 0.5 if t == "score_gt_0.5" else 0.7


def frob_scores(m, kind, formula):
    """Return H1 x H0 matrix of composition scores (layer1 head x layer0 head)."""
    res = np.zeros((N_HEADS, N_HEADS))
    for h2 in range(N_HEADS):
        Q = m.W_Q[1][h2].detach()  # dm x dh
        K = m.W_K[1][h2].detach()
        Ov2 = m.W_V[1][h2].detach() @ m.W_O[1][h2].detach()
        QK = Q @ K.T
        for h1 in range(N_HEADS):
            V0 = m.W_V[0][h1].detach()
            O0 = m.W_O[0][h1].detach()
            OV = V0 @ O0
            if kind == "K":
                facs_corrected = [QK, OV.T]
                if formula == "corrected":
                    num = torch.linalg.matrix_norm(QK @ OV.T)
                elif formula == "buggy_transposed_K_first":
                    num = torch.linalg.matrix_norm(K.T @ O0.T @ V0.T @ Q)
                else:  # buggy_transposed_V_first
                    num = torch.linalg.matrix_norm(O0.T @ V0.T @ Q @ K.T)
                den = torch.linalg.matrix_norm(QK) * torch.linalg.matrix_norm(OV)
            elif kind == "Q":
                if formula == "corrected":
                    num = torch.linalg.matrix_norm(OV @ QK)
                else:
                    num = torch.linalg.matrix_norm(K.T @ V0 @ O0 @ Q)
                den = torch.linalg.matrix_norm(QK) * torch.linalg.matrix_norm(OV)
            else:  # V
                num = torch.linalg.matrix_norm(Ov2 @ OV) if formula == "corrected" else \
                    torch.linalg.matrix_norm(m.W_O[1][h2].detach() @ OV @ m.W_V[1][h2].detach())
                den = torch.linalg.matrix_norm(Ov2) * torch.linalg.matrix_norm(OV)
            res[h2, h1] = float(num / (den + 1e-12))
    return res


def random_baseline(kind, formula, n=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n):
        Q = torch.randn(D_MODEL, D_HEAD, generator=g)
        K = torch.randn(D_MODEL, D_HEAD, generator=g)
        V0 = torch.randn(D_MODEL, D_HEAD, generator=g)
        O0 = torch.randn(D_HEAD, D_MODEL, generator=g)
        QK = Q @ K.T
        OV = V0 @ O0
        if formula == "corrected":
            num = torch.linalg.matrix_norm(QK @ OV.T) if kind != "Q" else torch.linalg.matrix_norm(OV @ QK)
        elif formula == "buggy_transposed_K_first":
            num = torch.linalg.matrix_norm(K.T @ O0.T @ V0.T @ Q)
        else:
            num = torch.linalg.matrix_norm(O0.T @ V0.T @ Q @ K.T)
        den = torch.linalg.matrix_norm(QK) * torch.linalg.matrix_norm(OV)
        vals.append(float(num / den))
    v = np.array(vals)
    return float(v.mean()), float(v.std() + 1e-9)


@torch.no_grad()
def behavioral(m, n_seq, seed=123):
    gen = torch.Generator().manual_seed(seed)
    toks = make_batch(n_seq, gen)
    logits, attns = m(toks, want_attn=True)
    lp = F.cross_entropy(logits[:, :-1].reshape(-1, V_TOT), toks[:, 1:].reshape(-1),
                         reduction="none").view(toks.shape[0], -1)
    T = lp.shape[1]
    first = float(lp[:, : T // 2].mean())
    second = float(lp[:, T // 2:].mean())
    r1 = float(lp[:, :REP_LEN].mean())
    r2 = float(lp[:, REP_LEN:2 * REP_LEN].mean())
    # prefix-matching: attention from position i (in reps 2,3) to position of the
    # token that FOLLOWED the earlier occurrence of token at i.
    # prefix-matching mask: query i attends to key j (j<i) such that toks[j-1] == toks[i]
    B, T2 = toks.shape
    prevtok = torch.full((B, T2), -1, dtype=torch.long)
    prevtok[:, 1:] = toks[:, :-1]
    match = (prevtok.unsqueeze(1) == toks.unsqueeze(2))  # B,i,j
    idx = torch.arange(T2)
    match &= (idx.view(1, 1, -1) < idx.view(1, -1, 1))
    qvalid = match.any(-1)  # queries with at least one matching key
    pm = []
    for a in attns:
        s_ = (a * match.unsqueeze(1).float()).sum(-1)  # B,H,T
        w = qvalid.unsqueeze(1).float()
        pm.append(((s_ * w).sum((0, 2)) / w.sum().clamp(min=1)).numpy())
    return dict(first_half=first, second_half=second, rep1=r1, rep2=r2, pm=pm,
                n_tokens=int(toks.numel()))


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    metrics = {}
    inter = {}

    m1 = train(1, SEED)
    m2 = train(2, SEED)

    # copying (one-layer model, 12 heads) -> headline claim
    cs1 = copying_scores(m1, 0)
    thr = copy_threshold()
    metrics["count_of_copying_heads"] = float(np.sum(cs1 > thr))
    metrics["copying_score_mean_1layer"] = float(cs1.mean())
    metrics["copying_score_max_1layer"] = float(cs1.max())

    cs2 = copying_scores(m2, 1)
    metrics["copying_score_max_2layer_layer1"] = float(cs2.max())
    metrics["copying_score_mean_2layer_layer1"] = float(cs2.mean())

    # behavioral
    b1 = behavioral(m1, N_EVAL)
    b2 = behavioral(m2, N_EVAL)
    metrics["loss_gap_1layer"] = b1["first_half"] - b1["second_half"]
    metrics["loss_gap_2layer"] = b2["first_half"] - b2["second_half"]
    metrics["first_half_loss_2layer"] = b2["first_half"]
    metrics["second_half_loss_2layer"] = b2["second_half"]
    metrics["first_half_loss_1layer"] = b1["first_half"]
    metrics["second_half_loss_1layer"] = b1["second_half"]
    pm2 = b2["pm"][1]
    pm1 = b1["pm"][0]
    metrics["prefix_matching_score_max_2layer"] = float(pm2.max())
    metrics["prefix_matching_score_max_1layer"] = float(pm1.max())

    # composition
    comp = {}
    for formula in ["corrected", "buggy_transposed_K_first", "buggy_transposed_V_first"]:
        S = frob_scores(m2, "K", formula)
        mu, sd = random_baseline("K", formula, seed=SEED)
        comp[formula] = (S - mu) / sd
    z = C["comp_z_threshold"]
    Zbug = comp["buggy_transposed_K_first"]
    Zcor = comp["corrected"]
    sig_bug_layer0 = [h1 for h1 in range(N_HEADS) if np.any(Zbug[:, h1] > z)]
    metrics["count_of_layer0_heads_with_significant_k_composition"] = float(len(sig_bug_layer0))

    # induction heads: high prefix matching AND positive copying (OV) eigenvalue score
    ind = [h for h in range(N_HEADS)
           if pm2[h] > C["prefix_match_threshold"] and cs2[h] > 0.5]
    metrics["n_induction_heads_2layer"] = float(len(ind))
    # primary previous-token head: layer-0 head with strongest prev-token attention
    prev_attn = None
    with torch.no_grad():
        gen = torch.Generator().manual_seed(999)
        toks = make_batch(4, gen)
        _, attns = m2(toks, want_attn=True)
        a0 = attns[0]
        prev = np.array([float(a0[:, h].diagonal(offset=-1, dim1=-2, dim2=-1).mean()) for h in range(N_HEADS)])
    top_prev = int(np.argmax(prev))
    extra = 0
    for h2 in ind:
        others = [h1 for h1 in range(N_HEADS) if h1 != top_prev and Zcor[h2, h1] > z]
        sig_bug = [h1 for h1 in range(N_HEADS) if h1 != top_prev and Zbug[h2, h1] > z]
        if len(others) > len(sig_bug):
            extra += 1
    metrics["count_of_induction_heads_composing_with_extra_layer0_head"] = float(extra)

    inter["scale"] = dict(vocab=V_TOT, d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD,
                          n_ctx=SEQ, paper_n_ctx=2048, paper_d_head=64, paper_vocab=50000,
                          steps=STEPS, batch=BATCH, scale=SCALE, smoke=SMOKE,
                          corpus="synthetic_repeated_random_tokens (tier C: paper corpus unavailable)")
    inter["copying_scores_1layer"] = [round(float(x), 4) for x in cs1]
    inter["copying_scores_2layer_layer1"] = [round(float(x), 4) for x in cs2]
    inter["prefix_matching_2layer_layer1"] = [round(float(x), 4) for x in pm2]
    inter["prev_token_attn_layer0"] = [round(float(x), 4) for x in prev]
    inter["top_prev_token_head"] = top_prev
    inter["induction_heads"] = ind
    inter["k_comp_z_corrected_max_per_layer0"] = [round(float(Zcor[:, h].max()), 3) for h in range(N_HEADS)]
    inter["k_comp_z_buggy_max_per_layer0"] = [round(float(Zbug[:, h].max()), 3) for h in range(N_HEADS)]
    inter["config"] = C
    inter["runtime_s"] = round(time.time() - t0, 1)
    metrics["_intermediates"] = inter

    res = dict(seed=SEED, split=str(C["split"]), n_examples=int(N_EVAL * SEQ),
               shuffled=bool(SHUFFLE), metrics=metrics)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "_intermediates"}, indent=1))
    print("runtime", inter["runtime_s"])


if __name__ == "__main__":
    main()
