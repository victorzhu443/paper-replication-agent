"""Reduced-scale re-implementation of "Attention Is All You Need" (Vaswani et al. 2017).

Compute tier 2 (CPU only): WMT14 EN-DE / EN-FR and PTB parsing are out of reach, so the
paper-scale claims (Table 2 / Table 3 / Table 4) are recorded as UNTESTED.  What we do run:

  * a 2-layer encoder / 2-layer decoder Transformer (d_model=128, d_ff=512, h=4, post-norm,
    sinusoidal PE, tied embeddings, label smoothing 0.1, Adam(0.9,0.98,1e-9) with the paper's
    inverse-sqrt warmup schedule) on a synthetic sequence-transduction task (default: reverse),
  * the same skeleton with every attention sub-layer deleted (mean-pooled encoder context,
    position-wise FFN stacks only) as an internal no-attention control,
  * test-split cross-entropy loss, token accuracy, corpus BLEU (greedy + beam 4 / alpha 0.6),
    per-token perplexity, parameter count, and an analytic training-FLOPs estimate.

Every entry of the frozen ambiguity list is a config key read from REPLICATOR_CONFIG.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2
N_SYMBOLS = 32
SPECIALS = 3
VOCAB = SPECIALS + N_SYMBOLS + 2  # + two task-marker tokens (copy / reverse)
TASK_COPY, TASK_REV = SPECIALS + N_SYMBOLS, SPECIALS + N_SYMBOLS + 1


# --------------------------------------------------------------------------- config
DEFAULTS = {
    # ---- ambiguity list (frozen spec) ----
    "lr_schedule": "as_paper",
    "warmup_steps": "10_percent_of_total_steps",
    "attention_dropout": "same_as_residual_p_drop",
    "ppl_definition": "unsmoothed_ce",
    "length_penalty_form": "gnmt_wu2016",
    "vocab_sharing_enfr": "shared",
    "base_enfr_config": "same_as_base_ende_100k_pdrop0.1",
    "checkpoint_interval": "10_minutes_as_base",
    "augmentation": "none",
    "batch_vs_steps": "steps",
    "mixed_precision": "fp32",
    "split": "test",
    "reduced_task": "reverse",
    "bleu_tool": "sacrebleu_13a",
    "baseline_arch": "mean_pooled_context_ffn",
    "checkpoint_averaging": "none",
    "beam_size": "1_greedy",
    "init": "xavier_uniform",
    "grad_clip": "none",
    "norm_placement": "post_norm",
    "label_smoothing": "0.1_as_paper",
    "n_seeds": "3",
    "enfr_reference_value": "41.8_table2",
    "claim_status": "untested",
    # ---- reduced-scale knobs (not in the paper) ----
    "n_pairs": 20000,
    "min_len": 5,
    "max_len": 20,
    "d_model": 128,
    "d_ff": 512,
    "n_heads": 4,
    "n_layers": 2,
    "p_drop": 0.1,
    "train_steps": 1500,
    "batch_size": 64,
    "eval_decode_n": 1000,
    "beam_check_n": 100,
    "decode_max_offset": 5,
    "_shuffle_labels": False,
}


def get_cfg(user: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (user or {}).items()})
    return cfg


# --------------------------------------------------------------------------- data
def make_data(cfg, rng):
    """Unique random symbol sequences, disjoint 80/10/10 train/valid/test split."""
    n = int(cfg["n_pairs"])
    lo, hi = int(cfg["min_len"]), int(cfg["max_len"])
    task = cfg["reduced_task"]
    seen, seqs = set(), []
    while len(seqs) < n:
        L = int(rng.integers(lo, hi + 1))
        s = tuple(int(x) for x in rng.integers(SPECIALS, SPECIALS + N_SYMBOLS, size=L))
        if s in seen:
            continue
        seen.add(s)
        seqs.append(s)
    pairs = []
    for i, s in enumerate(seqs):
        if task == "copy":
            kind = "copy"
        elif task == "reverse":
            kind = "reverse"
        else:  # copy_and_reverse / iwslt fallback -> mixed with a task marker token
            kind = "copy" if i % 2 == 0 else "reverse"
        tgt = list(s) if kind == "copy" else list(s)[::-1]
        if task == "copy_and_reverse":
            src = [TASK_COPY if kind == "copy" else TASK_REV] + list(s)
        else:
            src = list(s)
        pairs.append((src, tgt))
    rng.shuffle(pairs)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    return pairs[:n_tr], pairs[n_tr:n_tr + n_va], pairs[n_tr + n_va:]


def batchify(pairs, idx):
    src = [pairs[i][0] for i in idx]
    tgt = [pairs[i][1] for i in idx]
    ls, lt = max(len(s) for s in src) + 1, max(len(t) for t in tgt) + 2
    S = torch.full((len(idx), ls), PAD, dtype=torch.long)
    T = torch.full((len(idx), lt), PAD, dtype=torch.long)
    for j, (s, t) in enumerate(zip(src, tgt)):
        S[j, :len(s) + 1] = torch.tensor(list(s) + [EOS])
        T[j, :len(t) + 2] = torch.tensor([BOS] + list(t) + [EOS])
    return S, T


# --------------------------------------------------------------------------- model
def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, p_attn):
        super().__init__()
        self.h, self.dk = h, d_model // h
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p_attn)

    def forward(self, q, kv, key_pad_mask=None, causal=False):
        B, Lq, _ = q.shape
        Lk = kv.shape[1]
        def split(x, L):
            return x.view(B, L, self.h, self.dk).transpose(1, 2)
        Q, K, V = split(self.wq(q), Lq), split(self.wk(kv), Lk), split(self.wv(kv), Lk)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.dk)
        if key_pad_mask is not None:
            scores = scores.masked_fill(key_pad_mask[:, None, None, :], float("-inf"))
        if causal:
            m = torch.triu(torch.ones(Lq, Lk, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(m[None, None], float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, Lq, self.h * self.dk)
        return self.wo(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class SubLayer(nn.Module):
    """LayerNorm(x + Dropout(Sublayer(x))) (post-norm, Section 3.1) or pre-norm variant."""
    def __init__(self, d_model, p_drop, post_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)
        self.post = post_norm

    def forward(self, x, fn):
        if self.post:
            return self.norm(x + self.drop(fn(x)))
        return x + self.drop(fn(self.norm(x)))


class Seq2Seq(nn.Module):
    """attention=True -> the paper's Transformer.  attention=False -> no-attention control."""
    def __init__(self, cfg, attention=True):
        super().__init__()
        d, dff, h = int(cfg["d_model"]), int(cfg["d_ff"]), int(cfg["n_heads"])
        N, p = int(cfg["n_layers"]), float(cfg["p_drop"])
        p_attn = p if cfg["attention_dropout"] == "same_as_residual_p_drop" else (
            0.0 if cfg["attention_dropout"] == "0.0" else 0.1)
        post = cfg["norm_placement"] == "post_norm"
        self.d_model, self.attention = d, attention
        self.emb = nn.Embedding(VOCAB, d, padding_idx=PAD)  # shared in/out + pre-softmax (3.4)
        self.learned_pe = None
        self.register_buffer("pe", sinusoidal_pe(512, d), persistent=False)
        self.emb_drop = nn.Dropout(p)
        self.enc, self.dec = nn.ModuleList(), nn.ModuleList()
        for _ in range(N):
            layer = nn.ModuleDict()
            if attention:
                layer["self"] = MultiHeadAttention(d, h, p_attn)
            else:
                layer["ffn0"] = FFN(d, dff)
            layer["ffn"] = FFN(d, dff)
            layer["s1"] = SubLayer(d, p, post)
            layer["s2"] = SubLayer(d, p, post)
            self.enc.append(layer)
        for _ in range(N):
            layer = nn.ModuleDict()
            if attention:
                layer["self"] = MultiHeadAttention(d, h, p_attn)
                layer["cross"] = MultiHeadAttention(d, h, p_attn)
            else:
                layer["ffn0"] = FFN(d, dff)
                layer["ffn1"] = FFN(d, dff)
            layer["ffn"] = FFN(d, dff)
            layer["s1"] = SubLayer(d, p, post)
            layer["s2"] = SubLayer(d, p, post)
            layer["s3"] = SubLayer(d, p, post)
            self.dec.append(layer)
        self.enc_norm = nn.LayerNorm(d) if not post else nn.Identity()
        self.dec_norm = nn.LayerNorm(d) if not post else nn.Identity()
        self._init(cfg)

    def _init(self, cfg):
        if cfg["init"] == "framework_default":
            return
        for n, prm in self.named_parameters():
            if prm.dim() > 1:
                if "emb" in n:
                    nn.init.normal_(prm, 0.0, self.d_model ** -0.5)
                elif cfg["init"] == "normal_0_0.02":
                    nn.init.normal_(prm, 0.0, 0.02)
                else:
                    nn.init.xavier_uniform_(prm)
            elif prm.dim() == 1 and "norm" not in n:
                nn.init.zeros_(prm)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()

    def embed(self, x):
        e = self.emb(x) * math.sqrt(self.d_model)
        e = e + self.pe[: x.shape[1]].unsqueeze(0)
        return self.emb_drop(e)

    def encode(self, src):
        pad = src.eq(PAD)
        x = self.embed(src)
        for l in self.enc:
            if self.attention:
                x = l["s1"](x, lambda y: l["self"](y, y, key_pad_mask=pad))
            else:
                x = l["s1"](x, l["ffn0"])
            x = l["s2"](x, l["ffn"])
        x = self.enc_norm(x)
        if self.attention:
            return x, pad
        keep = (~pad).float().unsqueeze(-1)
        ctx = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)  # fixed mean-pooled summary
        return ctx, pad

    def decode(self, mem, pad, tgt_in):
        x = self.embed(tgt_in)
        if not self.attention:
            x = x + mem.unsqueeze(1)
        for l in self.dec:
            if self.attention:
                x = l["s1"](x, lambda y: l["self"](y, y, causal=True))
                x = l["s2"](x, lambda y: l["cross"](y, mem, key_pad_mask=pad))
            else:
                x = l["s1"](x, l["ffn0"])
                x = l["s2"](x, l["ffn1"])
            x = l["s3"](x, l["ffn"])
        x = self.dec_norm(x)
        return x @ self.emb.weight.t()  # tied pre-softmax projection

    def forward(self, src, tgt_in):
        mem, pad = self.encode(src)
        return self.decode(mem, pad, tgt_in)


# --------------------------------------------------------------------------- losses
def losses(logits, gold, eps):
    mask = gold.ne(PAD)
    lp = F.log_softmax(logits.float(), dim=-1)
    nll = -lp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    ce = (nll * mask).sum() / mask.sum()
    if eps > 0:
        smooth = -lp.mean(dim=-1)
        loss = ((1 - eps) * nll + eps * smooth)
        loss = (loss * mask).sum() / mask.sum()
    else:
        loss = ce
    acc = ((logits.argmax(-1) == gold) & mask).sum() / mask.sum()
    return loss, ce.detach(), acc.detach(), int(mask.sum())


def lr_at(step, cfg, total):
    d = int(cfg["d_model"])
    w = cfg["warmup_steps"]
    if w == "4000_as_paper":
        warm = 4000
    elif w == "400":
        warm = 400
    else:
        warm = max(1, int(0.1 * total))
    base = d ** -0.5
    sched = cfg["lr_schedule"]
    if sched == "as_paper":
        return base * min(step ** -0.5, step * warm ** -1.5)
    peak = base * warm ** -0.5
    if sched == "constant":
        return peak
    if sched == "cosine":
        if step <= warm:
            return peak * step / warm
        t = (step - warm) / max(1, total - warm)
        return peak * 0.5 * (1 + math.cos(math.pi * t))
    if sched == "step":
        return peak * (0.5 ** ((step - 1) // max(1, total // 3)))
    return base * min(step ** -0.5, step * warm ** -1.5)


# --------------------------------------------------------------------------- decoding
def greedy(model, src, max_new):
    model.eval()
    with torch.no_grad():
        mem, pad = model.encode(src)
        B = src.shape[0]
        ys = torch.full((B, 1), BOS, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_new):
            nxt = model.decode(mem, pad, ys)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt.eq(EOS)
            if bool(done.all()):
                break
    return ys[:, 1:]


def beam_decode(model, src_row, max_new, beam, alpha, lp_form):
    """Single-sentence beam search with the GNMT length penalty (Wu et al. 2016)."""
    model.eval()
    with torch.no_grad():
        mem, pad = model.encode(src_row.unsqueeze(0))
        memb = mem.repeat(beam, *([1] * (mem.dim() - 1)))
        padb = pad.repeat(beam, 1)
        ys = torch.full((1, 1), BOS, dtype=torch.long)
        scores = torch.zeros(1)
        fin = []
        for _ in range(max_new):
            k = ys.shape[0]
            lp = F.log_softmax(model.decode(memb[:k], padb[:k], ys)[:, -1].float(), -1)
            lp[:, PAD] = -1e9
            cand = (scores.unsqueeze(1) + lp).view(-1)
            top = cand.topk(min(beam, cand.numel()))
            new_ys, new_sc = [], []
            for sc, flat in zip(top.values.tolist(), top.indices.tolist()):
                b, tok = flat // lp.shape[1], flat % lp.shape[1]
                seq = torch.cat([ys[b], torch.tensor([tok])])
                if tok == EOS:
                    fin.append((sc, seq))
                else:
                    new_ys.append(seq)
                    new_sc.append(sc)
            if not new_ys:
                break
            ys = torch.stack(new_ys)
            scores = torch.tensor(new_sc)
        if not fin:
            fin = [(float(scores[0]), ys[0])]
        def norm(sc, seq):
            L = len(seq) - 1
            if lp_form == "none":
                return sc
            if lp_form == "divide_by_length_alpha":
                return sc / max(L, 1) ** alpha
            return sc / (((5.0 + L) / 6.0) ** alpha)
        best = max(fin, key=lambda x: norm(*x))[1]
        out = [int(t) for t in best[1:] if int(t) not in (EOS, PAD, BOS)]
    return out


# --------------------------------------------------------------------------- BLEU
def _internal_bleu(hyps, refs, max_n=4):
    num = [0] * max_n
    den = [0] * max_n
    hl = rl = 0
    for h, r in zip(hyps, refs):
        hl += len(h)
        rl += len(r)
        for n in range(1, max_n + 1):
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            num[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            den[n - 1] += max(0, len(h) - n + 1)
    if min(den) == 0 or min(num) == 0:
        precs = [(num[i] + 1.0) / (den[i] + 1.0) for i in range(max_n)]  # add-1 smoothing
    else:
        precs = [num[i] / den[i] for i in range(max_n)]
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    return 100.0 * bp * math.exp(sum(math.log(p) for p in precs) / max_n)


def corpus_bleu(hyps, refs, tool):
    """hyps/refs are lists of token-id lists; symbols rendered as 's<k>' strings."""
    if tool in ("sacrebleu_13a",):
        try:
            import sacrebleu
            H = [" ".join("s%d" % t for t in h) for h in hyps]
            R = [" ".join("s%d" % t for t in r) for r in refs]
            return float(sacrebleu.corpus_bleu(H, [R], tokenize="13a").score)
        except Exception:
            pass
    return _internal_bleu(hyps, refs)


def token_f1(hyps, refs):
    tp = fp = fn = 0
    for h, r in zip(hyps, refs):
        hc, rc = Counter(h), Counter(r)
        inter = sum(min(c, rc[k]) for k, c in hc.items())
        tp += inter
        fp += len(h) - inter
        fn += len(r) - inter
    p = tp / max(tp + fp, 1)
    rr = tp / max(tp + fn, 1)
    return 100.0 * (0.0 if p + rr == 0 else 2 * p * rr / (p + rr))


# --------------------------------------------------------------------------- train/eval
def evaluate(model, pairs, cfg, eps, decode_n, beam_n):
    # teacher-forced loss / accuracy / perplexity over the whole split
    tot_ce = tot_loss = tot_tok = tot_corr = 0.0
    bs = 256
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            S, T = batchify(pairs, range(i, min(i + bs, len(pairs))))
            logits = model(S, T[:, :-1])
            gold = T[:, 1:]
            loss, ce, acc, ntok = losses(logits, gold, eps)
            tot_loss += float(loss) * ntok
            tot_ce += float(ce) * ntok
            tot_tok += ntok
            tot_corr += float(acc) * ntok
    ce = tot_ce / tot_tok
    ppl = math.exp(ce if cfg["ppl_definition"] == "unsmoothed_ce" else tot_loss / tot_tok)

    # greedy decode
    sub = pairs[:decode_n]
    hyps, refs = [], []
    off = int(cfg["decode_max_offset"])
    for i in range(0, len(sub), 128):
        chunk = sub[i:i + 128]
        S, _ = batchify(chunk, range(len(chunk)))
        out = greedy(model, S, S.shape[1] + off)
        for row, (_, t) in zip(out.tolist(), chunk):
            seq = []
            for tok in row:
                if tok in (EOS, PAD):
                    break
                seq.append(tok)
            hyps.append(seq)
            refs.append(list(t))
    res = {
        "loss": ce,
        "smoothed_loss": tot_loss / tot_tok,
        "perplexity": ppl,
        "token_accuracy": tot_corr / tot_tok,
        "bleu_greedy": corpus_bleu(hyps, refs, cfg["bleu_tool"]),
        "f1": token_f1(hyps, refs),
        "exact_match": float(np.mean([h == r for h, r in zip(hyps, refs)])),
        "n_decoded": len(hyps),
    }
    # secondary check: beam 4 / alpha 0.6 on a subsample
    if beam_n > 0:
        bh, br = [], []
        for src_tokens, tgt in pairs[:beam_n]:
            S, _ = batchify([(src_tokens, tgt)], [0])
            bh.append(beam_decode(model, S[0], S.shape[1] + off, 4, 0.6, cfg["length_penalty_form"]))
            br.append(list(tgt))
        res["bleu_beam4"] = corpus_bleu(bh, br, cfg["bleu_tool"])
        res["n_beam"] = len(bh)
    else:
        res["bleu_beam4"] = float("nan")
        res["n_beam"] = 0
    return res


def train_one(cfg, seed, attention, train, valid, test):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 991)
    model = Seq2Seq(cfg, attention=attention)
    n_params = sum(p.numel() for p in model.parameters())
    eps = 0.1 if cfg["label_smoothing"] == "0.1_as_paper" else 0.0
    opt = torch.optim.Adam(model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9)
    total = int(cfg["train_steps"])
    bs = int(cfg["batch_size"])
    t0 = time.time()
    tokens = 0
    model.train()
    for step in range(1, total + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg, total)
        idx = rng.integers(0, len(train), size=bs)
        S, T = batchify(train, idx)
        logits = model(S, T[:, :-1])
        loss, ce, acc, ntok = losses(logits, T[:, 1:], eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"] == "1.0_global_norm":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens += ntok + int(S.ne(PAD).sum())
    train_s = time.time() - t0
    split = test if cfg["split"] == "test" else valid
    ev = evaluate(model, split, cfg, eps, int(cfg["eval_decode_n"]), int(cfg["beam_check_n"]))
    ev["params"] = n_params
    ev["train_seconds"] = train_s
    ev["train_tokens"] = tokens
    # analytic FLOPs estimate: fwd+bwd ~ 6 * non-embedding-ish params * tokens processed
    ev["training_flops"] = 6.0 * n_params * tokens
    ev["final_train_loss"] = float(ce)
    return ev


def run(config: dict, seed: int) -> dict:
    cfg = get_cfg(config)
    smoke = bool(cfg.get("_smoke"))
    torch.set_num_threads(max(1, min(15, os.cpu_count() or 1)))
    n_seeds = 1 if smoke else int(cfg["n_seeds"])
    rng = np.random.default_rng(seed)
    train, valid, test = make_data(cfg, rng)
    if cfg.get("_shuffle_labels"):
        perm = rng.permutation(len(train))
        train = [(train[i][0], train[j][1]) for i, j in zip(range(len(train)), perm)]

    runs = {"transformer": [], "no_attention": []}
    for k in range(n_seeds):
        s = int(seed) * 100 + k
        runs["transformer"].append(train_one(cfg, s, True, train, valid, test))
        runs["no_attention"].append(train_one(cfg, s + 50, False, train, valid, test))

    def agg(name, key):
        v = np.array([r[key] for r in runs[name]], dtype=float)
        return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0

    metrics = {}
    for name, pre in (("transformer", "transformer"), ("no_attention", "no_attention")):
        for key in ("loss", "perplexity", "token_accuracy", "bleu_greedy", "bleu_beam4",
                    "exact_match", "f1", "params", "train_seconds", "training_flops"):
            m, sd = agg(name, key)
            metrics[f"{pre}_{key}"] = m
            metrics[f"{pre}_{key}_std"] = sd

    # headline metric names (claims' `metric` fields), measured at reduced scale
    metrics["bleu"] = metrics["transformer_bleu_greedy"]
    metrics["bleu_std"] = metrics["transformer_bleu_greedy_std"]
    metrics["perplexity"] = metrics["transformer_perplexity"]
    metrics["perplexity_std"] = metrics["transformer_perplexity_std"]
    metrics["f1"] = metrics["transformer_f1"]
    metrics["training_flops"] = metrics["transformer_training_flops"]
    metrics["param_count_millions"] = metrics["transformer_params"] / 1e6
    # attention-vs-no-attention gap (internal control, mirrors the paper's ablation logic)
    d = np.array([a["bleu_greedy"] - b["bleu_greedy"]
                  for a, b in zip(runs["transformer"], runs["no_attention"])], dtype=float)
    metrics["bleu_delta"] = float(d.mean())
    metrics["bleu_delta_std"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    metrics["loss"] = metrics["transformer_loss"]
    metrics["token_accuracy"] = metrics["transformer_token_accuracy"]
    metrics["n_seeds"] = float(n_seeds)

    split_pairs = test if cfg["split"] == "test" else valid
    inter = {
        "scale": {
            "paper_model": "N=6, d_model=512, d_ff=2048, h=8, 100K steps, WMT14 4.5M pairs",
            "ours": f"N={cfg['n_layers']}, d_model={cfg['d_model']}, d_ff={cfg['d_ff']}, "
                    f"h={cfg['n_heads']}, {cfg['train_steps']} steps, batch {cfg['batch_size']}",
            "task": f"synthetic {cfg['reduced_task']} of random symbol sequences "
                    f"(len {cfg['min_len']}-{cfg['max_len']}, {N_SYMBOLS} symbols)",
            "data_pairs": int(cfg["n_pairs"]),
            "tokens_per_batch_paper": 25000,
            "tokens_per_batch_ours": int(np.mean([r["train_tokens"] for r in runs["transformer"]])
                                        / int(cfg["train_steps"])),
            "seeds": n_seeds,
            "device": "cpu_fp32",
            "smoke": smoke,
        },
        "rows": {"train": len(train), "valid": len(valid), "test": len(test),
                 "report_split": cfg["split"], "n_report": len(split_pairs),
                 "n_decoded": runs["transformer"][0]["n_decoded"],
                 "n_beam": runs["transformer"][0]["n_beam"]},
        "vocab_size": VOCAB,
        "splits_disjoint": True,
        "shuffle_labels_control": bool(cfg.get("_shuffle_labels")),
        "untested_paper_scale_claims": [
            "c_t2_big_ende_bleu", "c_t2_big_enfr_bleu", "c_t2_base_ende_bleu",
            "c_t2_base_enfr_bleu", "c_t2_base_flops", "c_t2_big_flops",
            "c_t3_base_ppl_dev", "c_t3_base_bleu_dev", "c_t3_base_params",
            "c_t3_big_ppl_dev", "c_t3_big_bleu_dev", "c_t3_big_params",
            "c_t3_A_h1_ppl", "c_t3_A_h1_bleu", "c_t3_A_h16_bleu",
            "c_text_single_head_delta", "c_t3_C_N2_ppl", "c_t3_C_N2_bleu",
            "c_t3_D_drop0_ppl", "c_t3_D_drop0_bleu", "c_t3_E_learned_pe_ppl",
            "c_t3_E_learned_pe_bleu", "c_t4_parse_wsj_only_f1", "c_t4_parse_semisup_f1",
        ],
        "claim_status": cfg["claim_status"],
        "config_used": {k: cfg[k] for k in DEFAULTS},
        "runtime_seconds": None,
    }
    return metrics, inter, len(split_pairs), cfg


def main():
    t0 = time.time()
    seed = int(os.environ.get("SEED", "0"))
    cfg_user = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}")
    smoke = os.environ.get("SMOKE", "0") == "1"
    if smoke:
        cfg_user.setdefault("_smoke", True)
        cfg_user.setdefault("n_pairs", 1200)
        cfg_user.setdefault("train_steps", 40)
        cfg_user.setdefault("eval_decode_n", 64)
        cfg_user.setdefault("beam_check_n", 8)
        cfg_user.setdefault("batch_size", 32)
    metrics, inter, n_ex, cfg = run(cfg_user, seed)
    inter["runtime_seconds"] = round(time.time() - t0, 1)
    out = {
        "seed": seed,
        "split": cfg["split"],
        "n_examples": int(n_ex),
        "metrics": {k: (float(v) if v is not None else float("nan")) for k, v in metrics.items()},
        "_intermediates": inter,
    }
    path = os.environ.get("METRICS_OUT", "metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    with open("intermediates.json", "w") as f:
        json.dump(inter, f, indent=1)
    print(json.dumps({k: out["metrics"][k] for k in
                      ("bleu", "perplexity", "token_accuracy", "bleu_delta",
                       "no_attention_bleu_greedy", "param_count_millions")}, indent=1))
    print("runtime_s", inter["runtime_seconds"])


if __name__ == "__main__":
    main()
