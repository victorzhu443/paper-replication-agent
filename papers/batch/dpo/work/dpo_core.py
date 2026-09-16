"""Core DPO pieces: tiny transformer LM, synthetic preference data, Eq.7 loss."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- synthetic world ----------------
VOCAB = 32
POS = list(range(1, 16))    # "target attribute" tokens
NEG = list(range(16, 32))   # non-target tokens
PROMPT_LEN = 4
COMP_LEN = 8


def attribute_score(comp: np.ndarray) -> np.ndarray:
    """Ground-truth scorer: fraction of target-attribute tokens in the completion."""
    return (comp < 16).mean(axis=-1)


def sample_prompts(rng, n):
    return rng.integers(1, VOCAB, size=(n, PROMPT_LEN))


# ---------------- model ----------------
class TinyLM(nn.Module):
    def __init__(self, d=64, nlayer=2, nhead=4, maxlen=PROMPT_LEN + COMP_LEN):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.0,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.head = nn.Linear(d, VOCAB)
        self.maxlen = maxlen

    def forward(self, x):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        h = self.enc(h, mask=mask)
        return self.head(h)


def seq_logprob(model, seq, n_prompt=PROMPT_LEN, aggregation="sum", prompt_masking="completion_only"):
    """log pi(y|x) over completion tokens (or all tokens if prompt_and_completion)."""
    logits = model(seq[:, :-1])
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, seq[:, 1:, None]).squeeze(-1)  # (B, T-1)
    if prompt_masking == "completion_only":
        tok_lp = tok_lp[:, n_prompt - 1:]
    if aggregation == "mean":
        return tok_lp.mean(-1)
    return tok_lp.sum(-1)


# ---------------- DPO loss (Eq. 7) ----------------
def dpo_loss(pi_lp_w, pi_lp_l, ref_lp_w, ref_lp_l, beta):
    pi_logratios = pi_lp_w - pi_lp_l
    ref_logratios = ref_lp_w - ref_lp_l
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (pi_lp_w - ref_lp_w).detach()
    rejected_rewards = beta * (pi_lp_l - ref_lp_l).detach()
    return losses, chosen_rewards, rejected_rewards


@torch.no_grad()
def generate(model, prompts, temperature=0.0, decoding="pure_temperature_sampling", gen=None):
    x = prompts.clone()
    for _ in range(COMP_LEN):
        logits = model(x)[:, -1].float()
        if temperature <= 0:
            nxt = logits.argmax(-1)
        else:
            logits = logits / temperature
            if decoding == "top_k_50":
                v, _ = torch.topk(logits, min(50, logits.shape[-1]), dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], -1e9)
            elif decoding == "top_p_0.95":
                s, idx = torch.sort(logits, descending=True, dim=-1)
                p = torch.softmax(s, -1).cumsum(-1)
                s = s.masked_fill(p - torch.softmax(s, -1) > 0.95, -1e9)
                logits = torch.full_like(logits, -1e9).scatter(-1, idx, s)
            probs = torch.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        x = torch.cat([x, nxt[:, None]], dim=1)
    return x


@torch.no_grad()
def seq_kl(policy, ref, seqs, n_prompt=PROMPT_LEN):
    """sequence-level KL(pi||pi_ref) = sum over completion timesteps of per-step KL."""
    lp = F.log_softmax(policy(seqs[:, :-1]).float(), -1)[:, n_prompt - 1:]
    lr = F.log_softmax(ref(seqs[:, :-1]).float(), -1)[:, n_prompt - 1:]
    kl = (lp.exp() * (lp - lr)).sum(-1).sum(-1)
    return kl.mean().item()
