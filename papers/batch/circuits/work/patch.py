import re
s = open("train.py").read()

s = s.replace('''def make_batch(bs, gen):
    core = torch.randint(0, VOCAB, (bs, REP_LEN), generator=gen)
    seq = torch.cat([torch.full((bs, 1), START_TOK), core, core, core], dim=1)
    return seq''',
'''PERIOD_LO = int(cfg("scale.period_lo", 25))
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
    return out''')

s = s.replace('''    pm = []
    for a in attns:  # per layer: B,H,T,T
        B, H, T2, _ = a.shape
        sc = np.zeros(H)
        cnt = 0
        for i in range(REP_LEN + 1, SEQ - 1):
            j = i - REP_LEN + 1  # position after earlier copy of token i
            sc += a[:, :, i, j].mean(0).numpy()
            cnt += 1
        pm.append(sc / max(cnt, 1))''',
'''    # prefix-matching mask: query i attends to key j (j<i) such that toks[j-1] == toks[i]
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
        pm.append(((s_ * w).sum((0, 2)) / w.sum().clamp(min=1)).numpy())''')

s = s.replace('BASE_STEPS = int(cfg("scale.steps", 3000))', 'BASE_STEPS = int(cfg("scale.steps", 2500))')
open("train.py", "w").write(s)
print("ok")
