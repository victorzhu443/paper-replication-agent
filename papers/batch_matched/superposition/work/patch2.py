s = open('toy.py').read()
old_start = s.index("def adversarial(")
old_end = s.index("def cluster_mode(")
new = '''def attack_delta(W, b, dens_k, cfg, n, gen, nsamp=4096):
    """Loss increase (L' - L) of models W,b when inputs are drawn at density dens_k
    and perturbed by the analytic optimal per-feature L2 attack (worst feature)."""
    K = W.shape[0]
    d = torch.tensor(dens_k, dtype=torch.float32).view(K, 1, 1)
    I = torch.ones(1, 1, n)
    WtW = torch.bmm(W.transpose(1, 2), W)
    dirs = WtW / (WtW.norm(dim=2, keepdim=True) + 1e-9)   # row i = attack direction for feature i
    x0 = sample(K, 2048, n, d, gen)
    avg_norm = x0.norm(dim=2).mean(1)                     # average input L2 norm at this density
    scale = 0.1 * avg_norm
    if cfg["adv_attack_scale"] == "fixed_absolute_0.1":
        scale = torch.full((K,), 0.1)
    x = sample(K, nsamp, n, d, gen)
    with torch.no_grad():
        clean = (I * (forward(x, W, b, cfg) - x) ** 2).sum(-1).mean(1)
        worst = clean.clone()
        for i in range(n):
            delta = scale.view(K, 1, 1) * dirs[:, i, :].unsqueeze(1)
            l = (I * (forward(x + delta, W, b, cfg) - x) ** 2).sum(-1).mean(1)
            worst = torch.maximum(worst, l)
    return clean.numpy(), (worst - clean).numpy()


def baseline_model(W, b, cfg, n, m):
    """Non-superposition reference model for the adversarial ratio."""
    opt = cfg["adv_baseline_model"]
    if opt == "dense_end_model_of_same_sweep":
        return W[0:1].clone(), b[0:1].clone()
    # hand-constructed / linear: identity on the top-m features, zero elsewhere
    W0 = torch.zeros(1, m, n)
    for i in range(m):
        W0[0, i, i] = 1.0
    return W0, torch.zeros(1, 1, n)


def adversarial_curve(W, b, dens, cfg, n, m, seed):
    gen = torch.Generator().manual_seed(seed + 777)
    _, delta = attack_delta(W, b, dens, cfg, n, gen)
    Wb, bb = baseline_model(W, b, cfg, n, m)
    K = W.shape[0]
    Wb = Wb.expand(K, -1, -1).contiguous(); bb = bb.expand(K, -1, -1).contiguous()
    _, delta0 = attack_delta(Wb, bb, dens, cfg, n, gen)
    return delta / np.maximum(delta0, 1e-12)


'''
s = s[:old_start] + new + s[old_end:]
s = s.replace('''    clean, worst = adversarial(W, b, dens, cfg, n, seed)
    delta = worst - clean
    base = delta[0] if cfg["adv_baseline_model"] == "dense_end_model_of_same_sweep" else delta.max()
    ratio = delta / max(base, 1e-12)
    adv_ratio = float(np.max(ratio))''',
'''    ratio = adversarial_curve(W, b, dens, cfg, n, m, seed)
    adv_ratio = float(np.max(ratio))''')
open('toy.py', 'w').write(s)
print('ok')
