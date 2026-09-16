p="trace_rome.py"; s=open(p).read()
s = s.replace('cfg_get(cfg, "model.name", "gpt2")', 'cfg_get(cfg, "model.name", "gpt2-medium")')
s = s.replace('int(cfg_get(cfg, "tracing.n_noise_draws", 3))', 'int(cfg_get(cfg, "tracing.n_noise_draws", 5))')
old = '''    if smoke:
        n_prompts, n_noise = 2, 1
    else:
        n_prompts = max(2, int(round(n_prompts * min(scale, 1.0)))) if scale < 1.0 else n_prompts
        n_noise = max(1, int(round(n_noise * min(scale, 1.0)))) if scale < 1.0 else n_noise'''
new = '''    if smoke:
        n_prompts, n_noise = 2, 1
    else:
        n_prompts = min(len(FACTS), max(2, int(round(n_prompts * scale))))
        n_noise = max(1, int(round(n_noise * scale)))'''
assert old in s; s = s.replace(old, new)
s = s.replace('            if float(loss) < 5e-2:', '            if float(loss.detach()) < 5e-2:')
s = s.replace('"final_loss": float(loss) if steps else None', '"final_loss": float(loss.detach()) if steps else None')
open(p,"w").write(s); print("ok")
