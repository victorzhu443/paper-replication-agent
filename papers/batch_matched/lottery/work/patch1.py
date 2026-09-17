import re
s = open("lth.py").read()
old = """def _conv_cfgs(cfg, S, smoke):
    return dict(arch="conv4", lr=cfg.get("conv4_lr", 0.0003), bs=60, width=float(cfg.get("conv_width", 0.5)),
                iters=max(150, int(round((200 if smoke else 3000) * S))),
                eval_every=(50 if smoke else max(20, int(round(200 * S)))),
                rounds=(3 if smoke else int(cfg.get("conv4_rounds", 10))),
                n_val=(500 if smoke else 1500), n_test=(1000 if smoke else 2500),
                train_sub=(3000 if smoke else None))"""
new = """def _conv_cfgs(cfg, S, smoke):
    iters = max(120, int(round((120 if smoke else 3000) * S)))
    # keep the eval budget proportional to SCALE so cost scales linearly
    n_evals = 3 if smoke else max(4, int(round(15 * S)))
    return dict(arch="conv4", lr=cfg.get("conv4_lr", 0.0003), bs=60, width=float(cfg.get("conv_width", 0.5)),
                iters=iters, eval_every=max(10, iters // n_evals),
                rounds=(2 if smoke else int(cfg.get("conv4_rounds", 10))),
                n_val=(500 if smoke else 1500), n_test=(1000 if smoke else 2500),
                train_sub=(3000 if smoke else None))"""
assert old in s
open("lth.py", "w").write(s.replace(old, new))
print("ok")
