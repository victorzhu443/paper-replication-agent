s = open("lth.py").read()
old_l = """                iters=max(200, int(round((300 if smoke else 50000) * S))),
                eval_every=max(10, int(round(100 * S))) if not smoke else 50,"""
new_l = """                iters=max(200, int(round((300 if smoke else 50000) * S))),
                eval_every=(50 if smoke else 100),  # paper: evaluate every 100 iterations"""
assert old_l in s
s = s.replace(old_l, new_l)
old_c = """    iters = max(120, int(round((120 if smoke else 3000) * S)))"""
new_c = """    iters = max(120, int(round((120 if smoke else 4000) * S)))"""
assert old_c in s
s = s.replace(old_c, new_c)
old_w = 'width=float(cfg.get("conv_width", 0.5))'
assert old_w in s
s = s.replace(old_w, 'width=float(cfg.get("conv_width", 0.25))')
open("lth.py", "w").write(s)
r = open("run_lth.py").read().replace("half width (32/32/64/64 channels)", "quarter width (16/16/32/32 channels)")
open("run_lth.py", "w").write(r)
print("ok")
