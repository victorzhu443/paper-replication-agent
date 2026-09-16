p = "run_experiments.py"
s = open(p).read()
old = """sel = di[(nk ** 2 > 0.3) & (di > 0.25) & (di < 0.75)]
metrics["feature_dimensionality__antipodal_pair"] = float(np.median(sel)) if sel.size else float("nan")"""
new = """sel = di[(nk ** 2 > 0.3) & (di > 0.25) & (di < 0.75)]
if sel.size == 0:
    sel = di[nk ** 2 > 0.3]
if sel.size == 0:
    sel = di
metrics["feature_dimensionality__antipodal_pair"] = float(np.median(sel))"""
assert old in s
open(p, "w").write(s.replace(old, new))
print("ok")
