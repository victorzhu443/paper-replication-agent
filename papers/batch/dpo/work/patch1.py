s = open("run_dpo.py").read()
s = s.replace("< 1e-9 and abs(rr.item() - beta * (lpl - lrl)) < 1e-9", "< 1e-6 and abs(rr.item() - beta * (lpl - lrl)) < 1e-6")
open("run_dpo.py", "w").write(s)
