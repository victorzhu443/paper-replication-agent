import os, subprocess, time, json, sys
sc = sys.argv[1] if len(sys.argv) > 1 else "0.3"
env = dict(os.environ, SMOKE="0", SEED="0", SCALE=sc, METRICS_OUT="metrics_t.json")
t = time.time()
p = subprocess.run(["bash", "reproduce.sh"], env=env, capture_output=True, text=True)
print(p.returncode, time.time() - t)
print(p.stdout[-2000:], p.stderr[-2000:])
d = json.load(open("metrics_t.json"))
print(d["metrics"]["_intermediates"]["dstar_curve"])
print(d["metrics"]["_intermediates"]["adv_ratio_curve"])
