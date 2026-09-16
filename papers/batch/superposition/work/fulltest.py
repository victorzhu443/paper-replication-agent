import os, subprocess, json, time
env = dict(os.environ, SEED="0", SMOKE="0", SCALE="1.0", METRICS_OUT="metrics_full.json",
           REPLICATOR_CONFIG="{}")
t = time.time()
p = subprocess.run(["bash", "reproduce.sh"], env=env, capture_output=True, text=True)
print(p.stdout[-4000:], p.stderr[-2000:], time.time() - t)
m = json.load(open("metrics_full.json"))["metrics"]["_intermediates"]
print(json.dumps(m["uniform_dstar_curve"], indent=0))
print(json.dumps(m["relu_n20_m5_sweep"], indent=0)[:2000])
