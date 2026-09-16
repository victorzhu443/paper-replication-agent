import os, subprocess, json, time
env = dict(os.environ, SEED="0", SMOKE="1", SCALE="1.0", METRICS_OUT="metrics_smoke.json",
           REPLICATOR_CONFIG="{}")
t = time.time()
p = subprocess.run(["bash", "reproduce.sh"], env=env, capture_output=True, text=True)
print(p.stdout[-3000:], p.stderr[-3000:], time.time() - t)
