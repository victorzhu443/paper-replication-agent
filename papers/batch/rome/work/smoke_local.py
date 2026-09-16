import os, subprocess, json, time
env = dict(os.environ, SEED="0", SMOKE="1", SCALE="1.0", METRICS_OUT="m_smoke.json", REPLICATOR_CONFIG="{}")
t=time.time()
r = subprocess.run(["bash","reproduce.sh"], env=env, capture_output=True, text=True)
print(r.returncode, time.time()-t)
print(r.stdout[-3000:]); print(r.stderr[-3000:])
