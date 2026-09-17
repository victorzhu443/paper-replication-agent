import os, subprocess, json, time, sys
env = {**os.environ, "SEED": "0", "SMOKE": "1", "SCALE": "1.0",
       "METRICS_OUT": "metrics_try.json", "REPLICATOR_CONFIG": "{}"}
t = time.time()
p = subprocess.run(["bash", "reproduce.sh"], env=env, capture_output=True, text=True)
print(p.returncode, "%.1fs" % (time.time() - t))
print(p.stdout[-3000:]); print(p.stderr[-4000:])
if os.path.exists("metrics_try.json"):
    d = json.load(open("metrics_try.json"))
    d["metrics"].pop("_intermediates")
    print(json.dumps(d, indent=1)[:2000])
