import json, os, subprocess, sys, time
smoke = sys.argv[1] if len(sys.argv) > 1 else "1"
env = {**os.environ, "SEED": "0", "SMOKE": smoke, "REPLICATOR_CONFIG": "{}", "METRICS_OUT": f"metrics_local_{smoke}.json"}
t = time.time()
p = subprocess.run(["bash", "reproduce.sh"], env=env, capture_output=True, text=True)
print("rc", p.returncode, f"{time.time()-t:.0f}s")
print(p.stdout[-4000:])
print("ERR", p.stderr[-3000:])
if p.returncode == 0:
    d = json.load(open(env["METRICS_OUT"]))
    m = {k: v for k, v in d["metrics"].items() if k != "_intermediates"}
    print(json.dumps({**{k: d[k] for k in ("seed", "split", "n_examples")}, "metrics": m}, indent=1))
    print("scale:", json.dumps(d["metrics"]["_intermediates"]["scale"], indent=1)[:1200])
