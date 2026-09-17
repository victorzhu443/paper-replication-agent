import os, subprocess, json, time
env = {**os.environ, "SEED": "0", "SMOKE": "0", "SCALE": "0.1",
       "METRICS_OUT": "metrics_s01.json", "REPLICATOR_CONFIG": "{}"}
t = time.time()
p = subprocess.run(["bash", "reproduce.sh"], env=env, capture_output=True, text=True)
print(p.returncode, "%.1fs" % (time.time() - t))
print(p.stdout[-2000:]); print(p.stderr[-2000:])
if os.path.exists("metrics_s01.json"):
    d = json.load(open("metrics_s01.json"))
    i = d["metrics"].pop("_intermediates")
    print(json.dumps(d["metrics"], indent=1))
    print({k: i[k] for k in ["lenet_pm", "lenet_es_iter", "lenet_test_acc_es", "conv4_test_acc_es", "runtime_s"]})
