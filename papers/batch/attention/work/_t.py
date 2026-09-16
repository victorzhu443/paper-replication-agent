import os, subprocess, sys, json, time
env = dict(os.environ, SEED="0", SMOKE=sys.argv[1] if len(sys.argv) > 1 else "1",
           REPLICATOR_CONFIG=sys.argv[2] if len(sys.argv) > 2 else "{}",
           METRICS_OUT="metrics_test.json")
t = time.time()
p = subprocess.run(["bash", "reproduce.sh"], env=env, capture_output=True, text=True)
print(p.stdout[-3000:]); print("ERR", p.stderr[-3000:]); print("rc", p.returncode, "s", round(time.time()-t, 1))
