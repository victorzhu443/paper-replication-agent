import os, subprocess
env = dict(os.environ, SMOKE="0", SEED="0", SCALE="1.0", METRICS_OUT="metrics_full.json")
f = open("full.log", "w")
p = subprocess.Popen(["bash", "reproduce.sh"], env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
print("pid", p.pid)
