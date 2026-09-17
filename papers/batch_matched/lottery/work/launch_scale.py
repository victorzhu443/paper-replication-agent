import os, subprocess, time, sys
scale = sys.argv[1] if len(sys.argv) > 1 else "0.1"
env = {**os.environ, "SEED": "0", "SMOKE": "0", "SCALE": scale,
       "METRICS_OUT": "metrics_s%s.json" % scale, "REPLICATOR_CONFIG": "{}"}
log = open("scale_%s.log" % scale, "w")
log.write("start %f\n" % time.time()); log.flush()
p = subprocess.Popen(["bash", "reproduce.sh"], env=env, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
print("pid", p.pid)
