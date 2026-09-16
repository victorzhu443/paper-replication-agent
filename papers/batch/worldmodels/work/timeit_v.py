import os, time
os.environ.update(SMOKE="0", SEED="0", METRICS_OUT="metrics_v.json", SCALE="1.0", WM_VERBOSE="1")
import wm_main
t = time.time(); wm_main.main(); print("TOTAL", time.time()-t)
