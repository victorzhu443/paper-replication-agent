import os, time
os.environ["SMOKE"] = "0"
os.environ["SEED"] = "0"
os.environ["METRICS_OUT"] = "metrics_test.json"
os.environ["SCALE"] = "0.1"
import wm_main
t = time.time(); wm_main.main(); print("TOTAL", time.time() - t)
