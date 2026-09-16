import os, time, sys
os.environ["SMOKE"] = os.environ.get("SMOKE", "1")
os.environ["SEED"] = "0"
os.environ["METRICS_OUT"] = "metrics_test.json"
os.environ["SCALE"] = os.environ.get("SCALE", "1.0")
import wm_main
t = time.time(); wm_main.main(); print("TOTAL", time.time() - t)
