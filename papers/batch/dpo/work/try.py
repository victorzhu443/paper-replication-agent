import json, os, subprocess
for lm in [20, 60, 150]:
    env = dict(os.environ, REPLICATOR_CONFIG=json.dumps({"lr_multiplier": lm}),
               METRICS_OUT=f"m_{lm}.json", SEED="0")
    subprocess.run(["python", "run_dpo.py"], env=env, check=True, capture_output=True)
    m = json.load(open(f"m_{lm}.json"))["metrics"]
    print(lm, {k: round(v, 2) for k, v in m.items() if isinstance(v, float) and ("win_rate" in k or "kl" in k)})
