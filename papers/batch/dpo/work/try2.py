import json, os, subprocess
s = open("run_dpo.py").read().replace('cfg("lr_multiplier", 1000.0)', 'cfg("lr_multiplier", 60.0)')
open("run_dpo.py", "w").write(s)
env = dict(os.environ, REPLICATOR_CONFIG=json.dumps({"_shuffle_labels": True}), METRICS_OUT="m_sh.json", SEED="0")
subprocess.run(["python", "run_dpo.py"], env=env, check=True, capture_output=True)
d = json.load(open("m_sh.json"))
print(d["shuffled"], {k: v for k, v in d["metrics"].items() if "win_rate" in k or k == "agreement_rate"})
