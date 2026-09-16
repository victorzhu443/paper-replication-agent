"""Fixture reproduce.sh scripts for the CS verifier: each writes metrics.json per the contract.
Modes: clean (honors shuffle, reports chance), leaky (labels leak: shuffled run keeps accuracy),
ignores_flag (never shuffles), rl (return with random-policy null), na (declares not applicable)."""
import json, os, sys
mode = sys.argv[1]
cfg = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}"))
shuf = bool(cfg.get("_shuffle_labels"))
scale = float(os.environ.get("SCALE", "1.0"))
out = {"seed": int(os.environ.get("SEED", "0")), "split": "test", "n_examples": 1000, "scale": scale,
       "shuffled": False, "matched_scale": True, "metrics": {}}
if mode == "clean":
    out["shuffled"] = shuf
    out["metrics"] = {"accuracy": 55.0 if shuf else 92.0, "chance_level": 50.0}
elif mode == "leaky":
    out["shuffled"] = shuf
    out["metrics"] = {"accuracy": 91.0 if shuf else 92.0, "chance_level": 50.0}   # survives the shuffle
elif mode == "ignores_flag":
    out["metrics"] = {"accuracy": 92.0, "chance_level": 50.0}                       # never shuffles
elif mode == "rl":
    out["shuffled"] = shuf
    out["metrics"] = {"mean_return": 31.0 if shuf else 55.0, "mean_return_random_policy": 37.8}
elif mode == "na":
    out["shuffle_not_applicable"] = True
    out["metrics"] = {"num_features_represented": 12.9}
json.dump(out, open(os.environ["METRICS_OUT"], "w"))
