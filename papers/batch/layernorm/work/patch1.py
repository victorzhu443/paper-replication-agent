import re
s = open("train_eval.py").read()
old = '    cfg_user = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}")'
new = '''    raw_cfg = os.environ.get("REPLICATOR_CONFIG", "") or "{}"
    try:
        cfg_user = json.loads(raw_cfg)
    except Exception:
        cfg_user = {}
    if not isinstance(cfg_user, dict):
        cfg_user = {}'''
assert old in s
open("train_eval.py", "w").write(s.replace(old, new))
print("patched")
