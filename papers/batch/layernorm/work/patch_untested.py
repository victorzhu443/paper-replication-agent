import re
p = "train_eval.py"
s = open(p).read()
anchor = '    metrics["_intermediates"] = inter\n'
add = '''    # ---- claims whose data source is unavailable (MSCOCO order-embeddings, Table 2 rows
    # 'Sym [Vendrov et al., 2016]' / 'OE ...'): reported as NaN = Untested, per the plan's
    # substitution table ("mscoco: UNAVAILABLE"). Keys are emitted so the contract is explicit.
    for k in ("recall_at_1", "recall_at_5", "recall_at_10", "mean_rank"):
        metrics[k] = float("nan")
    inter["untested_metrics"] = {
        "keys": ["recall_at_1", "recall_at_5", "recall_at_10", "mean_rank"],
        "reason": ("MSCOCO images/captions and the pre-trained VGG 10-crop features are not available "
                   "offline; the Table 2 retrieval numbers (including the copied Vendrov et al. rows) "
                   "cannot be re-implemented. Reported as NaN (Untested) rather than substituted."),
    }
'''
assert anchor in s
s = s.replace(anchor, add + anchor)
open(p, "w").write(s)
print("patched")
