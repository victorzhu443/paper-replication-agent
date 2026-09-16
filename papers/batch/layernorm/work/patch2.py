import re
s = open("lncore.py").read()

old_scale = s[s.index('    # compute-tier-2 scaling knobs'):s.index('def resolve_config')]
new_scale = '''    # compute-tier-2 scaling knobs (NOT paper choices; recorded in metrics._intermediates.scale)
    "scale_mlp_train_subset": 55000,
    "scale_mlp_epochs_bz128": 20,
    "scale_mlp_epochs_bz4": 5,
    "scale_mlp_train_subset_bz4": 8000,
    "scale_val_subset": 5000,
    "scale_eval_every_steps": 100,
    "run_bz4": True,
    "run_gru_aux": True,
    "run_draw_lite": True,
    "scale_gru_hidden": 128,
    "scale_gru_train_subset": 8000,
    "scale_gru_epochs": 4,
    "scale_gru_batch": 64,
    "scale_draw_glimpses": 16,
    "scale_draw_hidden": 128,
    "scale_draw_latent": 32,
    "scale_draw_train_subset": 12000,
    "scale_draw_epochs": 8,
    "scale_draw_batch": 128,
    "conv_smooth_window": 3,
    "train_nll_threshold": 0.1,
}


'''
s = s.replace(old_scale, new_scale)

# smoothing helper + threshold-on-train-loss helper
s = s.replace('''def steps_to_own_best(curve''', '''def smooth_curve(curve, window: int):
    """Moving average over the value channel (robust 'best' detection on noisy val curves)."""
    if window <= 1:
        return list(curve)
    out = []
    vals = [c[1] for c in curve]
    for i in range(len(curve)):
        lo = max(0, i - window + 1)
        out.append((curve[i][0], float(np.mean(vals[lo:i + 1])), curve[i][2]))
    return out


def steps_to_own_best(curve''')
open("lncore.py", "w").write(s)
print("lncore patched", "conv_smooth_window" in s, "smooth_curve" in s)
