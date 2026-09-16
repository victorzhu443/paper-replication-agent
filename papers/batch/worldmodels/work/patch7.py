s = open("wm_main.py").read()
s = s.replace("def __init__(self, z_dim=32, ch=(16, 32, 64, 128)):", "def __init__(self, z_dim=32, ch=(8, 16, 32, 64)):")
s = s.replace('vae_steps = int(cfg.get("_vae_steps", 500))', 'vae_steps = int(cfg.get("_vae_steps", 1500))')
s = s.replace('"vae_channels": "16/32/64/128 (paper 32/64/128/256), z=32 as paper"',
              '"vae_channels": "8/16/32/64 (paper 32/64/128/256), z=32 as paper"')
open("wm_main.py", "w").write(s)
print("ok")
