import os, glob
print(os.getcwd())
for p in ["../data_cache", "../../data_cache", "data_cache"]:
    if os.path.isdir(p):
        print(p, os.listdir(p)[:20])
import importlib
for m in ["torch","torchvision","datasets","transformers"]:
    try:
        mod=importlib.import_module(m); print(m, mod.__version__)
    except Exception as e: print(m, "ERR", e)
