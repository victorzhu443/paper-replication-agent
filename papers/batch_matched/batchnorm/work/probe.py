import importlib, os
for m in ["torch","torchvision","datasets"]:
    try:
        mod=importlib.import_module(m); print(m, mod.__version__)
    except Exception as e: print(m,"ERR",e)
print(os.listdir(".."))
for p in ["../data_cache"]:
    if os.path.exists(p): print(p, os.listdir(p))
