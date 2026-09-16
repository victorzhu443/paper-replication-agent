import os, glob, pathlib
print(os.getcwd())
for p in ["..", "../..", "../data_cache", "../../data_cache"]:
    try:
        print(p, os.listdir(p)[:30])
    except Exception as e:
        print(p, e)
import torchvision, torch
print(torchvision.__version__, torch.__version__, torch.get_num_threads())
