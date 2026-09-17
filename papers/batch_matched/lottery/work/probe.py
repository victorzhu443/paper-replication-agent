import os, sys, glob
print(sys.version)
for p in ["data_cache", "../data_cache", "../../data_cache"]:
    if os.path.isdir(p):
        print("DIR", os.path.abspath(p))
        for root, d, f in os.walk(p):
            if root.count(os.sep) - p.count(os.sep) < 3:
                print(root, d[:10], f[:10])
try:
    import torch, torchvision
    print("torch", torch.__version__, torchvision.__version__, torch.get_num_threads())
except Exception as e:
    print("err", e)
