import os, glob
print({k:v for k,v in os.environ.items() if 'HF' in k or 'CACHE' in k or 'DATA' in k})
print(glob.glob(os.path.expanduser("~/.cache/*")))
import torchvision
try:
    d=torchvision.datasets.MNIST(root="data_cache", train=True, download=True)
    print("ok", len(d))
except Exception as e:
    print("ERR", type(e).__name__, e)
try:
    from datasets import load_dataset
    ds=load_dataset("mnist")
    print(ds)
except Exception as e:
    print("HF ERR", type(e).__name__, str(e)[:300])
