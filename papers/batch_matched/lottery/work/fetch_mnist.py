import os, numpy as np
base = "/Users/vzhu/Developer/research-copy/data_cache/mnist"
os.makedirs(base, exist_ok=True)
out = os.path.join(base, "mnist.npz")
if os.path.exists(out):
    z = np.load(out); print({k: v.shape for k, v in z.items()}); raise SystemExit
try:
    from torchvision import datasets
    tr = datasets.MNIST(base, train=True, download=True)
    te = datasets.MNIST(base, train=False, download=True)
    Xtr = tr.data.numpy(); ytr = tr.targets.numpy()
    Xte = te.data.numpy(); yte = te.targets.numpy()
except Exception as e:
    print("torchvision failed:", e)
    os.environ.setdefault("HF_HOME", "/Users/vzhu/Developer/research-copy/data_cache/hf")
    from datasets import load_dataset
    d = load_dataset("ylecun/mnist")
    Xtr = np.stack([np.array(i) for i in d["train"]["image"]])
    ytr = np.array(d["train"]["label"])
    Xte = np.stack([np.array(i) for i in d["test"]["image"]])
    yte = np.array(d["test"]["label"])
np.savez_compressed(out, Xtr=Xtr.astype("uint8"), ytr=ytr.astype("int64"),
                    Xte=Xte.astype("uint8"), yte=yte.astype("int64"))
print(Xtr.shape, Xte.shape, ytr[:10])
