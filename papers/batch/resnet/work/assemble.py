import os, tarfile, pickle, numpy as np, hashlib
d = "/Users/vzhu/Developer/research-copy/data_cache/cifar10"
tgz = os.path.join(d, "cifar-10-python.tar.gz")
with open(tgz, "wb") as out:
    for i in range(12):
        with open(os.path.join(d, f"part{i}"), "rb") as f:
            out.write(f.read())
print("size", os.path.getsize(tgz))
tf = tarfile.open(tgz, "r:gz")
tf.extractall(d)
tf.close()
base = os.path.join(d, "cifar-10-batches-py")
def load(files):
    xs, ys = [], []
    for fn in files:
        with open(os.path.join(base, fn), "rb") as f:
            b = pickle.load(f, encoding="bytes")
        xs.append(b[b"data"]); ys += list(b[b"labels"])
    X = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.uint8)
    return X, np.array(ys, dtype=np.int64)
Xtr, ytr = load([f"data_batch_{i}" for i in range(1, 6)])
Xte, yte = load(["test_batch"])
print(Xtr.shape, ytr.shape, Xte.shape, yte.shape, np.bincount(ytr), np.bincount(yte))
np.savez(os.path.join(d, "cifar10.npz"), Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
for i in range(12):
    os.remove(os.path.join(d, f"part{i}"))
print("saved")
