"""Hand replication #2 (CS, train_and_eval, compute tier 1): a 2-layer MLP with 300 hidden
units on MNIST, LeCun et al. (1998) row "2-layer NN, 300 HU, mean square error": 4.7% test error.
Config keys (ambiguities): loss {mse, cross_entropy}, optimizer {sgd, adam}, epochs, hidden.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

cfg = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}"))
seed = int(os.environ.get("SEED", "0"))
smoke = os.environ.get("SMOKE", "0") == "1"
out = os.environ["METRICS_OUT"]
torch.manual_seed(seed); np.random.seed(seed)

hidden = int(cfg.get("hidden", 300))
loss_name = cfg.get("loss", "mse")
opt_name = cfg.get("optimizer", "sgd")
epochs = 1 if smoke else int(cfg.get("epochs", 20))
n_train = 2048 if smoke else None
lr = float(cfg.get("lr", 0.1 if opt_name == "sgd" else 1e-3))
root = os.environ.get("MNIST_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "data_cache", "mnist"))

tf = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST(root, train=True, download=True, transform=tf)
test = datasets.MNIST(root, train=False, download=True, transform=tf)
if n_train:
    train = torch.utils.data.Subset(train, range(n_train))
tl = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
te = torch.utils.data.DataLoader(test, batch_size=1000)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.Sigmoid() if loss_name == "mse" else nn.ReLU(), nn.Linear(hidden, 10))
opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) if opt_name == "sgd" else torch.optim.Adam(model.parameters(), lr=lr)

shuffle_labels = bool(cfg.get("_shuffle_labels"))
t0 = time.time()
for ep in range(epochs):
    model.train()
    for x, y in tl:
        if shuffle_labels:
            y = y[torch.randperm(len(y))]
        logits = model(x)
        if loss_name == "mse":
            loss = F.mse_loss(torch.sigmoid(logits), F.one_hot(y, 10).float())
        else:
            loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
model.eval(); correct = 0; n = 0
with torch.no_grad():
    for x, y in te:
        correct += (model(x).argmax(1) == y).sum().item(); n += len(y)
acc = correct / n
json.dump({"seed": seed, "split": "test", "n_examples": n,
           "metrics": {"accuracy": acc * 100, "test_error": (1 - acc) * 100, "epochs": epochs, "train_seconds": time.time() - t0}},
          open(out, "w"))
print(f"seed={seed} epochs={epochs} loss={loss_name} opt={opt_name} test_error={(1-acc)*100:.2f}%", file=sys.stderr)
