"""Open CS datasets via torchvision (and, by name, Hugging Face datasets). The training script
downloads into the shared cache; this adapter only probes availability and reports split sizes
for the split-size checkpoint. No rows ever reach a prompt."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pandas as pd

from ..cache import CACHE_ROOT

KNOWN = {
    "mnist": {"train": 60000, "test": 10000, "url": "https://ossci-datasets.s3.amazonaws.com/mnist/"},
    "fashion_mnist": {"train": 60000, "test": 10000, "url": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"},
    "cifar10": {"train": 50000, "test": 10000, "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"},
    "cifar100": {"train": 50000, "test": 10000, "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"},
}


def probe() -> tuple[bool, str]:
    if importlib.util.find_spec("torchvision") is None:
        return False, "torchvision not installed"
    try:
        r = httpx.head(KNOWN["mnist"]["url"] + "train-images-idx3-ubyte.gz", timeout=15, follow_redirects=True)
        return r.status_code < 400, f"torchvision importable; mirror HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001
        return True, f"torchvision importable; mirror probe failed ({e}); download may still succeed via torchvision mirrors"


def fetch(params: dict) -> tuple[pd.DataFrame, str]:
    """Returns split sizes only (the checkpoint), never examples."""
    name = params["name"].lower()
    k = KNOWN.get(name)
    if k is None:
        raise ValueError(f"unknown dataset {name}")
    df = pd.DataFrame({"split": ["train", "test"], "n_examples": [k["train"], k["test"]]})
    return df, f"torchvision:{name}"


def cache_dir(name: str) -> Path:
    p = CACHE_ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return p
