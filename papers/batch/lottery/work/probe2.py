import os
from torchvision import datasets
root = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "..", "data_cache"))
print(root, os.path.exists(root))
os.makedirs("data_cache", exist_ok=True)
d = datasets.MNIST("data_cache", train=True, download=True)
print(len(d))
