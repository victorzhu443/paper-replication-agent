import os, traceback
root = "/Users/vzhu/Developer/research-copy/data_cache/cifar10"
os.makedirs(root, exist_ok=True)
try:
    from torchvision.datasets import CIFAR10
    d = CIFAR10(root, train=True, download=True)
    print("tv ok", len(d))
except Exception as e:
    traceback.print_exc()
    try:
        os.environ["HF_HOME"]="/Users/vzhu/Developer/research-copy/data_cache/hf"
        from datasets import load_dataset
        ds = load_dataset("cifar10")
        print("hf ok", ds)
    except Exception as e2:
        traceback.print_exc()
