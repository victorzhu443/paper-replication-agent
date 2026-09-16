import os, sys, glob, json
print("cwd", os.getcwd())
for p in [".", "..", "../..", "../../data_cache", "../data_cache", "data_cache"]:
    try:
        print("==", p, sorted(os.listdir(p))[:40])
    except Exception as e:
        print("==", p, "ERR", e)
try:
    import torch, torchvision
    print("torch", torch.__version__, "tv", torchvision.__version__, "threads", torch.get_num_threads())
except Exception as e:
    print("torch err", e)
for root in ["/", os.path.expanduser("~")]:
    pass
hits = glob.glob("/**/MNIST/raw/*ubyte*", recursive=False)
print("hits", hits[:5])
import subprocess
print(subprocess.run(["bash","-lc","ls -d /*/ ; echo ---; find / -maxdepth 6 -name 'train-images-idx3-ubyte*' 2>/dev/null | head; echo ---; find / -maxdepth 5 -iname '*mnist*' 2>/dev/null | head -20"],capture_output=True,text=True).stdout)
print("env HF_HOME", os.environ.get("HF_HOME"))
