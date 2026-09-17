import os, glob
base="/Users/vzhu/Developer/research-copy/data_cache/hf"
for root, ds, fs in os.walk(base):
    if root.count(os.sep)-base.count(os.sep) < 3:
        print(root, ds[:10], fs[:10])
