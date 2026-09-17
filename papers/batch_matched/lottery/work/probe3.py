import os
d="/Users/vzhu/Developer/research-copy/runs/data_cache"
for root, ds, fs in os.walk(d):
    if root.count(os.sep)-d.count(os.sep) < 2:
        print(root, ds[:10], fs[:10])
