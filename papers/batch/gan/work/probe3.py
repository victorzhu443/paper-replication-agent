import os
for p in ["/Users/vzhu/Developer/research-copy/data_cache","/Users/vzhu/Developer/research-copy/runs/data_cache"]:
    for root,d,f in os.walk(p):
        if root.count(os.sep) - p.count(os.sep) > 3: continue
        print(root, d[:10], f[:10])
