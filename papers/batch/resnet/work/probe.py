import os, glob
root = "/Users/vzhu/Developer/research-copy"
for p in [root, root+"/data_cache"]:
    try:
        print(p, sorted(os.listdir(p))[:40])
    except Exception as e:
        print(p, "err", e)
print(glob.glob(root+"/data_cache/**/*cifar*", recursive=True)[:20])
print(glob.glob(root+"/**/cifar*", recursive=True)[:20])
