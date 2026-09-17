import os, glob
base="/Users/vzhu/Developer/research-copy"
print(glob.glob(base+"/**/cifar*", recursive=True)[:20])
print(os.listdir(base))
print(os.listdir(base+"/runs"))
