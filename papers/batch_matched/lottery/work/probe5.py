import os, numpy as np, glob
base="/Users/vzhu/Developer/research-copy/data_cache"
print(os.listdir(base))
z=np.load(base+"/cifar10/cifar10.npz")
print({k:(v.shape,v.dtype) for k,v in z.items()})
print(glob.glob(base+"/mnist*")+glob.glob(base+"/MNIST*"))
for p in glob.glob(base+"/**/*.npz", recursive=True):
    print(p)
