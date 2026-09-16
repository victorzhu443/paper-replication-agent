import os
for r,d,f in os.walk("/Users/vzhu/Developer/research-copy/data_cache"):
    for n in f:
        p=os.path.join(r,n); print(p, os.path.getsize(p))
