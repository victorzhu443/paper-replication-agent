import os, time, urllib.request
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
dst = "/Users/vzhu/Developer/research-copy/data_cache/cifar10/cifar-10-python.tar.gz"
pos = os.path.getsize(dst) if os.path.exists(dst) else 0
req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"})
t0 = time.time()
try:
    r = urllib.request.urlopen(req, timeout=60)
    total = pos + int(r.headers.get("Content-Length", 0))
    with open(dst, "ab") as f:
        while time.time() - t0 < 250:
            b = r.read(1 << 20)
            if not b: break
            f.write(b); pos += len(b)
except Exception as e:
    print("err", e)
print("size", pos, "total", total, "elapsed", time.time()-t0)
