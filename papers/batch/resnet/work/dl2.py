import os, time, urllib.request, threading
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
d = "/Users/vzhu/Developer/research-copy/data_cache/cifar10"
TOTAL = 170498071
N = 12
part = (TOTAL + N - 1) // N
t0 = time.time()

def work(i):
    start = i * part
    end = min(TOTAL, start + part) - 1
    p = os.path.join(d, f"part{i}")
    have = os.path.getsize(p) if os.path.exists(p) else 0
    if start + have > end: return
    req = urllib.request.Request(url, headers={"Range": f"bytes={start+have}-{end}"})
    try:
        r = urllib.request.urlopen(req, timeout=60)
        with open(p, "ab") as f:
            while time.time() - t0 < 250:
                b = r.read(1 << 18)
                if not b: break
                f.write(b)
    except Exception as e:
        print(i, "err", e)

ts = [threading.Thread(target=work, args=(i,)) for i in range(N)]
[t.start() for t in ts]; [t.join() for t in ts]
tot = 0
for i in range(N):
    p = os.path.join(d, f"part{i}")
    s = os.path.getsize(p) if os.path.exists(p) else 0
    tot += s
    print(i, s, "/", min(TOTAL, (i+1)*part) - i*part)
print("total", tot, "of", TOTAL, time.time()-t0)
