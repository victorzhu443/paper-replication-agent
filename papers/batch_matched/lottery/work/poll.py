import time, os, json, sys
time.sleep(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
for f in os.listdir("."):
    if f.startswith("scale_") and f.endswith(".log"):
        print("==", f)
        print(open(f).read()[-1500:])
