import pathlib, os
root = pathlib.Path("../../..").resolve()
print(root, sorted(x.name for x in root.iterdir())[:40])
for cand in root.rglob("data_cache"):
    print("DC", cand)
    for x in list(cand.rglob("*"))[:60]: print("  ", x)
    break
