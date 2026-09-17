import os, glob
for root in ["/", os.path.expanduser("~")]:
    pass
hits = []
for pat in ["**/MNIST/**/*.gz","**/MNIST/**/*ubyte*","**/cifar-10-batches-py*","**/*.tar.gz"]:
    for base in [os.path.abspath(".."), os.path.abspath("../..") ,os.path.abspath("../../..")]:
        try:
            hits += glob.glob(os.path.join(base, pat), recursive=True)[:5]
        except Exception as e:
            print(e)
print("\n".join(sorted(set(hits))[:40]))
print(os.environ.get("HF_HOME"))
print(os.listdir(os.path.abspath("..")))
