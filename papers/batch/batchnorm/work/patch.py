import re
s = open("train_mnist_bn.py").read()
s = s.replace('''    metrics["accuracy"] = curve_bn[final]''',
'''    metrics["accuracy"] = curve_bn[final]
    # steps for BN net to first reach the non-BN net's final accuracy (speed-of-training proxy);
    # ImageNet 'training_steps' claims are UNTESTED (ILSVRC2012 unavailable, CPU-only).
    target = curve_nb[final]
    reached = [s for s in eval_steps if curve_bn[s] >= target]
    metrics["training_steps"] = float(reached[0] if reached else final)''')
open("train_mnist_bn.py","w").write(s)
print("ok")
