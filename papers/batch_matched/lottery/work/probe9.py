import time, torch, numpy as np
import lth
torch.set_num_threads(1)
cfg = dict(input_normalization="scale_0_1", conv_padding="same", val_split_seed_value=1234)
data = lth.load_data("mnist", cfg)
m = lth.Net("lenet", cfg); g = torch.Generator().manual_seed(0); lth.glorot_init(m, g)
th0 = [p.weight.detach().clone() for p in m.prunable]
mk = [torch.ones_like(w) for w in th0]
t = time.time()
r = lth.train_run("lenet", cfg, data, th0, mk, 10000, 0.0012, 60, 100, seed=0)
print("lenet 10k iters: %.1fs" % (time.time()-t), r["es_iter"], r["test_acc_es"], r["test_acc_final"])

# conv4 quarter width
cd = lth.load_data("cifar10", cfg)
cm = lth.Net("conv4", cfg, width=0.25); g = torch.Generator().manual_seed(0); lth.glorot_init(cm, g)
cth0 = [p.weight.detach().clone() for p in cm.prunable]
cmk = [torch.ones_like(w) for w in cth0]
t = time.time()
r = lth.train_run("conv4", cfg, cd, cth0, cmk, 300, 0.0003, 60, 150, seed=0, width=0.25,
                  n_val=1500, n_test=2500)
dt = time.time()-t
print("conv4 w.25 300 iters: %.1fs" % dt, r["es_iter"], r["test_acc_es"])
