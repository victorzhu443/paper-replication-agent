import os, time, torch
print("cpu_count", os.cpu_count(), "torch threads", torch.get_num_threads())
try:
    print("affinity", len(os.sched_getaffinity(0)))
except Exception as e:
    print(e)
import lth
torch.set_num_threads(1)
cfg = dict(input_normalization="scale_0_1", conv_padding="same", val_split_seed_value=1234)
t = time.time(); data = lth.load_data("mnist", cfg); print("load", time.time()-t)
proto = lth.Net("lenet", cfg)
g = torch.Generator().manual_seed(0); lth.glorot_init(proto, g)
th0 = [m.weight.detach().clone() for m in proto.prunable]
mk = [torch.ones_like(w) for w in th0]
t = time.time()
r = lth.train_run("lenet", cfg, data, th0, mk, 2000, 0.0012, 60, 100, seed=0)
print("2000 iters + 20 evals", time.time()-t, r["es_iter"], r["test_acc_es"])
