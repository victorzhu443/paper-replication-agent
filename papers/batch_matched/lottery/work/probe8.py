import time, torch, torch.nn.functional as F
import lth
torch.set_num_threads(1)
cfg = dict(input_normalization="scale_0_1", conv_padding="same", val_split_seed_value=1234)
data = lth.load_data("mnist", cfg)
model = lth.Net("lenet", cfg)
g = torch.Generator().manual_seed(0); lth.glorot_init(model, g)
masks = [torch.ones_like(m.weight) for m in model.prunable]
opt = torch.optim.Adam(model.parameters(), lr=1.2e-3)
X, y = data["Xtr"], data["ytr"]
perm = torch.randperm(len(X))

def loop(n, do_mask=True, do_index=True):
    xb = X[perm[:60]]; yb = y[perm[:60]]
    t = time.time()
    for i in range(n):
        if do_index:
            idx = perm[i*60:(i+1)*60]; xb = X[idx]; yb = y[idx]
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        opt.step()
        if do_mask:
            with torch.no_grad():
                for m, mk in zip(model.prunable, masks):
                    m.weight.mul_(mk)
    return (time.time()-t)/n*1000

print("full ms", loop(500))
print("no mask ms", loop(500, do_mask=False))
print("no index ms", loop(500, do_index=False))
print("neither ms", loop(500, False, False))
t=time.time(); lth.evaluate(model, data["Xv"], data["yv"]); print("eval val", time.time()-t)
t=time.time(); lth.evaluate(model, data["Xte"], data["yte"]); print("eval test", time.time()-t)
