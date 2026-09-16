import time, torch, torch.nn as nn
torch.set_num_threads(8)
X = torch.rand(55000, 784)
Y = torch.randint(0, 10, (55000,))
m = nn.Sequential(nn.Linear(784,300), nn.ReLU(), nn.Linear(300,100), nn.ReLU(), nn.Linear(100,10))
opt = torch.optim.Adam(m.parameters(), lr=1.2e-3)
lf = nn.CrossEntropyLoss()
for bs in (60, 240):
    t=time.time(); n=1000
    for i in range(n):
        idx = torch.randint(0,55000,(bs,))
        opt.zero_grad(); loss = lf(m(X[idx]), Y[idx]); loss.backward(); opt.step()
    print(bs, (time.time()-t)/n*1000, "ms/step")
t=time.time()
with torch.no_grad():
    for i in range(0, 55000, 5000):
        m(X[i:i+5000])
print("eval 55k", time.time()-t)
