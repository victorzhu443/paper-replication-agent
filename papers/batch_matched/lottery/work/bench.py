import time, torch, torch.nn as nn
torch.set_num_threads(1)

def lenet():
    return nn.Sequential(nn.Flatten(), nn.Linear(784,300), nn.ReLU(), nn.Linear(300,100), nn.ReLU(), nn.Linear(100,10))

def conv4(w=1.0):
    c=lambda a,b: nn.Conv2d(a,b,3,padding=1)
    k=lambda n:int(n*w)
    return nn.Sequential(c(3,k(64)),nn.ReLU(),c(k(64),k(64)),nn.ReLU(),nn.MaxPool2d(2),
                         c(k(64),k(128)),nn.ReLU(),c(k(128),k(128)),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(k(128)*64,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,10))

def bench(model, shape, bs=60, n=20):
    opt=torch.optim.Adam(model.parameters(),1e-3)
    x=torch.randn(bs,*shape); y=torch.randint(0,10,(bs,))
    lf=nn.CrossEntropyLoss()
    for _ in range(3):
        opt.zero_grad(); lf(model(x),y).backward(); opt.step()
    t=time.time()
    for _ in range(n):
        opt.zero_grad(); lf(model(x),y).backward(); opt.step()
    return (time.time()-t)/n

print("lenet step", bench(lenet(),(1,28,28)))
print("conv4 step", bench(conv4(1.0),(3,32,32)))
print("conv4/2 step", bench(conv4(0.5),(3,32,32)))
print("conv4/4 step", bench(conv4(0.25),(3,32,32)))
m=lenet(); x=torch.randn(1000,1,28,28)
t=time.time()
with torch.no_grad(): m(x)
print("lenet eval 1000", time.time()-t)
m=conv4(0.5); x=torch.randn(500,3,32,32)
t=time.time()
with torch.no_grad(): m(x)
print("conv4/2 eval 500", time.time()-t)
