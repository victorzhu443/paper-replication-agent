import torch, time
torch.set_num_threads(15)
K,n,m,B=20,400,30,1024
W=torch.randn(K,m,n,requires_grad=True); b=torch.zeros(K,1,n,requires_grad=True)
opt=torch.optim.Adam([W,b],lr=1e-3)
t=time.time()
for i in range(100):
    x=(torch.rand(K,B,n)<0.2).float()*torch.rand(K,B,n)
    h=torch.bmm(x,W.transpose(1,2))
    out=torch.relu(torch.bmm(h,W)+b)
    loss=((out-x)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print("per step", (time.time()-t)/100)
