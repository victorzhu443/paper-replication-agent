import numpy as np, torch, time
from toy import train, stats, importance_vec, pick_best
for steps, batch, lr in [(10000,4096,1e-3),(50000,1024,1e-3),(20000,1024,1e-2)]:
    cfg={"learning_rate":lr}
    g=torch.Generator().manual_seed(0)
    imp=importance_vec(20,0.7,cfg)
    t=time.time()
    W,b,l=train(20,5,"relu",imp,np.full(5,0.99,dtype=np.float32),steps,batch,cfg,g,restarts=5)
    Wb,bb,lb=pick_best(W,b,l); s=stats(Wb,cfg)
    print(steps,batch,lr,round(s["frob2"],2),np.round(s["norms"],2), round(time.time()-t,1))
