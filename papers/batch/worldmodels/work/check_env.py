import time, numpy as np
try:
    import gymnasium as gym
    e = gym.make("CartPole-v1", render_mode="rgb_array")
    o,_ = e.reset(seed=0)
    t=time.time()
    n=0
    for i in range(200):
        f = e.render(); n+=1
        o,r,term,trunc,_ = e.step(e.action_space.sample())
        if term or trunc: e.reset()
    print("frame", np.array(f).shape, "fps", n/(time.time()-t))
except Exception as ex:
    print("ERR", type(ex).__name__, ex)
import torch
print("torch", torch.__version__, torch.get_num_threads())
try:
    import cma; print("cma ok")
except Exception as ex:
    print("no cma", ex)
