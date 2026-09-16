import os, time, torch
print(os.environ.get("HF_HOME"))
from transformers import AutoTokenizer, AutoModelForCausalLM
t0=time.time()
for name in ["gpt2","gpt2-medium"]:
    try:
        tok=AutoTokenizer.from_pretrained(name)
        m=AutoModelForCausalLM.from_pretrained(name)
        print(name,"ok",sum(p.numel() for p in m.parameters()), m.config.n_layer, time.time()-t0)
    except Exception as e:
        print(name,"FAIL",repr(e)[:300])
