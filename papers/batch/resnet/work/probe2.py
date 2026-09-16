import importlib, os
for m in ["torch","torchvision","datasets","transformers","numpy","PIL"]:
    try:
        mod=importlib.import_module(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e: print(m,"ERR",e)
print(os.environ.get("HF_HOME"))
import torch; print("threads", torch.get_num_threads())
