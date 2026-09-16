import os, datasets
os.environ.setdefault("HF_HOME", os.path.abspath("../../../../data_cache/hf"))
try:
    d = datasets.load_dataset("nyu-mll/glue", "sst2", cache_dir=os.environ["HF_HOME"])
    print(d)
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:300])
from transformers import AutoTokenizer, AutoModelForSequenceClassification
for m in ["roberta-base", "gpt2"]:
    try:
        tok = AutoTokenizer.from_pretrained(m, cache_dir=os.environ["HF_HOME"])
        print(m, "tok ok")
    except Exception as e:
        print(m, "ERR", type(e).__name__, str(e)[:200])
