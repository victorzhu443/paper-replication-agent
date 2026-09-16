import os, time, torch
os.environ.setdefault("HF_HOME", os.path.abspath("../../../../data_cache/hf"))
torch.set_num_threads(15)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("roberta-base")
m = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
import datasets
ds = datasets.load_dataset("nyu-mll/glue","sst2")["train"].select(range(320))
texts = ds["sentence"]
t0=time.time()
for i in range(0,320,16):
    b = tok(texts[i:i+16], padding=True, truncation=True, max_length=512, return_tensors="pt")
    out = m(**b, labels=torch.zeros(len(texts[i:i+16]),dtype=torch.long))
    out.loss.backward(); opt.step(); opt.zero_grad()
print("20 FT steps", time.time()-t0)
m.eval()
t0=time.time()
with torch.no_grad():
    for i in range(0,320,32):
        b = tok(texts[i:i+32], padding=True, truncation=True, max_length=512, return_tensors="pt")
        m(**b)
print("eval 320", time.time()-t0)
