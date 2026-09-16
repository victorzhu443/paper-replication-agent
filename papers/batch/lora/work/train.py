import os, json, math, time, random
import numpy as np, torch

HF = os.environ.get("HF_HOME") or os.path.abspath("../../../../data_cache/hf")
os.environ["HF_HOME"] = HF
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(int(os.environ.get("NTHREADS", "15")))

import datasets as hfds
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

SEED = int(os.environ.get("SEED", "0"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
SCALE = float(os.environ.get("SCALE", "1.0"))
CFG = json.loads(os.environ.get("REPLICATOR_CONFIG") or "{}")
OUT = os.environ.get("METRICS_OUT", "metrics.json")

# ---- ambiguity config keys (spec defaults) ----
C = dict(
    base_model="roberta_base_125m_as_paper",
    train_subset_size="full_train_split",
    train_subset_sampling="stratified_random_seed0",
    num_epochs="60",
    batch_vs_steps="steps",
    lr_schedule="as_paper",
    learning_rate="lora=5e-4,ft=2e-5",
    batch_size="16",
    max_seq_len="512",
    lora_rank="8",
    lora_alpha="8",
    lora_target_modules="q_v_only",
    lora_layer_coverage="all_layers",
    lora_dropout="0.0",
    weight_decay="0.01",
    n_seeds="5_median",
    checkpoint_selection="best_epoch_on_eval",
    error_bar_semantics="std_over_seeds",
    split="validation",
    mnli_metric="overall_m_and_mm_accuracy",
    glue_avg_definition="unweighted_mean_of_eight_task_metrics",
    wikisql_accuracy="logical_form",
    gpt3_lora_4p7m_config="rv_2",
    e2e_eval_script="official_e2e_metrics_script",
    head_treatment="trainable_not_counted",
    param_fraction_denominator="all_base_params_incl_embeddings",
    mnli_transfer_init="mnli_init_for_mrpc_rte_stsb",
    augmentation="none",
    mixed_precision="fp32",
    pooling="last_non_pad_token",
)
C.update({k: v for k, v in CFG.items() if not k.startswith("_")})
SHUFFLE = bool(CFG.get("_shuffle_labels", False))

MODEL = "roberta-base" if C["base_model"] == "roberta_base_125m_as_paper" else "gpt2"
BS = int(C["batch_size"])
MAXLEN = int(C["max_seq_len"])
R = int(C["lora_rank"])
ALPHA = float(C["lora_alpha"])
LDROP = float(C["lora_dropout"])
WD = float(C["weight_decay"])
lr_map = dict(p.split("=") for p in C["learning_rate"].split(",")) if "=" in C["learning_rate"] else {"lora": "5e-4", "ft": "2e-5"}
LR_LORA, LR_FT = float(lr_map.get("lora", 5e-4)), float(lr_map.get("ft", 2e-5))

# ---- compute scaling (tier 2) ----
BASE_STEPS = int(CFG.get("max_steps", 800))
MAX_STEPS = max(1, int(round(BASE_STEPS * SCALE)))
EVAL_EVERY = max(1, MAX_STEPS // 4)
if SMOKE:
    MAX_STEPS, EVAL_EVERY = 16, 8
N_SEEDS = 1

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

ds = hfds.load_dataset("nyu-mll/glue", "sst2")
train_full, val = ds["train"], ds["validation"]

# training subset
n_needed = MAX_STEPS * BS
if C["train_subset_size"] == "2000_examples":
    n_sub = 2000
elif C["train_subset_size"] == "10000_examples":
    n_sub = 10000
else:
    n_sub = min(len(train_full), n_needed)
rng = np.random.RandomState(0)
if C["train_subset_sampling"] == "first_n":
    idx = np.arange(min(n_sub, len(train_full)))
elif C["train_subset_sampling"] == "random_seed0":
    idx = rng.choice(len(train_full), size=min(n_sub, len(train_full)), replace=False)
else:  # stratified
    labels = np.array(train_full["label"])
    idx = []
    for lab in np.unique(labels):
        pool = np.where(labels == lab)[0]
        k = int(round(n_sub * len(pool) / len(labels)))
        idx.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
train = train_full.select([int(i) for i in idx])

if SMOKE:
    val = val.select(range(min(128, len(val))))

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

train_texts, train_labels = list(train["sentence"]), np.array(train["label"])
val_texts, val_labels = list(val["sentence"]), np.array(val["label"])


def batches(texts, labels, bs, shuffle_rng=None):
    order = np.arange(len(texts))
    if shuffle_rng is not None:
        shuffle_rng.shuffle(order)
    for i in range(0, len(order), bs):
        sel = order[i:i + bs]
        yield [texts[j] for j in sel], labels[sel]


def encode(txts):
    return tok(txts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")


@torch.no_grad()
def evaluate(model):
    model.eval()
    correct = 0
    for txts, labs in batches(val_texts, val_labels, 32):
        logits = model(**encode(txts)).logits
        correct += int((logits.argmax(-1).numpy() == labs).sum())
    model.train()
    return 100.0 * correct / len(val_texts)


def build(method):
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    if MODEL == "gpt2":
        m.config.pad_token_id = tok.pad_token_id
    base_params = sum(p.numel() for n, p in m.named_parameters() if not n.startswith(("classifier", "score")))
    if method == "lora":
        tgt = {"q_v_only": ["query", "value"], "q_k_v_o": ["query", "key", "value", "output.dense"],
               "q_only": ["query"]}.get(C["lora_target_modules"], ["query", "value"])
        if MODEL == "gpt2":
            tgt = ["c_attn"]
        layers_to_transform = None
        nl = m.config.num_hidden_layers
        if C["lora_layer_coverage"] == "top_half":
            layers_to_transform = list(range(nl // 2, nl))
        elif C["lora_layer_coverage"] == "last_two_layers":
            layers_to_transform = list(range(nl - 2, nl))
        lc = LoraConfig(task_type="SEQ_CLS", r=R, lora_alpha=ALPHA, lora_dropout=LDROP,
                        target_modules=tgt, layers_to_transform=layers_to_transform,
                        modules_to_save=["classifier"])
        m = get_peft_model(m, lc)
        counted = sum(p.numel() for n, p in m.named_parameters() if "lora_" in n)
    else:
        for p in m.parameters():
            p.requires_grad_(True)
        counted = base_params
    return m, counted, base_params


def train_one(method, seed):
    set_seed(seed)
    model, counted, base_params = build(method)
    lr = LR_LORA if method == "lora" else LR_FT
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    warm = int(0.06 * MAX_STEPS) if C["lr_schedule"] == "as_paper" else 0
    if C["lr_schedule"] == "constant":
        sched = None
    else:
        sched = get_linear_schedule_with_warmup(opt, warm, MAX_STEPS)
    brng = np.random.RandomState(seed)
    srng = np.random.RandomState(seed + 1234)
    step, best, accs = 0, -1.0, []
    model.train()
    t0 = time.time()
    while step < MAX_STEPS:
        for txts, labs in batches(train_texts, train_labels, BS, brng):
            if step >= MAX_STEPS:
                break
            labs = np.array(labs)
            if SHUFFLE:  # shuffle labels within each batch
                labs = labs[srng.permutation(len(labs))]
            enc = encode(txts)
            out = model(**enc, labels=torch.tensor(labs, dtype=torch.long))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                a = evaluate(model)
                accs.append((step, a))
                best = max(best, a)
    final = accs[-1][1]
    acc = best if C["checkpoint_selection"] == "best_epoch_on_eval" else final
    return dict(accuracy=acc, final_accuracy=final, trainable_params=counted,
                base_params=base_params, curve=accs, train_seconds=round(time.time() - t0, 1))


res = {m: train_one(m, SEED) for m in ["lora", "ft"]}
lora, ft = res["lora"], res["ft"]
metrics = {
    "accuracy": lora["accuracy"],
    "accuracy_full_ft": ft["accuracy"],
    "accuracy_gap_lora_minus_ft": lora["accuracy"] - ft["accuracy"],
    "glue_avg": lora["accuracy"],
    "glue_avg_full_ft": ft["accuracy"],
    "trainable_parameters_millions": lora["trainable_params"] / 1e6,
    "trainable_parameters_millions_ft": ft["trainable_params"] / 1e6,
    "trainable_param_fraction_percent": 100.0 * lora["trainable_params"] / ft["base_params"],
    "_intermediates": {
        "n_train_examples": len(train_texts),
        "n_eval_examples": len(val_texts),
        "steps": MAX_STEPS,
        "batch_size": BS,
        "eval_curve_lora": lora["curve"],
        "eval_curve_ft": ft["curve"],
        "train_seconds": {"lora": lora["train_seconds"], "ft": ft["train_seconds"]},
        "config_used": C,
        "glue_avg_note": "only SST-2 was run; glue_avg is the unweighted mean over the 1 task run (= SST-2 accuracy), NOT the paper's 8-task average",
        "scale": {
            "base_model": MODEL,
            "train_examples_seen": MAX_STEPS * BS,
            "paper_train_examples": 67349,
            "paper_epochs": 60,
            "steps_run": MAX_STEPS,
            "paper_equivalent_steps": 60 * 67349 // BS,
            "n_seeds": N_SEEDS,
            "paper_n_seeds": 5,
            "eval_examples": len(val_texts),
            "smoke": SMOKE,
            "scale_factor": SCALE,
        },
    },
}
json.dump({"seed": SEED, "split": C["split"], "n_examples": len(val_texts),
           "shuffled": SHUFFLE, "metrics": metrics}, open(OUT, "w"), indent=1)
print("wrote", OUT, {k: v for k, v in metrics.items() if k != "_intermediates"})
