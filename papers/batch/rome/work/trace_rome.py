"""Reduced CPU replication of Meng et al. 2022 (ROME): causal tracing + one rank-one edit.

Scaled down (compute tier 2): GPT-2 small/medium instead of GPT-2 XL, ~10 facts instead of
1000, 3 noise draws instead of 10. Peak layers are reported both raw and rescaled to the
paper's 48-layer stack.
"""
from __future__ import annotations
import json, os, time, math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)

FACTS = [
    ("France", "The capital of France is", " Paris"),
    ("Japan", "The capital of Japan is", " Tokyo"),
    ("Italy", "The capital of Italy is", " Rome"),
    ("Germany", "The capital of Germany is", " Berlin"),
    ("Russia", "The capital of Russia is", " Moscow"),
    ("Spain", "The capital of Spain is", " Madrid"),
    ("China", "The capital of China is", " Beijing"),
    ("Egypt", "The capital of Egypt is", " Cairo"),
    ("Greece", "The capital of Greece is", " Athens"),
    ("Cuba", "The capital of Cuba is", " Havana"),
    ("The Eiffel Tower", "The Eiffel Tower is located in the city of", " Paris"),
    ("The Space Needle", "The Space Needle is located in downtown", " Seattle"),
    ("Mount Everest", "Mount Everest is located in the country of", " Nepal"),
    ("Toyota", "Toyota is a car company based in", " Japan"),
    ("Honda", "Honda is a car company based in", " Japan"),
    ("Nokia", "Nokia is a phone company based in", " Finland"),
    ("Shakespeare", "Shakespeare wrote his plays in the language of", " English"),
    ("Beethoven", "Beethoven is famous for playing the", " piano"),
    ("Seattle", "Seattle is a city in the state of", " Washington"),
    ("Chicago", "Chicago is a city in the state of", " Illinois"),
    ("Dallas", "Dallas is a city in the state of", " Texas"),
    ("Miami", "Miami is a city in the state of", " Florida"),
    ("Boston", "Boston is a city in the state of", " Massachusetts"),
    ("Denver", "Denver is a city in the state of", " Colorado"),
    ("Volkswagen", "Volkswagen is a car company based in", " Germany"),
    ("Ferrari", "Ferrari is a car company based in", " Italy"),
]


def cfg_get(cfg, key, default):
    v = cfg.get(key, default)
    return default if v is None else v


class Tracer:
    def __init__(self, model_name, dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).eval()
        self.L = self.model.config.n_layer
        self.blocks = self.model.transformer.h
        self.sigma_t = float(self.model.transformer.wte.weight.std())

    def subject_span(self, prompt, subject):
        enc = self.tok(prompt, return_offsets_mapping=True)
        ids = enc["input_ids"]
        start_c = prompt.index(subject)
        end_c = start_c + len(subject)
        # leading-space handling: a token whose span overlaps [start_c-1, end_c) counts
        toks = [i for i, (a, b) in enumerate(enc["offset_mapping"])
                if b > max(start_c - 1, 0) and a < end_c]
        return ids, toks

    # ---- caching hooks ----
    def clean_run(self, ids):
        cache = {"h": {}, "mlp": {}, "attn": {}}
        handles = []

        def saver(key, l):
            def hook(m, i, o):
                t = o[0] if isinstance(o, tuple) else o
                cache[key][l] = t[0].clone()
            return hook
        for l, blk in enumerate(self.blocks):
            handles.append(blk.register_forward_hook(saver("h", l)))
            handles.append(blk.mlp.register_forward_hook(saver("mlp", l)))
            handles.append(blk.attn.register_forward_hook(saver("attn", l)))
        out = self.model(torch.tensor([ids]))
        for h in handles:
            h.remove()
        probs = torch.softmax(out.logits[0, -1], dim=-1)
        return cache, probs

    def run_patched(self, ids, noise, subj_toks, specs, component, cache, target_id):
        """specs: list of (layer_center, token_idx). Returns probs of target_id per spec row."""
        B = max(len(specs), 1)
        input_ids = torch.tensor([ids]).repeat(B, 1)
        handles = []

        def emb_hook(m, i, o):
            o = o.clone()
            for j, t in enumerate(subj_toks):
                o[:, t, :] += noise[j]
            return o
        handles.append(self.model.transformer.drop.register_forward_hook(emb_hook))

        # rows to patch at each layer
        per_layer = {l: [] for l in range(self.L)}
        for r, (c, t) in enumerate(specs):
            if component == "state":
                per_layer[c].append((r, t, c))
            else:
                for l in range(max(0, c - 4), min(self.L, c + 6)):
                    per_layer[l].append((r, t, l))

        def mk(l, rows, key):
            def hook(m, i, o):
                is_tuple = isinstance(o, tuple)
                hid = (o[0] if is_tuple else o).clone()
                for (r, t, src) in rows:
                    hid[r, t, :] = cache[key][src][t]
                if is_tuple:
                    return (hid,) + tuple(o[1:])
                return hid
            return hook

        for l in range(self.L):
            rows = per_layer[l]
            if not rows:
                continue
            if component == "state":
                handles.append(self.blocks[l].register_forward_hook(mk(l, rows, "h")))
            elif component == "mlp":
                handles.append(self.blocks[l].mlp.register_forward_hook(mk(l, rows, "mlp")))
            else:
                handles.append(self.blocks[l].attn.register_forward_hook(mk(l, rows, "attn")))
        out = self.model(input_ids)
        for h in handles:
            h.remove()
        p = torch.softmax(out.logits[:, -1, :], dim=-1)[:, target_id]
        return p.numpy()

    def corrupted_prob(self, ids, noise, subj_toks, target_id):
        return float(self.run_patched(ids, noise, subj_toks, [], "state", None, target_id)[0])


def build_facts(tr, n, seed, shuffle):
    rng = np.random.default_rng(seed)
    kept = []
    for subj, prompt, target in FACTS:
        tid = tr.tok(target)["input_ids"][0]
        ids, toks = tr.subject_span(prompt, subj)
        if not toks:
            continue
        _, probs = tr.clean_run(ids)
        if int(probs.argmax()) != tid:
            continue
        kept.append(dict(subject=subj, prompt=prompt, target=target, target_id=tid,
                         ids=ids, subj_toks=toks, clean_p=float(probs[tid])))
        if len(kept) >= n:
            break
    if shuffle:
        # leakage control: shuffle target objects across facts within the period (the batch of
        # traced facts), so restored subject states no longer predict the target.
        perm = rng.permutation(len(kept))
        if len(kept) > 1:
            while any(perm[i] == i for i in range(len(kept))):
                perm = rng.permutation(len(kept))
        tgt = [(f["target"], f["target_id"]) for f in kept]
        for i, f in enumerate(kept):
            f["target"], f["target_id"] = tgt[perm[i]]
            _, probs = tr.clean_run(f["ids"])
            f["clean_p"] = float(probs[f["target_id"]])
    return kept


def main():
    t_start = time.time()
    seed = int(os.environ.get("SEED", "0"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    scale = float(os.environ.get("SCALE", "1.0"))
    cfg = json.loads(os.environ.get("REPLICATOR_CONFIG", "{}") or "{}")
    out_path = os.environ.get("METRICS_OUT", "metrics.json")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(os.environ.get("N_THREADS", "15")))

    model_name = cfg_get(cfg, "model.name", "gpt2-medium")
    n_prompts_cfg = str(cfg_get(cfg, "tracing.n_prompts", "10 (CPU budget)"))
    n_prompts = 1000 if n_prompts_cfg.startswith("1000") else (100 if n_prompts_cfg.startswith("100") else 10)
    noise_cfg = str(cfg_get(cfg, "tracing.n_noise_samples_seed", "10_samples_seed0"))
    n_noise = 1 if noise_cfg.startswith("1_sample") else int(cfg_get(cfg, "tracing.n_noise_draws", 5))
    noise_param = str(cfg_get(cfg, "tracing.noise_parameterization", "nu_is_std (sigma = 3*sigma_t)"))
    noise_nu_rule = str(cfg_get(cfg, "tracing.noise_nu", "3*sigma_t measured on the same model's embeddings"))
    shuffle = bool(cfg.get("_shuffle_labels", False))
    split = str(cfg_get(cfg, "eval.split", cfg_get(cfg, "split", "test")))

    if smoke:
        n_prompts, n_noise = 2, 1
    else:
        n_prompts = min(len(FACTS), max(2, int(round(n_prompts * scale))))
        n_noise = max(1, int(round(n_noise * scale)))

    tr = Tracer(model_name)
    L = tr.L
    facts = build_facts(tr, n_prompts, seed, shuffle)
    n_facts = len(facts)

    # noise scale
    if noise_nu_rule.startswith("fixed"):
        nu = 0.1
    else:
        nu = 3.0 * tr.sigma_t
    sigma_noise = nu if noise_param.startswith("nu_is_std") else math.sqrt(nu)

    comps = ["state", "mlp", "attn"]
    # accumulators: per component, AIE grids keyed by (bucket, layer)
    aie_last_subj = {c: np.zeros(L) for c in comps}
    aie_last_tok = {c: np.zeros(L) for c in comps}
    restored_last_subj = {c: np.zeros(L) for c in comps}
    restored_last_tok = {c: np.zeros(L) for c in comps}
    max_restored = {c: [] for c in comps}        # max over (layer, token) at last subject token
    max_restored_any = {c: [] for c in comps}    # max over all (layer, token)
    clean_ps, corr_ps = [], []
    gen = torch.Generator().manual_seed(seed)

    for f in facts:
        ids, subj, tid = f["ids"], f["subj_toks"], f["target_id"]
        T = len(ids)
        last_subj = subj[-1]
        cache, probs = tr.clean_run(ids)
        clean_p = float(probs[tid])
        for _ in range(n_noise):
            noise = torch.randn(len(subj), tr.model.config.n_embd, generator=gen) * sigma_noise
            p_corr = tr.corrupted_prob(ids, noise, subj, tid)
            clean_ps.append(clean_p)
            corr_ps.append(p_corr)
            specs = [(l, t) for l in range(L) for t in range(T)]
            for c in comps:
                p = tr.run_patched(ids, noise, subj, specs, c, cache, tid)
                grid = p.reshape(L, T)
                aie_last_subj[c] += grid[:, last_subj] - p_corr
                aie_last_tok[c] += grid[:, T - 1] - p_corr
                restored_last_subj[c] += grid[:, last_subj]
                restored_last_tok[c] += grid[:, T - 1]
                max_restored[c].append(float(grid[:, last_subj].max()))
                max_restored_any[c].append(float(grid.max()))

    n_runs = max(len(clean_ps), 1)
    for c in comps:
        aie_last_subj[c] /= n_runs
        aie_last_tok[c] /= n_runs
        restored_last_subj[c] /= n_runs
        restored_last_tok[c] /= n_runs

    pk_state = int(np.argmax(aie_last_subj["state"]))
    pk_mlp = int(np.argmax(aie_last_subj["mlp"]))
    pk_attn = int(np.argmax(aie_last_tok["attn"]))
    rescale = 48.0 / L

    m = {
        "average_total_effect": 100.0 * (np.mean(clean_ps) - np.mean(corr_ps)),
        "average_indirect_effect": 100.0 * float(aie_last_subj["state"][pk_state]),
        "average_indirect_effect_mlp": 100.0 * float(aie_last_subj["mlp"][pk_mlp]),
        "average_indirect_effect_attn_last_subject": 100.0 * float(aie_last_subj["attn"].max()),
        "peak_layer": pk_state * rescale,
        "peak_layer_mlp": pk_mlp * rescale,
        "peak_layer_attn": pk_attn * rescale,
        "peak_layer_raw_state": float(pk_state),
        "peak_layer_raw_mlp": float(pk_mlp),
        "peak_layer_raw_attn": float(pk_attn),
        "mean_probability": 100.0 * float(np.mean(clean_ps)),
        "mean_probability_corrupted": 100.0 * float(np.mean(corr_ps)),
        "mean_restored_probability": 100.0 * float(np.mean(max_restored["state"])),
        "mean_restored_probability_mlp": 100.0 * float(np.mean(max_restored_any["mlp"])),
        "mean_restored_probability_attn": 100.0 * float(np.mean(max_restored_any["attn"])),
        "mean_restored_probability_peak_layer_state": 100.0 * float(restored_last_subj["state"][pk_state]),
        "mean_restored_probability_peak_layer_mlp": 100.0 * float(restored_last_subj["mlp"][pk_mlp]),
        "mean_restored_probability_peak_layer_attn": 100.0 * float(restored_last_tok["attn"][pk_attn]),
    }

    # ---------- one ROME rank-one edit ----------
    rome = rome_edit(tr, cfg, seed, smoke)
    m["efficacy_score"] = 100.0 if rome["p_after"] > rome["p_before"] else 0.0
    m["efficacy_magnitude"] = 100.0 * (rome["p_after"] - rome["p_before"])

    m["_intermediates"] = {
        "n_facts": n_facts,
        "n_noise_draws": n_noise,
        "n_runs": n_runs,
        "model": model_name,
        "n_layers": L,
        "sigma_t": tr.sigma_t,
        "sigma_noise": sigma_noise,
        "aie_last_subject_by_layer": {c: (100 * aie_last_subj[c]).round(3).tolist() for c in comps},
        "aie_last_token_by_layer": {c: (100 * aie_last_tok[c]).round(3).tolist() for c in comps},
        "rome": rome,
        "runtime_s": time.time() - t_start,
        "scale": {
            "model": f"{model_name} ({L} layers) instead of GPT-2 XL (48 layers)",
            "n_prompts": f"{n_facts} instead of 1000",
            "n_noise_draws": f"{n_noise} instead of 10",
            "rome_records": "1 fact edited instead of 7500 COUNTERFACT records",
            "counterfact_zsre": "datasets unavailable (data tier C): hand-built fact list",
            "peak_layer_rescale": f"raw layer index x {rescale:.2f} to map onto 48 layers",
            "SCALE_env": scale, "SMOKE": smoke,
        },
    }
    res = {"seed": seed, "split": split, "n_examples": int(n_facts),
           "shuffled": bool(shuffle), "metrics": m}
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in m.items() if k != "_intermediates"}, indent=1, default=float))
    print("runtime_s", time.time() - t_start)


def rome_edit(tr, cfg, seed, smoke):
    """Minimal ROME rank-one update of mlp.c_proj at one layer, on one counterfactual fact."""
    L = tr.L
    edit_layer_rule = str(cfg_get(cfg, "rome.edit_layer", "scaled_mid_layer"))
    if edit_layer_rule.startswith("layer_18"):
        layer = min(18, L - 1)
    else:
        layer = int(round(0.37 * L))
    subject, prompt, target = "The Eiffel Tower", "The Eiffel Tower is located in the city of", " Rome"
    tid = tr.tok(target)["input_ids"][0]
    ids, subj_toks = tr.subject_span(prompt, subject)
    i = subj_toks[-1]
    prefix_rule = str(cfg_get(cfg, "rome.k_star_prefixes", "20 texts: ten of length 5 and ten of length 10"))
    rng = np.random.default_rng(seed)
    if prefix_rule.startswith("no prefix"):
        prefixes = [[]]
    else:
        n_each, lens = (2, [5, 10]) if smoke else (5, [5, 10])
        if prefix_rule.startswith("50"):
            prefixes = [list(rng.integers(0, 50000, size=int(rng.integers(2, 11)))) for _ in range(10)]
        else:
            prefixes = [list(rng.integers(0, 50000, size=ln)) for ln in lens for _ in range(n_each)]
        prefixes = [[]] + prefixes

    blk = tr.blocks[layer]
    cproj = blk.mlp.c_proj  # Conv1D: weight (d_mlp, d_model)

    # k*: input to c_proj at token i, averaged over prefixes
    ks = []
    for pre in prefixes:
        full = list(pre) + ids
        off = len(pre)
        box = {}
        h = cproj.register_forward_hook(lambda m, inp, o: box.__setitem__("k", inp[0][0].clone()))
        tr.model(torch.tensor([full]))
        h.remove()
        ks.append(box["k"][off + i])
    k_star = torch.stack(ks).mean(0)

    W = cproj.weight.data  # (d_mlp, d_model)
    b = cproj.bias.data
    m_star_cur = k_star @ W + b

    # v*: optimize the mlp output vector at token i to maximize log P(o*) over prefixed prompts
    z = m_star_cur.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.5, weight_decay=1.5e-3)
    steps = 5 if smoke else 20
    with torch.enable_grad():
        for _ in range(steps):
            losses = []
            for pre in prefixes[: (3 if smoke else 6)]:
                full = list(pre) + ids
                off = len(pre)

                def hook(mod, inp, o, off=off):
                    o = o.clone()
                    o[0, off + i, :] = z
                    return o
                hh = blk.mlp.register_forward_hook(hook)
                out = tr.model(torch.tensor([full]))
                hh.remove()
                logp = torch.log_softmax(out.logits[0, -1], dim=-1)
                losses.append(-logp[tid])
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if float(loss.detach()) < 5e-2:
                break
    v_star = z.detach()

    p_before = float(torch.softmax(tr.model(torch.tensor([ids])).logits[0, -1], -1)[tid])
    # rank-one update with C = I (covariance from Wikipedia unavailable on CPU)
    resid = v_star - m_star_cur
    W_new = W + torch.outer(k_star, resid) / float(k_star @ k_star)
    W_old = W.clone()
    cproj.weight.data.copy_(W_new)
    p_after = float(torch.softmax(tr.model(torch.tensor([ids])).logits[0, -1], -1)[tid])
    top_after = tr.tok.decode(int(tr.model(torch.tensor([ids])).logits[0, -1].argmax()))
    cproj.weight.data.copy_(W_old)
    return {"layer": layer, "prompt": prompt, "target": target,
            "p_before": p_before, "p_after": p_after, "top1_after": top_after,
            "n_prefixes": len(prefixes), "cov_source": "identity_C",
            "final_loss": float(loss.detach()) if steps else None}


if __name__ == "__main__":
    main()
