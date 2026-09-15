"""Hugging Face datasets adapter: probe by repo id via the Hub API; split sizes from the Hub's
dataset-info when available. Examples never reach a prompt. Downloads happen in reproduce.sh."""
from __future__ import annotations

import httpx
import pandas as pd

HUB = "https://huggingface.co/api/datasets/"

# canonical spec names -> Hub repo ids (grown by hand; the KB, not the model, owns this)
CANONICAL = {
    "wmt14_en_de": "wmt/wmt14", "wmt14_en_fr": "wmt/wmt14", "newstest2014": "wmt/wmt14", "newstest2013": "wmt/wmt14",
    "wmt16_en_de": "wmt/wmt16", "glue": "nyu-mll/glue", "sst2": "stanfordnlp/sst2", "squad": "rajpurkar/squad",
    "imagenet": "ILSVRC/imagenet-1k", "cifar10_hf": "uoft-cs/cifar10", "mnist_hf": "ylecun/mnist",
    "wikitext103": "Salesforce/wikitext", "penn_treebank": "ptb-text-only/ptb_text_only", "penn_treebank_wsj": "ptb-text-only/ptb_text_only",
    "librispeech": "openslr/librispeech_asr", "coco": "detection-datasets/coco",
}


def resolve(canonical: str) -> str | None:
    if canonical in CANONICAL:
        return CANONICAL[canonical]
    if canonical.startswith("hf:"):
        return canonical[3:]
    return None


def probe(repo: str = "wmt/wmt14") -> tuple[bool, str]:
    try:
        r = httpx.get(HUB + repo, timeout=20, follow_redirects=True)
        if r.status_code == 200:
            j = r.json()
            gated = j.get("gated", False)
            return (not gated), f"HTTP 200 {repo}" + (" (gated: needs HF token)" if gated else "")
        return False, f"HTTP {r.status_code} {repo}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def fetch(params: dict) -> tuple[pd.DataFrame, str]:
    repo = params["repo"]
    r = httpx.get(HUB + repo, params={"expand[]": "cardData"}, timeout=30, follow_redirects=True)
    r.raise_for_status()
    j = r.json()
    rows = []
    for cfg in (j.get("cardData") or {}).get("dataset_info", []) or []:
        for sp in cfg.get("splits", []) or []:
            rows.append({"config": cfg.get("config_name"), "split": sp.get("name"), "n_examples": sp.get("num_examples")})
    return pd.DataFrame(rows or [{"config": None, "split": None, "n_examples": None}]), f"hf:{repo}:{j.get('sha', '')[:12]}"
