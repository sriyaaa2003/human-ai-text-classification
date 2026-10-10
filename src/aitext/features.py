"""Interpretable stylometric features (no model needed) and GPT-2 token statistics."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TRANSITIONS = (
    "however moreover furthermore additionally overall therefore thus consequently nevertheless "
    "importantly ultimately generally typically essentially specifically notably"
).split()
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our", "us"}
STOP = set(
    "the a an and or but of to in on for with as by at from that this it is are was were be been not no can will would should could".split()
)
WORD = re.compile(r"[A-Za-z']+")
SENT = re.compile(r"(?<=[.!?])\s+")

STYLE_COLS = [
    "sent_len_mean", "sent_len_cv", "word_len_mean", "type_token", "hapax", "bigram_distinct",
    "comma", "colon_semicolon", "paren", "quote", "question", "exclaim", "digit", "capital_start",
    "contraction", "first_person", "transition", "stopword", "passive", "long_word",
]


def stylometry(text: str) -> dict:
    words = WORD.findall(text)
    low = [w.lower() for w in words]
    n = max(len(words), 1)
    sents = [s for s in SENT.split(text) if s.strip()]
    lens = np.array([len(WORD.findall(s)) for s in sents] or [0], dtype=float)
    counts = pd.Series(low).value_counts() if low else pd.Series(dtype=int)
    bigrams = list(zip(low, low[1:]))
    chars = max(len(text), 1)
    return {
        "sent_len_mean": lens.mean(),
        "sent_len_cv": lens.std() / lens.mean() if lens.mean() > 0 else 0.0,
        "word_len_mean": float(np.mean([len(w) for w in words])) if words else 0.0,
        "type_token": len(counts) / n,
        "hapax": float((counts == 1).sum()) / n if len(counts) else 0.0,
        "bigram_distinct": len(set(bigrams)) / max(len(bigrams), 1),
        "comma": text.count(",") / n,
        "colon_semicolon": (text.count(":") + text.count(";")) / n,
        "paren": (text.count("(") + text.count(")")) / n,
        "quote": (text.count('"') + text.count("“") + text.count("”")) / n,
        "question": text.count("?") / max(len(sents), 1),
        "exclaim": text.count("!") / max(len(sents), 1),
        "digit": sum(c.isdigit() for c in text) / chars,
        "capital_start": sum(1 for w in words if w[0].isupper()) / n,
        "contraction": len(re.findall(r"\w'(?:s|t|re|ve|ll|d|m)\b", text)) / n,
        "first_person": sum(w in FIRST_PERSON for w in low) / n,
        "transition": sum(w in TRANSITIONS for w in low) / n,
        "stopword": sum(w in STOP for w in low) / n,
        "passive": len(re.findall(r"\b(?:was|were|been|is|are|be)\s+\w+ed\b", text)) / n,
        "long_word": sum(len(w) >= 9 for w in words) / n,
    }


def stylometry_frame(texts) -> pd.DataFrame:
    return pd.DataFrame([stylometry(t) for t in texts], columns=STYLE_COLS)


LM_COLS = ["lp_mean", "lp_std", "entropy_mean", "top1_frac", "top10_frac", "logrank_mean"]


def lm_frame(texts, model_name: str = "gpt2", max_tokens: int = 192, batch: int = 16) -> pd.DataFrame:
    """Token statistics of the text under a small language model: the classic 'perplexity' detector signals."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    out = []
    order = np.argsort([len(t) for t in texts])
    texts = list(texts)
    res = [None] * len(texts)
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            idx = order[i : i + batch]
            enc = tok([tok.bos_token + texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
            logits = model(**enc).logits[:, :-1].float()
            tgt = enc.input_ids[:, 1:]
            mask = enc.attention_mask[:, 1:].bool()
            logp = torch.log_softmax(logits, -1)
            tlp = logp.gather(-1, tgt[..., None])[..., 0]
            ent = -(logp.exp() * logp).sum(-1)
            rank = (logp > tlp[..., None]).sum(-1) + 1
            for r, j in enumerate(idx):
                m = mask[r]
                lp, e, rk = tlp[r][m].numpy(), ent[r][m].numpy(), rank[r][m].numpy()
                res[j] = {
                    "lp_mean": lp.mean(), "lp_std": lp.std(), "entropy_mean": e.mean(),
                    "top1_frac": float((rk == 1).mean()), "top10_frac": float((rk <= 10).mean()),
                    "logrank_mean": float(np.log(rk).mean()),
                }
    return pd.DataFrame(res, columns=LM_COLS)
