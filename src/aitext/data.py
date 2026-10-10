"""Build the labelled corpora. Every text is real: human-written, or written by ChatGPT, GPT-3 or GPT-4.

All texts are cut to their first MAX_WORDS words so that length does not decide the label.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MAX_WORDS = 160
MIN_WORDS = 40
HC3_TRAIN_DOMAINS = {"reddit_eli5": 2000, "finance": 2000, "medicine": 1000, "open_qa": 1000}
HC3_HELD_OUT_DOMAIN = "wiki_csai"

# name -> (file, label, group, description). label 1 = machine-written or machine-edited.
ESSAY_SETS = {
    "toefl_human": ("TOEFL_real_91", 0, "non-native human", "TOEFL essays by non-native English writers"),
    "toefl_gpt4_polished": ("TOEFL_gpt4polished_91", 1, "human text, GPT-4 polished", "the same TOEFL essays rewritten by GPT-4"),
    "us_student_human": ("HewlettStudentEssay_real_88", 0, "native human", "US 8th-grade essays (Hewlett)"),
    "us_student_gpt_simplified": ("HewlettStudentEssay_GPTsimplify_88", 1, "human text, GPT simplified", "the US essays simplified by GPT"),
    "cs224n_human": ("CS224N_real_145", 0, "academic human", "Stanford CS224N project abstracts"),
    "cs224n_gpt3": ("CS224N_gpt3_145", 1, "GPT-3", "GPT-3 abstracts for the same projects"),
    "cs224n_gpt3_evasion": ("CS224N_gpt3PromptEng_145", 1, "GPT-3, prompted to evade", "GPT-3 abstracts with a prompt asking for more literary language"),
    "college_human": ("CollegeEssay_real_70", 0, "native human", "US college-application essays"),
    "college_gpt3": ("CollegeEssay_gpt3_31", 1, "GPT-3", "GPT-3 essays on the same prompts"),
    "college_gpt3_evasion": ("CollegeEssay_gpt3PromptEng_31", 1, "GPT-3, prompted to evade", "the same, with the evasion prompt"),
}


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return " ".join(text.split(" ")[:MAX_WORDS])


def hc3(data_dir: Path, seed: int = 0) -> pd.DataFrame:
    """One human and one ChatGPT answer per question. Columns: text, label, domain, qid, split."""
    rng = np.random.default_rng(seed)
    rows = []
    for qid, line in enumerate((data_dir / "hc3_all.jsonl").open(encoding="utf-8")):
        r = json.loads(line)
        if not r["human_answers"] or not r["chatgpt_answers"]:
            continue
        rows.append((r["source"], qid, clean(r["human_answers"][0]), clean(r["chatgpt_answers"][0])))
    df = pd.DataFrame(rows, columns=["domain", "qid", "human", "chatgpt"])
    df = df[(df.human.str.split().str.len() >= MIN_WORDS) & (df.chatgpt.str.split().str.len() >= MIN_WORDS)]
    parts = []
    for dom, n in {**HC3_TRAIN_DOMAINS, HC3_HELD_OUT_DOMAIN: 10**9}.items():
        d = df[df.domain == dom]
        d = d.iloc[rng.permutation(len(d))[:n]].copy()
        if dom == HC3_HELD_OUT_DOMAIN:
            d["split"] = "heldout_domain"
        else:
            d["split"] = rng.choice(["train", "val", "test"], size=len(d), p=[0.7, 0.15, 0.15])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    long = pd.concat(
        [
            d.assign(text=d.human, label=0, group="human"),
            d.assign(text=d.chatgpt, label=1, group="ChatGPT"),
        ]
    )
    long["set"] = "hc3_" + long["split"]
    return long[["text", "label", "group", "set", "domain", "qid"]].reset_index(drop=True)


def essays(data_dir: Path) -> pd.DataFrame:
    rows = []
    for name, (file, label, group, _) in ESSAY_SETS.items():
        for item in json.loads((data_dir / f"{file}.json").read_text(encoding="utf-8")):
            text = clean(item["document"])
            if len(text.split()) >= MIN_WORDS:
                rows.append((text, label, group, name, name, -1))
    return pd.DataFrame(rows, columns=["text", "label", "group", "set", "domain", "qid"])


def load_all(data_dir: Path) -> pd.DataFrame:
    return pd.concat([hc3(data_dir), essays(data_dir)], ignore_index=True)
