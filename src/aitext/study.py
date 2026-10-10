"""Train detectors on HC3 and audit them on texts they were not trained on.

    python -m aitext.study --data data --out results

The decision threshold of every detector is fixed once, on the HC3 validation split, so that 5% of human
texts are flagged. Everything else is then measured at that threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ESSAY_SETS, load_all
from .features import LM_COLS, STYLE_COLS, lm_frame, stylometry_frame

TARGET_FPR = 0.05
# (name, human set, machine sets)
AUDITS = [
    ("HC3 test (same domains)", "hc3_test", None),
    ("HC3 held-out domain (CS abstracts)", "hc3_heldout_domain", None),
    ("CS224N abstracts vs GPT-3", ("cs224n_human",), ("cs224n_gpt3",)),
    ("CS224N abstracts vs GPT-3 asked to evade", ("cs224n_human",), ("cs224n_gpt3_evasion",)),
    ("College essays vs GPT-3", ("college_human",), ("college_gpt3",)),
    ("College essays vs GPT-3 asked to evade", ("college_human",), ("college_gpt3_evasion",)),
    ("TOEFL essays vs the same essays polished by GPT-4", ("toefl_human",), ("toefl_gpt4_polished",)),
    ("US student essays vs the same essays simplified by GPT", ("us_student_human",), ("us_student_gpt_simplified",)),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def build_features(df: pd.DataFrame, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    feats = pd.concat([stylometry_frame(df.text), lm_frame(df.text.tolist())], axis=1)
    out = pd.concat([df.reset_index(drop=True), feats], axis=1)
    out.to_parquet(cache)
    return out


def detectors():
    return {
        "stylometry (20 features)": (STYLE_COLS, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)),
        "GPT-2 token statistics (6 features)": (LM_COLS, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        "stylometry + GPT-2": (STYLE_COLS + LM_COLS, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    df = build_features(load_all(Path(a.data)), Path(a.data) / "features.parquet")
    train, val = df[df.set == "hc3_train"], df[df.set == "hc3_val"]

    scores = {}
    for name, (cols, make) in detectors().items():
        m = make().fit(train[cols], train.label)
        scores[name] = m.predict_proba(df[cols])[:, 1]
    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=100000), LogisticRegression(max_iter=2000, C=3))
    tfidf.fit(train.text, train.label)
    scores["TF-IDF words (content baseline)"] = tfidf.predict_proba(df.text)[:, 1]

    thresholds = {n: float(np.quantile(s[(df.set == "hc3_val") & (df.label == 0)], 1 - TARGET_FPR)) for n, s in scores.items()}

    rows, flags = [], []
    for audit, human, machine in AUDITS:
        for name, s in scores.items():
            if machine is None:
                sel = df.set == human
            else:
                sel = df.set.isin(human + machine)
            y = df.label[sel].values
            rows.append({"audit": audit, "detector": name, "n": int(sel.sum()), "auroc": roc_auc_score(y, s[sel.values])})
    for grp, g in df[df.set != "hc3_train"].groupby(["set", "group", "label"]):
        for name, s in scores.items():
            k = int((s[g.index] > thresholds[name]).sum())
            lo, hi = wilson(k, len(g))
            flags.append({"set": grp[0], "group": grp[1], "is_machine_text": grp[2], "detector": name, "n": len(g), "flagged": k, "flag_rate": k / len(g), "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(out / "auroc.csv", index=False)
    pd.DataFrame(flags).to_csv(out / "flag_rates.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps({"target_fpr_on_hc3_val_human": TARGET_FPR, "thresholds": thresholds}, indent=2))
    # how the signals differ between groups of human text
    summary = df[df.set != "hc3_train"].groupby(["group"])[LM_COLS + ["sent_len_cv", "type_token", "word_len_mean"]].mean().round(3)
    summary.to_csv(out / "feature_means_by_group.csv")
    print(pd.DataFrame(rows).pivot(index="audit", columns="detector", values="auroc").round(3).to_string())
    print(pd.DataFrame(flags)[lambda d: d.detector == "stylometry + GPT-2"][["set", "n", "flag_rate"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
