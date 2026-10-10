"""Figure: share of each text group flagged as machine-written at the 5%-false-positive threshold."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
f = pd.read_csv(R / "flag_rates.csv")
rows = [
    ("hc3_test", "ChatGPT answers vs human\n(in distribution, human)"),
    ("hc3_heldout_domain", "same, new domain\n(human)"),
    ("us_student_human", "US 8th-grade essays\n(human)"),
    ("college_human", "college essays\n(human)"),
    ("cs224n_human", "CS224N abstracts\n(human)"),
    ("toefl_human", "TOEFL essays,\nnon-native (human)"),
    ("toefl_gpt4_polished", "TOEFL essays\npolished by GPT-4"),
    ("us_student_gpt_simplified", "US essays simplified\nby GPT"),
    ("cs224n_gpt3", "GPT-3 abstracts"),
    ("cs224n_gpt3_evasion", "GPT-3 abstracts,\nprompted to evade"),
]
dets = ["stylometry (20 features)", "GPT-2 token statistics (6 features)", "stylometry + GPT-2", "TF-IDF words (content baseline)"]
fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.2
for i, d in enumerate(dets):
    vals, lo, hi = [], [], []
    for s, _ in rows:
        r = f[(f.set == s) & (f.detector == d)]
        r = r[r.is_machine_text == r.is_machine_text.max()] if s == "hc3_test" and False else r
        r = r.iloc[0] if s != "hc3_test" and s != "hc3_heldout_domain" else r[r.is_machine_text == 0].iloc[0]
        vals.append(r.flag_rate)
        lo.append(r.flag_rate - r.ci_lo)
        hi.append(r.ci_hi - r.flag_rate)
    ax.bar(np.arange(len(rows)) + (i - 1.5) * w, vals, w, yerr=[lo, hi], capsize=1.5, label=d)
ax.axhline(0.05, color="k", ls="--", lw=0.8)
ax.set_xticks(np.arange(len(rows)), [r[1] for r in rows], fontsize=7)
ax.set_ylabel("flagged as machine-written")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(R / "flag_rates.png", dpi=140)
