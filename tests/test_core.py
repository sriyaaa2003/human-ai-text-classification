import numpy as np

from aitext.data import MAX_WORDS, clean
from aitext.features import STYLE_COLS, stylometry, stylometry_frame
from aitext.study import wilson


def test_clean_collapses_whitespace_and_truncates():
    text = "word  " * (MAX_WORDS + 50)
    out = clean(text)
    assert len(out.split()) == MAX_WORDS and "  " not in out


def test_stylometry_has_every_declared_column_and_sane_values():
    f = stylometry("However, I can't say. It was explained (twice)! Why? Because 2 + 2 = 4.")
    assert set(f) == set(STYLE_COLS)
    assert f["transition"] > 0 and f["contraction"] > 0 and f["paren"] > 0 and f["first_person"] > 0
    assert 0 < f["type_token"] <= 1


def test_burstiness_is_higher_for_mixed_sentence_lengths():
    flat = stylometry("One two three four. Five six seven eight. Nine ten eleven twelve.")
    mixed = stylometry("Go. This sentence is deliberately much longer than the one before it by a wide margin. Stop.")
    assert mixed["sent_len_cv"] > flat["sent_len_cv"]


def test_empty_text_does_not_crash():
    df = stylometry_frame([""])
    assert list(df.columns) == STYLE_COLS and np.isfinite(df.values).all()


def test_wilson_interval_brackets_the_rate_and_handles_edges():
    lo, hi = wilson(5, 100)
    assert lo < 0.05 < hi
    assert wilson(0, 10)[0] == 0 or wilson(0, 10)[0] < 1e-9
    assert np.isnan(wilson(0, 0)[0])
