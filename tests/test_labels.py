import numpy as np

from mtah.labels import controlled
from mtah.schema import A, D, H


def _seq(*parts):
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in parts])


def test_short_spell_returns_to_previous_team():
    labels = _seq((H, 100), (A, 20), (H, 100))
    assert (controlled(labels, ((0, 220),), 2.0) == H).all()


def test_short_spell_after_stoppage_goes_to_next_team():
    labels = _seq((D, 50), (A, 20), (H, 100))
    out = controlled(labels, ((0, 170),), 2.0)
    assert (out[:50] == D).all() and (out[50:] == H).all()


def test_short_spell_between_stoppages_is_kept_and_input_untouched():
    labels = _seq((D, 50), (A, 20), (D, 50))
    before = labels.copy()
    assert (controlled(labels, ((0, 120),), 2.0) == labels).all()
    assert (labels == before).all()


def test_zero_threshold_changes_nothing():
    labels = _seq((H, 3), (A, 2), (H, 3))
    assert (controlled(labels, ((0, 8),), 0.0) == labels).all()
