"""
Leakage and disjointness tests for crossnet/data.py: evaluation, buffer and window flows are
disjoint; no target flow enters training or validation; folds are reproducible from their
configuration.
  python -m pytest tests -q
"""
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crossnet.data import FoldConfig, MethodInput, TESTBEDS, build_fold  # noqa: E402


def _cfg(**kw):
    fam = os.environ.get("NF_TEST_FAMILY", "nfv2")
    return FoldConfig(family=fam, target=os.environ.get("NF_TEST_TARGET", TESTBEDS[fam][0]),
                      n_src=int(os.environ.get("NF_TEST_NSRC", 20000)),
                      n_eval=int(os.environ.get("NF_TEST_NEVAL", 10000)), **kw)


def test_method_input_has_no_target_labels():
    names = {f.name for f in dataclasses.fields(MethodInput)}
    assert not any(n in names for n in ("y_eval", "y_buf", "y_win", "attack_eval")), names


def test_target_sets_are_disjoint():
    _, truth, meta = build_fold(_cfg())
    ev, buf, win = (set(meta[k]) for k in ("rows_eval", "rows_buf", "rows_win"))
    assert not (ev & buf) and not (ev & win) and not (buf & win)
    assert len(truth.y_eval) == len(meta["rows_eval"])


def test_window_contamination_matches_config():
    _, truth, _ = build_fold(_cfg(win_attack_frac=0.05))
    frac = float(truth.win_is_attack.mean())
    assert abs(frac - 0.05) < 0.02, frac


def test_fold_is_deterministic():
    a, _, ma = build_fold(_cfg(seed=7))
    b, _, mb = build_fold(_cfg(seed=7))
    assert np.array_equal(a.X_tr, b.X_tr) and np.array_equal(ma["rows_eval"], mb["rows_eval"])


def test_sources_exclude_target():
    inp, _, _ = build_fold(_cfg())
    assert _cfg().target not in inp.source_names and len(inp.source_names) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
