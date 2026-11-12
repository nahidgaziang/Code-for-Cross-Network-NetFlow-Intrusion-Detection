"""
Determinism of every method: two runs with the same configuration and seed must give
bitwise-identical scores on the validation, buffer, window and evaluation arrays.
  python tests/test_determinism.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crossnet.data import FoldConfig, build_fold  # noqa: E402
from crossnet.methods import METHODS, run_method, sample_configs  # noqa: E402

inp, _, _ = build_fold(FoldConfig(family="nfv2", target="NF-ToN-IoT-v2", seed=3, n_src=8000, n_eval=8000))
bad = []
for method in METHODS:
    hp = sample_configs(method, 3, seed=0)[2]
    a, _ = run_method(method, inp, hp, seed=11)
    b, _ = run_method(method, inp, hp, seed=11)
    same = all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in ("val", "buf", "win", "eval"))
    finite = all(np.isfinite(np.asarray(a[k])).all() for k in a)
    print(f"{method:9s} deterministic: {same}  finite: {finite}", flush=True)
    if not (same and finite):
        bad.append(method)
print("RESULT:", "PASS" if not bad else f"FAIL {bad}")
