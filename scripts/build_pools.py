"""
Build the per-testbed sampling pools used by every experiment.

For each testbed: a uniform pool (500,000 flows sampled uniformly from the full release) and a
benign pool (up to 200,000 benign flows), plus a statistics file (row counts, class and attack
counts, feature names). IP addresses and absolute timestamps are removed from the features; the
flow start time is kept separately (NF-v3) for the temporal experiments. Non-finite values in the
per-second rate features are replaced, and the replacement counts are recorded.
  python scripts/build_pools.py --raw data/raw --out data/pools
"""
import argparse
import hashlib
import io
import json
import os
import time
import zipfile

import numpy as np
import pandas as pd

N_UNIFORM, N_BENIGN, CHUNK, SEED = 500_000, 200_000, 1_000_000, 20261101
F32_MAX = float(np.finfo(np.float32).max) * 0.99
DROP = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack", "Dataset"}
# Absolute times identify the capture, not the traffic: metadata only, never features.
TIME_META = ("FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS")


class Reservoir:
    """Keeps the rows with the N smallest random keys seen so far."""

    def __init__(self, n):
        self.n, self.keys, self.parts = n, np.empty(0), None

    def add(self, keys, parts):
        if self.parts is None:
            self.parts = {k: v[:0] for k, v in parts.items()}
        keys = np.concatenate([self.keys, keys])
        merged = {k: np.concatenate([self.parts[k], v]) for k, v in parts.items()}
        if len(keys) > self.n:
            keep = np.argpartition(keys, self.n - 1)[:self.n]
            keys = keys[keep]
            merged = {k: v[keep] for k, v in merged.items()}
        self.keys, self.parts = keys, merged


def csv_members(path):
    with zipfile.ZipFile(path) as z:
        return [m for m in z.namelist() if m.lower().endswith(".csv") and "feature" not in m.lower()]


def process(zip_path, member, family, out_dir, split_by_dataset):
    rng = np.random.default_rng(SEED)
    with zipfile.ZipFile(zip_path) as z:
        header = pd.read_csv(io.TextIOWrapper(z.open(member)), nrows=0).columns.tolist()
    feats = [c for c in header if c not in DROP and c not in TIME_META]
    usecols = feats + [c for c in ("Label", "Attack", "Dataset") + TIME_META if c in header]
    state, attack_codes, t0, seen = {}, {}, time.time(), 0

    def get(tb):
        if tb not in state:
            state[tb] = {"uni": Reservoir(N_UNIFORM), "ben": Reservoir(N_BENIGN),
                         "count": {"benign": 0, "attack": 0}, "attack_counts": {},
                         "sum": np.zeros((2, len(feats))), "sumsq": np.zeros((2, len(feats))),
                         "nonfinite": np.zeros(len(feats), np.int64)}
        return state[tb]

    with zipfile.ZipFile(zip_path) as z:
        reader = pd.read_csv(io.TextIOWrapper(z.open(member)), usecols=usecols, chunksize=CHUNK, low_memory=False)
        for chunk in reader:
            rows = np.arange(seen, seen + len(chunk), dtype=np.int64)
            seen += len(chunk)
            try:  # fast path: all-numeric chunk
                X = chunk[feats].to_numpy(np.float64)
            except (ValueError, TypeError):
                X = chunk[feats].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)

            # zero-length flows). Non-finite values are counted, then set to 0 as in the original release preprocessing
            # pipeline; finite values beyond float32 range are clipped so the float32 pools stay finite.
            nonfinite = ~np.isfinite(X)
            X[nonfinite] = 0.0
            np.clip(X, -F32_MAX, F32_MAX, out=X)
            if "Label" in chunk:
                y = chunk["Label"].astype(int).to_numpy(np.int8)
            else:
                y = (chunk["Attack"].astype(str).str.lower() != "benign").to_numpy(np.int8)
            att_names = chunk["Attack"].astype(str).to_numpy() if "Attack" in chunk else np.full(len(chunk), "?")
            for a in np.unique(att_names):
                attack_codes.setdefault(a, len(attack_codes))
            att = np.vectorize(attack_codes.get, otypes=[np.int16])(att_names)
            ts = chunk[TIME_META[0]].to_numpy(np.int64) if TIME_META[0] in chunk else np.full(len(chunk), -1, np.int64)
            groups = chunk["Dataset"].to_numpy() if split_by_dataset else np.full(len(chunk), family_name(member))
            L = np.log1p(np.maximum(X, 0.0))
            for tb in np.unique(groups):
                m = groups == tb
                st = get(tb)
                st["nonfinite"] += nonfinite[m].sum(0)
                for cls, name in ((0, "benign"), (1, "attack")):
                    mc = m & (y == cls)
                    st["count"][name] += int(mc.sum())
                    st["sum"][cls] += L[mc].sum(0)
                    st["sumsq"][cls] += (L[mc] ** 2).sum(0)
                for a, c in zip(*np.unique(att_names[m], return_counts=True)):
                    st["attack_counts"][a] = st["attack_counts"].get(a, 0) + int(c)
                parts = {"X": X[m].astype(np.float32), "y": y[m], "attack": att[m], "row": rows[m], "t_start": ts[m]}
                st["uni"].add(rng.random(int(m.sum())), parts)
                mb = y[m] == 0
                if mb.any():
                    st["ben"].add(rng.random(int(mb.sum())), {k: v[mb] for k, v in parts.items()})
            print(f"  {member}: {seen:,} rows, {time.time() - t0:6.0f}s", flush=True)

    os.makedirs(os.path.join(out_dir, family), exist_ok=True)
    written = []
    for tb, st in state.items():
        base = os.path.join(out_dir, family, tb)
        for pool in ("uni", "ben"):
            p = st[pool].parts
            np.savez(f"{base}_{'uniform' if pool == 'uni' else 'benign'}.npz", features=np.array(feats), **p)
        stats = {"testbed": tb, "family": family, "source_member": member, "features": feats,
                 "rows_total": st["count"]["benign"] + st["count"]["attack"], "count": st["count"],
                 "attack_counts": st["attack_counts"], "attack_code_names": {v: k for k, v in attack_codes.items()},
                 "log1p_sum": st["sum"].tolist(), "log1p_sumsq": st["sumsq"].tolist(),
                 "nonfinite_raw_counts": {f: int(c) for f, c in zip(feats, st["nonfinite"]) if c},
                 "pool_sizes": {"uniform": int(len(st["uni"].keys)), "benign": int(len(st["ben"].keys))},
                 "seed": SEED}
        json.dump(stats, open(f"{base}_stats.json", "w"), indent=1)
        written.append(tb)
        print(f"  wrote {family}/{tb}: total {stats['rows_total']:,} flows, benign {st['count']['benign']:,}, "
              f"pools {stats['pool_sizes']}", flush=True)
    return written


CANONICAL = {"NF-CICIDS2018-v3": "NF-CSE-CIC-IDS2018-v3"}  # file name -> canonical testbed name


def family_name(member):
    name = os.path.splitext(os.path.basename(member))[0]
    return CANONICAL.get(name, name)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def main(args):
    jobs = [("NF-UQ-NIDS-v2.zip", "nfv2", True)] + [
        (f"NF-{n}-v3.zip", "nfv3", False) for n in ("UNSW-NB15", "BoT-IoT", "ToN-IoT", "CSE-CIC-IDS2018")]
    manifest = {}
    for fname, family, split in jobs:
        zpath = os.path.join(args.raw, fname)
        if not os.path.exists(zpath):
            print(f"skip (missing): {zpath}")
            continue
        for member in csv_members(zpath):
            if family == "nfv2" and "NF-UQ-NIDS-v2" not in member:
                continue
            print(f"[{family}] {fname}:{member}", flush=True)
            for tb in process(zpath, member, family, args.out, split):
                for suffix in ("_uniform.npz", "_benign.npz", "_stats.json"):
                    f = os.path.join(args.out, family, tb + suffix)
                    manifest[f"{family}/{tb}{suffix}"] = sha256(f)
    json.dump(manifest, open(os.path.join(args.out, "pools_manifest.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--out", default=os.path.join("data", "pools"))
    main(ap.parse_args())
