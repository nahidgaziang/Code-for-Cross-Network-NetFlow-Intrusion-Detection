"""
Download the NetFlow v2 and v3 benchmark releases (NF-UNSW-NB15, NF-BoT-IoT, NF-ToN-IoT,
NF-CSE-CIC-IDS2018) from their public repositories and record SHA-256 checksums.
  python scripts/download_datasets.py --out data/raw
"""
import argparse
import hashlib
import json
import os
import time
import urllib.request

API = "https://api.rdm.uq.edu.au/datasets/{}/download"
DATASETS = {
    # name: (UQ RDM uuid, DOI)
    "NF-UQ-NIDS-v2": ("e2412450-ef9c-11ed-827d-e762de186848", "UQ RDM (NF-UQ-NIDS-v2, Sarhan et al.)"),
    "NF-UNSW-NB15-v3": ("abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f", "10.48610/6E0EDA1"),
    "NF-BoT-IoT-v3": ("db0187cc-1f78-4215-8e34-303364498866", "10.48610/73C4EBC"),
    "NF-ToN-IoT-v3": ("343e2e8c-6e6e-4a0c-813d-a46acea1b7f4", "10.48610/44D7C5E"),
    "NF-CSE-CIC-IDS2018-v3": ("4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22", "10.48610/ECE9B83"),
}
UA = {"User-Agent": "Mozilla/5.0"}
BLOCK = 8 * 1024 * 1024


def signed_url(uuid):
    with urllib.request.urlopen(urllib.request.Request(API.format(uuid), headers=UA), timeout=30) as r:
        return json.loads(r.read().decode())["url"]


def total_size(url):
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


def download(name, uuid, out_dir, retries=8):
    path = os.path.join(out_dir, f"{name}.zip")
    for attempt in range(1, retries + 1):
        try:
            url = signed_url(uuid)
            size = total_size(url)
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have == size:
                return path, size
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={have}-"})
            t0, done = time.time(), have
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "ab") as fh:
                while True:
                    buf = r.read(BLOCK)
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                    print(f"  {name}: {done / 1e9:5.2f}/{size / 1e9:.2f} GB  {rate:5.1f} MB/s", flush=True)
            if os.path.getsize(path) == size:
                return path, size
        except Exception as exc:  # network hiccup or expired link: resume with a fresh link
            print(f"  {name}: attempt {attempt} failed ({exc}); retrying", flush=True)
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"{name}: download failed after {retries} attempts")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(BLOCK), b""):
            h.update(buf)
    return h.hexdigest()


def main(args):
    os.makedirs(args.out, exist_ok=True)
    mpath = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for name, (uuid, doi) in DATASETS.items():
        print(f"[{name}]", flush=True)
        path, size = download(name, uuid, args.out)
        manifest[name] = {"file": os.path.basename(path), "bytes": size, "sha256": sha256(path),
                          "uq_rdm_uuid": uuid, "doi": doi,
                          "downloaded": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(manifest, open(mpath, "w"), indent=2)
        print(f"  done: {size / 1e9:.2f} GB, sha256 {manifest[name]['sha256'][:16]}...", flush=True)
    print("All downloads complete; manifest:", mpath)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "raw"))
    main(ap.parse_args())
