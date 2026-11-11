#!/usr/bin/env python
"""Download only a stratified subset of ArchWorld (the full thing doesn't fit in 250GB).

    conda activate archworld
    python 02_get_subset.py --list                  # see what's in the HF repo first
    python 02_get_subset.py --n-dev 40 --n-test 40  # dev scenes + held-out scenes

Writes ~/ArchWorld/scenes_dev.txt and scenes_test.txt (share these with the team so
everyone's numbers are on the same scenes), downloads each scene tarball, extracts it,
and deletes the tarball right away so disk use ~= extracted size only.
"""
import argparse, random, subprocess, tarfile
from pathlib import Path
from collections import defaultdict

import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "ArchWorld/ArchWorld"
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path.home() / "ArchWorld" / "data"))
ap.add_argument("--n-dev", type=int, default=40)
ap.add_argument("--n-test", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--list", action="store_true")
args = ap.parse_args()
root = Path(args.root)
root.mkdir(parents=True, exist_ok=True)

files = list_repo_files(REPO, repo_type="dataset")
if args.list:
    for f in files:
        print(f)
    raise SystemExit

def fetch_extract(fname):
    p = Path(hf_hub_download(REPO, fname, repo_type="dataset", local_dir=root))
    if p.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(p) as t:
            t.extractall(root)
        p.unlink()  # free the space immediately

# 1) small shared files first: metadata csv + anything that isn't a per-scene tarball
#    (e.g. manual_filtration_masks). Check `--list` output if a name looks off.
tarballs = [f for f in files if f.endswith((".tar.gz", ".tgz", ".tar"))]
shared = [f for f in files if f not in tarballs and not f.startswith(".")]
for f in shared:
    hf_hub_download(REPO, f, repo_type="dataset", local_dir=root)
meta = pd.read_csv(next(root.rglob("ArchWorld_metadata.csv")))

# map scene name -> its tarball
by_scene = {Path(f).name.split(".tar")[0]: f for f in tarballs}
non_scene = [f for s, f in by_scene.items() if s not in set(meta.scene)]
print("tarballs that aren't scenes (downloaded as shared):", non_scene)
for f in non_scene:
    fetch_extract(f)

# 2) stratified split by Region, so every region shows up in dev AND test
random.seed(args.seed)
groups = defaultdict(list)
for r in meta.itertuples():
    if r.scene in by_scene:
        groups[r.Region].append(r.scene)
dev, test = [], []
total = sum(len(v) for v in groups.values())
for region, scenes in sorted(groups.items(), key=lambda kv: str(kv[0])):
    random.shuffle(scenes)
    k_dev = max(1, round(args.n_dev * len(scenes) / total))
    k_test = max(1, round(args.n_test * len(scenes) / total))
    dev += scenes[:k_dev]
    test += scenes[k_dev:k_dev + k_test]

repo_dir = Path.home() / "ArchWorld"
(repo_dir / "scenes_dev.txt").write_text("\n".join(dev) + "\n")
(repo_dir / "scenes_test.txt").write_text("\n".join(test) + "\n")
print(f"dev={len(dev)} test={len(test)}")

# 3) download + extract
for i, s in enumerate(dev + test, 1):
    if (root / "archworld" / s / "images").is_dir():
        continue
    print(f"[{i}/{len(dev)+len(test)}] {s}", flush=True)
    fetch_extract(by_scene[s])
subprocess.run(["df", "-h", str(root)])
