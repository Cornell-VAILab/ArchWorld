#!/usr/bin/env python
"""Make every scene's views reachable directly under images/ (and depth/).

Some downloaded scenes keep their photos in category subfolders
(images/<category>/<view>.jpg, images/commons/<view>.jpg), but run.py and
evaluate.py look for images/<view>.jpg. This adds symlinks one level up;
the original files are never moved or modified. Safe to re-run.

When the same file name exists in two subfolders with different content, the
copy whose size matches the view's depth map is used.

    python tools/flatten_scene_images.py --root data
"""
import argparse, json, os
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--views", default="intervention/assets/views.json")
    a = ap.parse_args()
    root = Path(a.root) / "archworld"
    views = json.load(open(a.views))

    made = 0
    for scene in root.iterdir():
        for sub in ("images", "depth"):
            top = scene / sub
            if not top.is_dir():
                continue
            for f in top.rglob("*"):
                if f.is_file() and f.parent != top:
                    dst = top / f.name
                    if not (dst.exists() or dst.is_symlink()):
                        os.symlink(os.path.relpath(f, top), dst)
                        made += 1

    relinked = 0
    for s, names in views.items():
        top = root / s / "images"
        if not top.is_dir():
            continue
        for name in names:
            d = list((root / s / "depth").rglob(name + ".npy"))
            t = top / name
            if not d or not t.exists():
                continue
            H, W = np.load(d[0], mmap_mode="r").shape[:2]
            if Image.open(t).size == (W, H):
                continue
            ok = [c for c in top.rglob(name) if c.resolve().is_file() and Image.open(c).size == (W, H)]
            if ok and t.is_symlink():
                t.unlink()
                os.symlink(os.path.relpath(ok[0].resolve(), top), t)
                relinked += 1

    have = {p.name for p in root.iterdir() if p.is_dir()}
    missing = {s: sum(not (root / s / "images" / x).exists() for x in xs)
               for s, xs in views.items() if s in have}
    missing = {k: v for k, v in missing.items() if v}
    print(f"links added: {made}, relinked to size-matching copy: {relinked}")
    print(f"scenes on disk not in views.json (skip these): {sorted(have - set(views))}")
    print(f"benchmark scenes still missing views: {missing or 'none'}")


if __name__ == "__main__":
    main()
