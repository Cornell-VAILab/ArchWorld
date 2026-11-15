#!/usr/bin/env python
"""Build the inputs eval/evaluate.py expects from the local ArchWorld dataset.

evaluate.py defaults to the authors' cluster paths (/share/elor/...). Rather than
edit its scoring code, this writes the two inputs it needs for a chosen scene list:

  <out>/scenes.csv                          --geocoded_csv
  <out>/views/<scene>/selected_images.json  --sam_features_dir  (the 50 benchmark
                                            views from intervention/assets/views.json)

    python eric/make_eval_inputs.py --root data --scenes scenes_dev.txt --out results/eval_inputs

Then:

    VGGT_HALF=1 python eval/evaluate.py --model vggt \
        --geocoded_csv results/eval_inputs/scenes.csv \
        --sam_features_dir results/eval_inputs/views \
        --sky_masks_dir results/vggt_base/sky_masks \
        --output_base results/vggt_base/points --output_csv results/vggt_base/metrics

evaluate.py appends the model name to both, giving results/vggt_base/points_vggt/
and results/vggt_base/metrics_vggt.csv.
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data", help="extracted dataset root")
    ap.add_argument("--scenes", type=Path, default=REPO / "scenes_dev.txt",
                    help="one scene per line (default: scenes_dev.txt)")
    ap.add_argument("--out", type=Path, default=Path("results/eval_inputs"))
    args = ap.parse_args()

    archworld = (Path(args.root).expanduser() / "archworld").resolve()
    views = json.loads((REPO / "intervention/assets/views.json").read_text(encoding="utf-8"))
    scenes = [l.strip() for l in args.scenes.read_text(encoding="utf-8").splitlines() if l.strip()]

    rows, skipped = [], []
    for scene in scenes:
        s = archworld / scene
        if scene not in views or not (s / "images").is_dir():
            skipped.append(scene)
            continue
        img, dep, rec = str(s / "images"), str(s / "depth"), str(s / "sparse")
        rows.append({"Category Name": scene, "Image Folder": img,
                     "Depth Map Folder": dep, "Recon Path": rec})
        d = args.out / "views" / scene
        d.mkdir(parents=True, exist_ok=True)
        (d / "selected_images.json").write_text(json.dumps(
            {"image_folder": img, "recon_path": rec, "basenames": views[scene]}, indent=1),
            encoding="utf-8")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "scenes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Category Name", "Image Folder",
                                          "Depth Map Folder", "Recon Path"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} scenes -> {args.out}")
    if skipped:
        print(f"skipped (not on disk or not in views.json): {skipped}")


if __name__ == "__main__":
    main()
