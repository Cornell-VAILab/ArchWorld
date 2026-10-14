#!/usr/bin/env python
"""Segment architectural elements with SAM 3, for ArchAttend to steer.

Pass 2 steers the model toward one element per scene, so by default this
segments exactly that element -- the anchor named in `assets/anchors.json`:

    python make_masks.py --root data --out masks
    python make_masks.py --root data --out masks --scene Akdamar_Church

Masks are written as

    masks/<scene>/mask/<view stem>_<element>.npy

which is the layout `run.py --masks-dir` expects. Existing files are skipped, so
the script can be resumed after an interruption.

To determine the anchor yourself rather than using the one we ship, pass
`--all-elements`. That segments every element in `assets/elements.json` -- the
vocabulary annotated per architecture type -- which is what Pass 1 ranks. It is
roughly twenty times the work.

Requires SAM 3 (`sam3`) importable; everything else comes from this repository.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
CONFIDENCE = 0.5   # the SAM 3 confidence threshold the paper's masks were made with


def load_asset(name):
    return json.loads((ASSETS / name).read_text(encoding="utf-8"))


def elements_for(scene, all_elements=False):
    """The elements to segment: every one Pass 1 ranks, or just the anchor."""
    if all_elements:
        return load_asset("elements.json").get(scene, [])
    anchor = load_asset("anchors.json").get(scene)
    return [anchor] if anchor else []


def views_for(scene):
    return load_asset("views.json").get(scene, [])


def segment_scene(processor, root, scene, out_dir, overwrite, verbose, all_elements=False):
    elements = elements_for(scene, all_elements)
    if not elements:
        print(f"  {scene}: no element listed for it, skipping")
        return 0
    img_dir = Path(root) / "archworld" / scene / "images"
    images = [img_dir / v for v in views_for(scene) if (img_dir / v).exists()]
    if not images:
        print(f"  {scene}: no views found under {img_dir}, skipping")
        return 0

    # The passes look for <masks-dir>/<scene>/mask/<view>_<element>.npy, so the
    # "mask" level is part of the contract, not a stylistic choice.
    dest = Path(out_dir) / scene / "mask"
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, image_path in enumerate(images, 1):
        stem = image_path.stem
        todo = [e for e in elements if overwrite or not (dest / f"{stem}_{e}.npy").exists()]
        if not todo:
            continue
        image = Image.open(image_path).convert("RGB")
        state = processor.set_image(image)
        for element in todo:
            try:
                processor.reset_all_prompts(state)
                masks = processor.set_text_prompt(state=state, prompt=element)["masks"]
            except Exception as exc:
                print(f"    {stem} / {element}: {type(exc).__name__}: {exc}")
                continue
            if len(masks):
                arr = (masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks))
                arr = arr.astype(np.uint8)
            else:
                # An empty array records "looked for it, found nothing", which is
                # what keeps a re-run from retrying every absent element.
                arr = np.array([], dtype=np.uint8)
            np.save(dest / f"{stem}_{element}.npy", arr)
            written += 1
        if verbose:
            print(f"  {scene}: view {i}/{len(images)} ({stem})")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="dataset root (default: $ARCHWORLD_ROOT, else ./data)")
    ap.add_argument("--out", required=True, help="where to write the masks")
    ap.add_argument("--scene", action="append", help="repeatable; omit for every extracted scene")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--all-elements", action="store_true",
                    help="segment every element, not just the anchor; needed only to run Pass 1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    args = ap.parse_args()

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    root = Path(args.root or __import__("os").environ.get("ARCHWORLD_ROOT", "data")).expanduser()
    scenes = args.scene
    if not scenes:
        d = root / "archworld"
        views = load_asset("views.json")
        scenes = sorted(p.name for p in d.iterdir()
                        if (p / "images").is_dir() and p.name in views) if d.is_dir() else []
    if not scenes:
        ap.error(f"no evaluation scenes found under {root / 'archworld'}")

    print(f"loading SAM 3 on {args.device}")
    processor = Sam3Processor(build_sam3_image_model(device=args.device),
                              confidence_threshold=CONFIDENCE)

    total = 0
    for i, scene in enumerate(scenes, 1):
        print(f"[{i}/{len(scenes)}] {scene}")
        total += segment_scene(processor, root, scene, args.out, args.overwrite,
                           args.verbose, args.all_elements)
    print(f"\nwrote {total} mask files to {args.out}/")
    print(f"now run:  python intervention/run.py --model da3 --root {root} "
          f"--masks-dir {args.out}")


if __name__ == "__main__":
    main()
