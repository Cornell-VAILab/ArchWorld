#!/usr/bin/env python
"""Run ArchAttend on ArchWorld scenes.

This is a thin driver. It translates the dataset as you downloaded it into the
inputs the per-model pass expects, then calls that pass once per step size:

    python intervention/run.py --model da3 --root data --masks-dir masks
    python intervention/run.py --model vggt --root data --scene Akdamar_Church

Each run writes the paper's own metrics file per scene and step size,

    <out>/output_<eta>/<scene>/eval_metrics_<first>_<last>.json

holding `base` (the model untouched) and `inter` (the model with ArchAttend), so
one invocation gives you both arms. `aggregate.py` turns a directory of these
into the tables the paper reports.

The anchor element each scene is steered toward ships in assets/anchors.json,
so a run performs Pass 2 only. See `stage_anchor` to supply your own.

For the benchmark numbers without any intervention, use `eval/evaluate.py`.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

ASSETS = HERE / "assets"

# Layer ranges and image count are the published settings for each model.
MODELS = {
    "vggt": {"module": "models.vggt.run_pass", "layers": (0, 23)},
    "da3":  {"module": "models.da3.run_pass",  "layers": (9, 23)},
    "pi3":  {"module": "models.pi3.run_pass",  "layers": (1, 35)},
}
STEP_SIZES = [1, 10, 100, 1000, 5000]


SKYSEG_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"


def ensure_skyseg(path=None):
    """Return a local skyseg.onnx, downloading it on first use.

    DA3 and pi3 build their evaluation cloud with sky pixels removed, which is
    how the published numbers were produced. The model is ~168 MB, so it is
    fetched rather than vendored -- the same file and the same source VGGT's own
    demo uses.
    """
    dest = Path(path).expanduser() if path else ASSETS / "skyseg.onnx"
    if dest.exists():
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading sky segmentation model to {dest} (~168 MB, once)")
    import urllib.request
    tmp = dest.with_suffix(".onnx.part")
    urllib.request.urlretrieve(SKYSEG_URL, tmp)
    tmp.rename(dest)
    return str(dest)


def load_asset(name):
    return json.loads((ASSETS / name).read_text(encoding="utf-8"))


def prepare_scene(root, scene, work_dir, views):
    """Lay out one scene the way the per-model pass expects to find it.

    The pass reads its views from `image_list.json` and locates the ground truth
    through `scene_info.json`; everything else it derives itself.
    """
    # Absolute, because the staged views are symlinked from another directory and
    # a relative target would resolve against that directory instead.
    scene_root = (Path(root).expanduser().resolve() / "archworld" / scene)
    images_src = scene_root / "images"
    missing = [v for v in views if not (images_src / v).exists()]
    if missing:
        raise FileNotFoundError(
            f"{scene}: {len(missing)} of {len(views)} evaluation views are missing "
            f"from {images_src} (first: {missing[0]})")

    # Absolute throughout: the pass derives each view's path relative to
    # image_folder, so the two have to be expressed the same way.
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    staged = work_dir / "images_50"
    staged.mkdir(exist_ok=True)
    # The pass keys masks and depth off the original file name, so the staged
    # copies keep it; the numeric prefix only fixes the view order.
    paths = []
    for i, name in enumerate(views):
        dst = staged / f"{i:03d}_{name}"
        if dst.is_symlink() and not dst.exists():
            dst.unlink()                      # stale link from an interrupted run
        if not dst.exists():
            try:
                os.symlink(images_src / name, dst)
            except OSError:
                shutil.copy2(images_src / name, dst)
        paths.append(str(dst))

    # `original_stems` is how the pass finds each view's SAM 3 masks: the staged
    # copies carry an index prefix, the masks are keyed by the original name.
    (work_dir / "image_list.json").write_text(json.dumps({
        "paths": paths,
        "basenames": [Path(p).name for p in paths],
        "original_stems": [Path(v).stem for v in views],
    }, indent=2))
    (work_dir / "scene_info.json").write_text(json.dumps({
        "recon_path": str(scene_root / "sparse"),
        "depth_folder": str(scene_root / "depth"),
        "image_folder": str(staged),
    }, indent=2))
    return work_dir


def stage_anchor(scene, dest):
    """Hand the pass the scene's anchor element.

    Pass 1 of the method ranks a scene's architectural elements by Semantic
    Cross-View Attention and takes the anchor to be the maximum. Its result is
    what assets/anchors.json holds -- one anchor per scene, shared by all three
    models -- so running this code performs Pass 2 only.

    To choose the anchor yourself instead, run Pass 1 and write the element you
    want here; the pass reads whichever element this file names.
    """
    anchors = load_asset("anchors.json")
    anchor = anchors.get(scene)
    if not anchor:
        raise KeyError(f"{scene} has no anchor element in assets/anchors.json")
    d = Path(dest) / scene
    d.mkdir(parents=True, exist_ok=True)
    # The pass picks the highest-scoring element in this file; with one entry
    # that is the anchor, and the score itself is never used.
    (d / "stats.json").write_text(json.dumps({anchor: {"sum": 1.0}}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--root", default=os.environ.get("ARCHWORLD_ROOT", "data"),
                    help="dataset root (default: $ARCHWORLD_ROOT, else ./data)")
    ap.add_argument("--masks-dir", help="SAM 3 element masks; see make_masks.py")
    ap.add_argument("--scene", action="append", help="repeatable; omit for every extracted scene")
    ap.add_argument("--scenes-file", type=Path, help="one scene name per line")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--eta", default=",".join(map(str, STEP_SIZES)),
                    help="step sizes to sweep (the paper uses all five)")
    ap.add_argument("--skyseg", help="path to skyseg.onnx; downloaded on first use if absent")
    ap.add_argument("--no-sky-filter", action="store_true",
                    help="skip sky removal for DA3 and pi3; results will not match the paper")
    ap.add_argument("--no-icp", action="store_true", help="skip ICP refinement")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--work-dir", type=Path, default=Path(".archattend_work"),
                    help="scratch space for the per-scene anchor files")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not (root / "archworld").is_dir():
        ap.error(f"{root / 'archworld'} not found. Point --root at the extracted dataset.")
    if not args.masks_dir:
        ap.error("--masks-dir is required; generate the element masks with make_masks.py")

    views_by_scene = load_asset("views.json")
    scenes = list(args.scene or [])
    if args.scenes_file:
        scenes += [l.strip() for l in args.scenes_file.read_text().splitlines() if l.strip()]
    if not scenes:
        scenes = sorted(s.name for s in (root / "archworld").iterdir()
                        if (s / "images").is_dir() and s.name in views_by_scene)
    if not scenes:
        ap.error(f"no evaluation scenes found under {root / 'archworld'}")

    spec = MODELS[args.model]
    first, last = spec["layers"]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = args.work_dir / "_attention"

    # The per-model pass is configured through the environment, exactly as the
    # published runs were. Everything below is a published setting.
    os.environ.update({
        "DEVICE": args.device,
        "NUM_IMAGES": "50",
        "USE_ICP": "0" if args.no_icp else "1",
        "ICP_THRESHOLD": "0.1",
        "USE_VALIDITY_MASKS": "1",
        "VALIDITY_MASKS_BASE": str(root / "manual_filtration_masks"),
        "SAM_MASK_DIR": str(Path(args.masks_dir).expanduser()),
        "ATTENTION_STATS_DIR": str(stats_dir),
        "INTERVENTION_LOSS_SCALE": "1e5",
        "FIRST_INTERVENTION_LAYER": str(first),
        "LAST_INTERVENTION_LAYER": str(last),
        "OUTPUT_BASE": str(args.out),
        "IMAGE_MANIFEST": "",
    })
    if args.model in ("da3", "pi3"):
        os.environ["SKYSEG_ONNX"] = "" if args.no_sky_filter else ensure_skyseg(args.skyseg)
    if args.model == "da3":
        os.environ.setdefault("MODEL_DIR", "depth-anything/DA3-LARGE-1.1")
        os.environ.setdefault("PROCESS_RES", "518")
        os.environ.setdefault("PROCESS_RES_METHOD", "upper_bound_resize")
        os.environ.setdefault("REF_VIEW_STRATEGY", "first")

    import importlib
    run_pass = importlib.import_module(spec["module"])

    etas = [int(float(e)) for e in args.eta.split(",") if e.strip()]
    print(f"{args.model}: {len(scenes)} scene(s) x {len(etas)} step size(s)")

    model = None
    failed = []
    for eta in etas:
        os.environ["INTERVENTION_STEP_SIZE"] = str(eta)
        out_dir = args.out / f"output_{eta}"
        for i, scene in enumerate(scenes, 1):
            done = out_dir / scene / f"eval_metrics_{first}_{last}.json"
            if done.exists():
                print(f"[eta={eta}] [{i}/{len(scenes)}] {scene}  done, skipping")
                continue
            print(f"[eta={eta}] [{i}/{len(scenes)}] {scene}")
            try:
                views = views_by_scene.get(scene)
                if not views:
                    raise KeyError(f"{scene} is not one of the benchmark's evaluation scenes")
                stage_anchor(scene, stats_dir)
                # The pass reads its inputs from, and writes its results to, the
                # same directory, so stage the scene where the results will land.
                prepare_scene(root, scene, out_dir / scene, views)
                os.environ["SCENE_NAME"] = scene
                model = run_pass.main(scene_name=scene, work_dir=out_dir / scene, model=model)
            except Exception as exc:
                failed.append((eta, scene, f"{type(exc).__name__}: {exc}"))
                print(f"  failed: {type(exc).__name__}: {exc}")

    if failed:
        print(f"\n{len(failed)} run(s) failed:", file=sys.stderr)
        for eta, scene, msg in failed[:20]:
            print(f"  eta={eta} {scene}: {msg}", file=sys.stderr)
    print(f"\nresults in {args.out}/   aggregate with:  "
          f"python intervention/aggregate.py {args.out} --model {args.model}")


if __name__ == "__main__":
    main()
