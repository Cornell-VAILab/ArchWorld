#!/usr/bin/env bash
# ArchAttend eta sweep on VGGT, scene by scene (all 5 eta per scene). Resumable.
# Setup: python tools/flatten_scene_images.py --root data ; masks via intervention/make_masks.py
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SAVE_RAW_TENSORS=1 SKIP_BASELINE_PASS=1
SCENES=${1:-scenes_dev.txt}; OUT=${2:-results/vggt_dev}
while read -r S; do
  python intervention/run.py --model vggt --root data --masks-dir masks --scene "$S" \
    --eta 1,10,100,1000,5000 --out "$OUT"
done < "$SCENES"
