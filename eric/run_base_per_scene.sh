#!/usr/bin/env bash
# VGGT base (no intervention) with eval/evaluate.py, one scene per process, so a
# scene that runs out of GPU memory is recorded and skipped instead of stopping
# the batch. Resumable: scenes with a metrics.json are not recomputed.
#
#   conda activate archworld
#   bash eric/run_base_per_scene.sh [scenes_file] [out_dir]
#
# Defaults: scenes_dev.txt, results/vggt_base. VGGT_HALF=1 unless set otherwise.
# Writes <out>/metrics_vggt.csv (per-scene scores), <out>/logs/<scene>.log and
# <out>/status.tsv (scene, status ok|oom|fail, views used by the alignment).
set -u
cd "$(dirname "$0")/.."
SCENES=${1:-scenes_dev.txt}
OUT=${2:-results/vggt_base}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VGGT_HALF=${VGGT_HALF:-1}

python eric/make_eval_inputs.py --root data --scenes "$SCENES" --out "$OUT/eval_inputs" || exit 1
mkdir -p "$OUT/logs"
STATUS="$OUT/status.tsv"
printf "scene\tstatus\tviews_used\n" > "$STATUS"

while read -r S; do
  [ -z "$S" ] && continue
  LOG="$OUT/logs/$S.log"
  if [ ! -f "$OUT/points_vggt/$S/metrics.json" ]; then
    echo "[$(date +%H:%M:%S)] $S"
    python eval/evaluate.py --model vggt --scene "$S" \
      --geocoded_csv "$OUT/eval_inputs/scenes.csv" \
      --sam_features_dir "$OUT/eval_inputs/views" \
      --sky_masks_dir "$OUT/sky_masks" \
      --output_base "$OUT/points" --output_csv "$OUT/metrics" > "$LOG" 2>&1
  fi
  if [ -f "$OUT/points_vggt/$S/metrics.json" ]; then st=ok
  elif grep -q "OutOfMemoryError" "$LOG" 2>/dev/null; then st=oom
  else st=fail; fi
  n=$(grep -oP "Computing alignment from \K[0-9]+" "$LOG" 2>/dev/null | tail -1)
  printf "%s\t%s\t%s\n" "$S" "$st" "${n:--}" >> "$STATUS"
  echo "  -> $st (${n:--} views)"
done < "$SCENES"

echo; cut -f2 "$STATUS" | tail -n +2 | sort | uniq -c
