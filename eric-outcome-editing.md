# Eric — VGGT base reproduction: what, outcomes, constraints

Branch: `eric/vggt-base` (from swati's `main` @ `516379d`). Last updated: 2026-11-14.

## Goal

Reproduce the paper's VGGT numbers in two halves, on the same scenes, so they can be compared:

| run | what | how | who |
|---|---|---|---|
| 1 | VGGT + ArchAttend | `intervention/run.py` via `swati/run_sweep.sh` | swati (done) |
| 2 | VGGT base (no intervention) | `eval/evaluate.py` | eric (this doc) |

Scene list: `scenes_dev.txt` (39 scenes, region-stratified, from swati).

## What was done

1. **Data paths.** `evaluate.py` defaults to the authors' cluster (`/share/elor/...`) for the scene CSV,
   the per-scene view lists and the sky-mask cache. Instead of editing its scoring code,
   `eric/make_eval_inputs.py` builds those inputs from the local dataset and a scene list;
   `--sky_masks_dir` is pointed at a local folder.
2. **Image layout.** Ran swati's `tools/flatten_scene_images.py --root data` once (41,850 symlinks
   added on the data disk, nothing moved; 4 duplicate-name images relinked to the size-matching copy).
3. **Precision.** Added swati's opt-in switch to `evaluate.py`: with `VGGT_HALF=1` the VGGT weights
   are stored in half precision (fp16 on the V100). Unset, `evaluate.py` is unchanged.
4. **Per-scene runner.** `eric/run_base_per_scene.sh` runs each scene in its own process so an
   out-of-memory scene is recorded and skipped instead of killing the batch.

Reproduce:

```bash
conda activate archworld
bash eric/run_base_per_scene.sh scenes_dev.txt results/vggt_base
# -> results/vggt_base/metrics_vggt.csv, status.tsv, logs/<scene>.log
```

## Outcomes

- Dendera smoke test (before the per-scene runner): ran end to end with `VGGT_HALF=1`, no NaN.
  Score 0.449 (Acc-mean 0.879, Comp-mean 0.019), 43 of 50 views used.
- First full batch: crashed out of GPU memory on the 2nd scene (`Paul_Kruger_Church`, 50 views),
  which ended the whole batch. Led to the per-scene runner.
- Per-scene run on the V100 (16 GB), interrupted after 12 of 39 scenes by the GPU upgrade:
  **2 ok** (Dendera, Shuri_Castle — both only 43 views), **10 out of memory** (every scene that
  loaded all 50 views). The V100 subset was unusable. Kept in `results/vggt_base_v100/` as a record.
- Switched the VM to an **NVIDIA L4 (24 GB)**. Paul_Kruger_Church (failed on V100) now runs with all
  50 views: score 0.306. All dev scenes are re-run on the L4 from scratch so precision is uniform.
- Per-scene run on the L4, 39 dev scenes: _running — results to be filled in._

## Constraints and caveats

- **GPU: now NVIDIA L4, 24 GB** (was Tesla V100, 16 GB). On the V100, 50 views did not fit for
  VGGT-1B even with `VGGT_HALF=1`; only 43-view scenes did. Likely cause (not verified): VGGT keeps
  all 24 layer outputs in memory during inference (VGGT's README: aggregator alone is 11.4 GB at
  50 frames on an H100). The L4 fits 50 views for base evaluation; whether it fits *every* dev scene
  is what the current run shows. ArchAttend needs more memory than base and may not fit on 24 GB.
- **CPU-bound: 4 vCPUs (`g2-standard-4`).** The GPU only runs the VGGT forward pass (seconds); GT
  point generation, RANSAC + ICP alignment, sky segmentation (ONNX on CPU), GLB export and Chamfer
  run on the CPU, which sits at ~100% on all 4 cores. ~3–5 min per scene, up to ~22 min for heavy
  scenes. `g2-standard-8` (same L4, 8 vCPUs) would roughly halve this; kept 4 vCPUs for now.
- **Resumable.** A VM stop/start loses only the scene in progress; re-running
  `bash eric/run_base_per_scene.sh scenes_dev.txt results/vggt_base` skips scenes with a
  `metrics.json`.
- **The surviving subset is not the dev set.** Scenes that fit are those with fewer usable views or
  smaller images. Numbers are "on the N dev scenes that fit on 16 GB", not "on the dev set":
  - the median can differ from the dev-set median (view count itself affects quality);
  - regional breakdowns can be skewed if some regions drop out;
  - the subset may be small.
  The base-vs-ArchAttend comparison is still fair **if swati's numbers are restricted to the same scenes.**
- **Precision depends on the GPU.** `VGGT_HALF=1` stores weights in bf16 on the L4 (fp16 on the
  V100, which risks overflow). bf16 is safer but can still drift slightly from the paper. Only
  comparable to swati if she ran the same precision: same `VGGT_HALF` setting *and* a GPU of the same
  class (bf16 vs fp16).
- **Different pipelines.** Base comes from `evaluate.py`; swati's ArchAttend from `run_pass.py`.
  They differ in view handling (evaluate.py scores only views with GT and orders them differently),
  ICP and validity masks. The base/ArchAttend gap is not purely the intervention effect.
  The cleanest baseline is `run.py`'s own `base` arm, which swati disabled (`SKIP_BASELINE_PASS=1`).

## Open questions

- [x] ~~Bigger GPU?~~ Upgraded V100 16 GB → L4 24 GB (resource-limited). Recommendation was A100
      40 GB minimum, 80 GB for running ArchAttend sweeps; revisit if L4 still drops scenes or when
      running ArchAttend.
- [ ] Ask swati: which GPU did the ArchAttend sweep run on, and was `VGGT_HALF=1` set?
- [ ] Get swati's per-scene results to restrict them to the surviving subset.
