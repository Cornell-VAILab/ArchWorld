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
- **Per-scene run on the L4, 39 dev scenes: 39 / 39 ok** (21:36 → 00:58, incl. retry).
  37 ok in the main pass; the 2 out-of-memory scenes below passed on retry with the GPU idle.
  Results: **`eric/vggt_base_dev_results.csv`** (one row per scene: region, type, views used,
  Acc/Comp mean & median, CD, CD_m, CD_norm). Every `metrics.json` matches `metrics_vggt.csv`.
  - **Median over 39 scenes: CD 0.140 · CD_m 1.25 m · CD_norm 20.13** (CD_norm as in `aggregate.py`).
  - By region (median CD_norm, n): Africa 48.05 (4), East Asia 35.44 (4), North America 34.64 (5),
    SE Asia 21.32 (5), South Asia 19.95 (6), Europe 10.62 (5), Latin America 9.33 (5),
    West/Central Asia 4.74 (5).
  - By type: Church/Cathedral 16.81 (12), Misc 31.36 (11), Public Building 17.29 (6),
    Mosque 19.81 (4), Tower 16.08 (3); Monument, Temple/Shrine and Tomb have 1 scene each.
  - Best: Qila-i-Kuhna_mosque 3.47, Golestan_Palace 3.74. Worst: Baguio_Cathedral 110.1,
    Mount_Rushmore 255.3 — one outlier, so use medians, not means.
  - Bell_tower_of_Vank_Cathedral used only 21 views (few of its views have GT depth); all others 42–50.
  - n is 4–6 per region: regional medians on the dev split are noisy.
  - First 15 scenes all ok. Then out-of-memory at 23:28–23:35 on `Santa_Prisca_Church_(Taxco_de_Alarcón)`
    (#16) and `Basílica_Colegiata_de_Nuestra_Señora_de_Guanajuato` (#17), inside VGGT's forward pass.
    Not caused by the scenes: VGGT pads every image in a batch to the largest one, so every 50-view
    scene here is 50 × 518×518 (~69k tokens), the same as scenes that passed. Likely cause: a SAM 3
    mask job started in the other worktree (`ArchWorld-predict-ita-from-pass-1-signal`) and was using
    the same GPU at that time. **Do not run GPU jobs concurrently with this run.** Failed scenes can
    be retried by re-running the script (it skips finished scenes).

## Run 2b — base through `run.py` at η=0 (branch `eric/vggt-base-eta0`, 2026-11-15)

**Why.** Swati lined the `evaluate.py` base up with her η=1 numbers: CD ~4.4× higher at the median
(Mount_Rushmore 38×), scene ranks only loosely agree (Spearman 0.45). The two pipelines score
differently, so the `evaluate.py` CSV cannot be the "before" for ArchAttend. It stays as a separate
result ("VGGT on the benchmark, dev subset"). Differences, largest first:
- `evaluate.py` was run without `--filter_pred`, so predictions were not masked with
  `manual_filtration_masks` (on Baguio_Cathedral, filtering alone: CD_norm 110.1 → 4.43);
- no ICP in `evaluate.py` (`run.py` sets `USE_ICP=1`);
- precision: that run was bf16 (`VGGT_HALF=1` on the L4); swati's η runs are fp32.

**What.** Untouched VGGT through swati's exact pipeline: `run.py` at η=0. In VGGT's `loss.py`
the step is `x -= η·g = x`, attention cannot rise, so the step is discarded and the original tokens
are kept (log: Δ0.000000 at all 24 layers). Same scoring code as her sweep: ICP + filtration masks.

```bash
conda activate archworld
unset VGGT_HALF                       # fp32, like swati
export SKIP_BASELINE_PASS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
python intervention/run.py --model vggt --root data \
  --masks-dir /home/eric/ArchWorld-predict-ita-from-pass-1-signal/pass1_out/masks \
  --scenes-file scenes_dev.txt --eta 0 --out results/vggt_dev
python eric/collect_eta0.py           # -> eric/vggt_base_eta0_dev_results.csv
```

- **Masks:** reused from the Pass 1 worktree (`make_masks_full.py`), not regenerated. It calls the
  repo's `make_masks` with the same SAM 3 threshold (0.5) and layout, just more elements, so the anchor
  masks are the ones `make_masks.py` makes. At η=0 the masks cannot change the prediction anyway.
- **Environment fix:** `open3d` (ICP) needed the system library `libusb-1.0`. Without it the eval step
  failed with `Inter eval: failed (libusb-1.0.so.0 ...)` but `run.py` still exited 0 and printed "Done."
  — no `eval_metrics_*.json` was written. Check the JSON count, not the exit code.
- **Detached run:** `results/vggt_dev_eta0_logs/run_all.sh`, started with `setsid nohup` at 05:53 UTC so
  it survives closing the session/VSCode (not stopping the VM). Resumable: scenes with
  `eval_metrics_0_23.json` are skipped. `collect_when_done.sh` (also detached) runs `collect_eta0.py`
  when the run writes `DONE`, output in `results/vggt_dev_eta0_logs/summary.txt`.

Progress:
```bash
ls results/vggt_dev/output_0/*/eval_metrics_0_23.json | wc -l          # of 39
grep -E "^\[eta=0\]|Inter eval" results/vggt_dev_eta0_logs/run_all.log | tail -4
cat results/vggt_dev_eta0_logs/summary.txt                              # after DONE
```

**Outcomes.**
- Dendera smoke test (fp32, L4, ~5–6 min, fits in memory): `Intervention token mask: 22821`,
  filtration masks on 50/50 views.

  | Dendera | Acc-mean | Acc-med | Comp-mean | Comp-med | CD | CD_norm |
  |---|---|---|---|---|---|---|
  | η=0, run.py (this run) | 0.0676 | 0.0217 | 0.0096 | 0.0052 | 0.0386 | 8.43 |
  | swati η=1 | 0.0680 | 0.0208 | 0.0085 | 0.0049 | 0.0382 | 8.35 |
  | swati best η (100) | | | | | 0.0382 | 8.34 |
  | evaluate.py base (bf16, unfiltered) | 0.881 | 0.135 | 0.019 | 0.010 | 0.450 | 98.34 |

  η=0 is within 1% of swati's η=1; her 5 η values span ~6% on this scene (incl. unseeded RANSAC
  noise). Pipeline change alone: ~12× lower CD than `evaluate.py`.
- **All 39 dev scenes: 39 / 39 scored** (05:55 → 11:42 UTC, ~9 min/scene, fp32 fits on the L4;
  no eval failures, no out-of-memory, no zero token masks). Per-scene numbers:
  **`eric/vggt_base_eta0_dev_results.csv`** (η=0 Acc/Comp mean & median, CD, CD_m, CD_norm, next to
  swati's η=1 and best-η CD_norm, her best η, and the `evaluate.py` CD_norm).

  | median over 39 dev scenes | CD_norm |
  |---|---|
  | **η=0 base, run.py (this run)** | **3.89** |
  | swati η=1 | 3.85 |
  | swati best η per scene (oracle) | 3.63 |
  | `evaluate.py` base (bf16, unfiltered) | 20.13 |

  Means (outliers weigh in): η=0 7.47, oracle 7.09.
  - **η=0 matches swati's pipeline.** Per scene, swati-η=1 / η=0 has median 0.998 (IQR 0.98–1.01);
    Spearman rank correlation 0.997 (vs 0.63 for `evaluate.py`). Largest gaps: Aguinaldo_Shrine 0.80,
    Shuri_Castle 1.29 — η=1 is either noise (unseeded RANSAC) or a real small nudge on those scenes.
  - **This is the "before" for ArchAttend.** Oracle best η vs η=0: median improvement **3.0%**
    (mean 5.6%); >5% on 16 scenes; worse than η=0 on 4 (best η is the min over 5 runs, so on those
    every η hurt). Biggest gains: Isa_Khan_Niyazi's_tomb 42.6% (η=1000), Aguinaldo_Shrine 27.0%,
    Jama_Masjid,_Delhi 20.9%, Cathedral_of_Holy_Wisdom,_Harbin 16.3% (η=5000),
    Disney_California_Adventure_Park 13.2% (η=100).
  - **Caveat on the oracle gain:** "best" is the minimum over 5 noisy runs, so part of the 3% is
    selection on noise (η=0 vs η=1 alone varies ±2% at the IQR). A fair estimate needs repeated η=0
    runs or seeded RANSAC.
  - Best η per scene: 5000 ×16, 100 ×7, 1000 ×7, 10 ×5, 1 ×4 — scene-dependent, as `AGENTS.md` says.
  - By region (median CD_norm, n): Latin America 1.87 (5), West/Central Asia 1.94 (5), Europe 3.07 (5),
    SE Asia 3.89 (5), South Asia 5.27 (6), North America 6.31 (5), East Asia 11.02 (4), Africa 20.19 (4).
    n is 4–6 per region, so these are noisy.
  - The `evaluate.py` base is 1.1×–40× higher per scene (median 4.0×), consistent with swati's 4.4×.

## Constraints and caveats

- **GPU: now NVIDIA L4, 24 GB** (was Tesla V100, 16 GB). On the V100, 50 views did not fit for
  VGGT-1B even with `VGGT_HALF=1`; only 43-view scenes did. Likely cause (not verified): VGGT keeps
  all 24 layer outputs in memory during inference (VGGT's README: aggregator alone is 11.4 GB at
  50 frames on an H100). The L4 fits 50 views for base evaluation; whether it fits *every* dev scene
  is what the current run shows. ArchAttend needs more memory than base and may not fit on 24 GB.
- **CPU-bound: 4 vCPUs (`g2-standard-4`).** The GPU only runs the VGGT forward pass (seconds); GT
  point generation, RANSAC alignment (numpy, 1000 iterations; evaluate.py has no ICP), sky
  segmentation (onnxruntime on CPU), GLB export (trimesh) and Chamfer (scipy KD-tree) run on the CPU, which sits at ~100% on all 4 cores. ~3–5 min per scene, up to ~22 min for heavy
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
  On the L4, 50-view scenes fit when the GPU is not shared, so the subset bias should be small; any
  scene that still fails after a retry on an idle GPU needs its region / type checked.
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
