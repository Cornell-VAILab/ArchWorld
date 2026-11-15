# Inference-time eta selection for ArchAttend (VGGT)

Goal: pick ArchAttend's step size eta per scene at inference time, by optimizing an
unsupervised objective computed from the model's own output (no ground truth).

## Setup used
- Scenes: 80 downloaded, split 40 dev / 40 test (`scenes_dev.txt`, `scenes_test.txt`; one dev scene,
  Bridge_of_San_Martín, is not in the benchmark views and was dropped -> 39 dev).
- Data fix: `python tools/flatten_scene_images.py --root data` (images live in category subfolders).
- Sweep: `swati/run_sweep.sh` (VGGT, layers 0-23, eta in {1,10,100,1000,5000}, SAVE_RAW_TENSORS=1,
  SKIP_BASELINE_PASS=1 because base+intervention does not fit in 24 GB at 50 views).
- Scores per scene and eta: `swati/results_vggt_dev/eta_*/<scene>.json`, table in `per_scene.csv`.

## Scripts
| script | what it does |
|---|---|
| `proxy_select.py` | first proxies (confidence, multi-view consistency) vs fixed eta and oracle |
| `picker.py` | computes per-(scene, eta) features incl. depth/rotation drift vs eta=1; writes `picker_features.csv` |
| `analyze_picker.py` | compares pickers (leave-one-out tau) with bootstrap CI and Wilcoxon test |
| `pick_eta.py` | the locked rule, to apply to new scenes (test set) |

## Dev results (39 scenes)
Regret = CD(chosen eta) / CD(best eta for the scene); 1.00 = perfect.

| picker | median | mean | worst | >1.2 |
|---|---|---|---|---|
| fixed eta=1000 (best single eta) | 1.019 | 1.044 | 1.22 | 1 |
| fixed eta=5000 | 1.014 | 1.389 | 6.25 | 7 |
| confidence argmax | 1.016 | 1.076 | 2.14 | 4 |
| **guarded-5000 (depth drift, LOO tau)** | **1.011** | **1.037** | 1.22 | 1 |

guarded-5000 vs fixed 1000: 10 wins / 27 ties / 2 losses, Wilcoxon p = 0.064, bootstrap 95% CI of the
mean regret difference [-0.018, +0.004]; closes 15% of the gap to the oracle. Depth drift at eta=5000
correlates with eta=5000's regret (Spearman +0.58), i.e. it flags the runs where the large step breaks
the reconstruction.

## Locked rule (for the test set)
pick eta=5000 if median relative depth change(eta=5000 vs eta=1) <= 0.00626, else eta=1000.
Needs three runs per scene (eta = 1, 1000, 5000).
