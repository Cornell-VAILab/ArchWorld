<p align="center">
  <img src="https://cornell-vailab.github.io/ArchWorld/static/images/teaser.png" width="100%" alt="A sample scene from each of ArchWorld's eight geographic regions">
</p>

<h1 align="center">Towards Reconstructing Geographically Diverse Architecture with 3D Foundation Models</h1>

<p align="center">
  Nicky Kriplani*, Yiwen Zhang*, Angelina Wang&dagger;, Hadar Averbuch-Elor&dagger;<br>
  Cornell Tech, Cornell University &middot; NeurIPS 2026<br>
  <sub>*Equal Contribution &nbsp;&dagger;Equal Advising</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-coming%20soon-B31B1B" alt="arXiv"></a>
  <a href="https://cornell-vailab.github.io/ArchWorld/"><img src="https://img.shields.io/badge/Project-Page-487F69" alt="Project page"></a>
  <a href="https://huggingface.co/datasets/ArchWorld/ArchWorld"><img src="https://img.shields.io/badge/Dataset-ArchWorld-FFD21E" alt="Dataset"></a>
</p>

**ArchWorld** is a geographically diverse benchmark of 396 in-the-wild 3D landmarks across eight geographic regions, with rich metadata and metric-scale dense ground-truth reconstructions, for evaluating 3D foundation models disaggregated by region and architecture type. **ArchAttend** is an inference-time intervention that steers a model's cross-view attention onto the architecture, improving reconstruction while narrowing those gaps.

This repository has two parts: `eval/` benchmarks a model on ArchWorld, and `intervention/` runs ArchAttend on top of it. See the [project page](https://cornell-vailab.github.io/ArchWorld/) for the method and results.

## Setup

```bash
git clone https://github.com/Cornell-VAILab/ArchWorld.git
cd ArchWorld
```

One environment runs all three models.

```bash
bash setup/install.sh
conda activate archworld
```

That creates the environment, clones VGGT, Depth Anything 3 and π³ from their own repositories at the commits this code was developed against, and installs them alongside the evaluation dependencies. Checkpoints download on first run.

Each model's own requirements pin a different torch, so all three are installed with `--no-deps`; the shared versions the script installs (torch 2.3.1, numpy 2.2.6) are what the published numbers were produced with.

Running ArchAttend also needs SAM 3, to segment the architectural elements it steers. SAM 3 wants a newer torch and an older numpy than the evaluation does, so it gets its own environment — it is only needed once, to make the masks:

```bash
bash setup/sam3.sh
conda activate archworld-sam3
```

SAM 3's weights are gated: request access at [facebook/sam3](https://huggingface.co/facebook/sam3), then `hf auth login`.

### Data

Download the dataset and extract it into one directory. Scenes are separate tarballs, so you can start with a handful.

```bash
huggingface-cli download ArchWorld/ArchWorld --repo-type dataset --local-dir data
cd data && for f in data/*.tar.gz; do tar xzf "$f"; done && cd ..
```

That gives the layout every entry point expects:

```
data/
  ArchWorld_metadata.csv                          region, architecture type, metric scale, extent
  archworld/<scene>/images/<view>.jpg             input views
  archworld/<scene>/depth/<view>.jpg.npy          MVS depth maps
  archworld/<scene>/sparse/                       COLMAP reconstruction
  manual_filtration_masks/<scene>/<view>.jpg.npy  landmark masks
```

Pass it as `--root data`, or export `ARCHWORLD_ROOT=data`.

## Evaluation

`eval/evaluate.py` measures a model on the benchmark: it reconstructs each scene, aligns the prediction to the metric ground truth, and scores it.

```bash
python eval/evaluate.py --model vggt --help     # its options
```

## ArchAttend

ArchAttend is a two-pass method. **Pass 1** picks the anchor: of the architectural elements annotated in a scene, the one whose patches draw the most Semantic Cross-View Attention from the other views. **Pass 2** runs the model again with a hook on every global attention block, taking one gradient-ascent step on that block's input token representation in the direction that increases attention to the anchor:

```
T^l  ←  T^l + η · ∇_{T^l} S^l(e*)
```

Model weights stay frozen; only the latent is nudged. The best `η` is scene-dependent, so the sweep covers `{1, 10, 100, 1000, 5000}` and `aggregate.py` takes the per-scene best. A single scene-agnostic `η` recovers much of the gain without needing ground truth to choose it.

### 1. Element masks

We ship the result of Pass 1, so Pass 1 is not run: `intervention/assets/anchors.json` names the anchor element for every scene, shared by all three models. Pass 2 steers that element, which requires its mask in each view. Masks are segmented with [SAM 3](https://github.com/facebookresearch/sam3):

```bash
python intervention/make_masks.py --root data --out masks
```

This segments the anchor element only, and writes `masks/<scene>/mask/<view>_<element>.npy`, the layout `--masks-dir` expects. Existing files are skipped, so the command can be resumed.

To determine the anchor yourself rather than using the one we ship, run Pass 1: segment the full element vocabulary with `--all-elements`, rank the elements by Semantic Cross-View Attention, and write your choice into `anchors.json`. That is roughly twenty times the segmentation work.

### 2. Run

```bash
python intervention/run.py --model vggt --root data --masks-dir masks
python intervention/run.py --model da3  --root data --masks-dir masks
python intervention/run.py --model pi3  --root data --masks-dir masks

python intervention/run.py --model da3  --root data --masks-dir masks --scene Akdamar_Church
```

Each run writes `results/output_<η>/<scene>/eval_metrics_<first>_<last>.json`, holding `base` (the model untouched) and `inter` (with ArchAttend), so one invocation gives you both arms.

DA3 and π³ build their evaluation cloud with sky pixels removed, which is how the published numbers were produced. `run.py` fetches the segmentation model on first use; to place it yourself instead, download [skyseg.onnx](https://huggingface.co/JianyuanWang/skyseg) and point `--skyseg` at it. Pass `--no-sky-filter` to skip sky removal entirely, at the cost of not matching the paper.

| Flag | Purpose |
| --- | --- |
| `--scene` / `--scenes-file` | run a subset instead of every extracted scene |
| `--eta` | step sizes to sweep |
| `--no-icp` | skip ICP refinement during alignment |
| `--skyseg` | use an existing `skyseg.onnx` instead of downloading one |
| `--no-sky-filter` | evaluate DA3 and π³ without sky removal |

### 3. Aggregate

```bash
python intervention/aggregate.py results --model da3 --root data
```

Prints the overall table and the regional and architectural breakdowns, under the paper's conventions: *before* is the baseline averaged over the sweep, *after* is the per-scene best, and both are reported as medians across scenes.

```
da3  CD_norm  over 378 scenes
  before     10.36
  after       8.92   (-13.9%, improved on 79.6% of scenes)

by region
           bucket  n  before  after  gain %
    North America 44   18.83  12.82  -31.93
        East Asia 43   15.52  13.54  -12.76
           ...
  worst-to-best spread  11.26 -> 6.96 (-38.2%)
```

Add `--metric CD_m` for metres or `--metric CD` for raw Chamfer distance, and `--csv` to write the per-scene table.

## License

Code is released under the license in [LICENSE](LICENSE). The dataset is distributed separately under the terms on its [dataset page](https://huggingface.co/datasets/ArchWorld/ArchWorld). The evaluated models and SAM 3 keep their own licenses.

## Citation

```bibtex
@inproceedings{kriplani2026archworld,
  title     = {Towards Reconstructing Geographically Diverse Architecture with 3D Foundation Models},
  author    = {Kriplani, Nicky and Zhang, Yiwen and Wang, Angelina and Averbuch-Elor, Hadar},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
