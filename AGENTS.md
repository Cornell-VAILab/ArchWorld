# Working in this repository

Task-oriented notes. Each section is one thing you might be asked to do, what to
run, and what to check afterwards.

Everything assumes the environment is built and the dataset extracted:

```bash
bash setup/install.sh && conda activate archworld
export ARCHWORLD_ROOT=data        # or pass --root data everywhere
```

## Benchmark a model

```bash
python eval/evaluate.py --model vggt --help
```

This reconstructs each scene, aligns the prediction to the metric ground truth,
and scores it. It needs nothing but the dataset.

## Run ArchAttend on a scene

Two steps. The first is one-off per scene; the second is what you iterate on.

```bash
python intervention/make_masks.py --root data --out masks --scene Akdamar_Church
python intervention/run.py --model da3 --root data --masks-dir masks \
       --scene Akdamar_Church --eta 1
```

Results land in `results/output_<eta>/<scene>/eval_metrics_<first>_<last>.json`,
with a `base` block (the model untouched) and an `inter` block (with ArchAttend).

**Check the log for `Intervention token mask: N` before believing the result.**
If N is 0 the masks did not match the views and the intervention did nothing —
the run still "succeeds" and still writes plausible metrics. N in the low
thousands is normal.

Masks must sit at `<masks-dir>/<scene>/mask/<view stem>_<element>.npy`, keyed by
the view's original name. `make_masks.py` writes them there; if you bring your
own, match that layout.

## Reproduce a table from the paper

```bash
python intervention/run.py --model da3 --root data --masks-dir masks   # all five eta
python intervention/aggregate.py results --model da3 --root data --metric CD_norm
```

The sweep matters. `aggregate.py` reports *before* as the baseline averaged over
the sweep and *after* as the best step size per scene, because the best eta is
scene-dependent. A single eta is a smoke test, not a result — for VGGT and π³ in
particular, η=1 barely moves the prediction, and reading an effect out of one
step size will mislead you.

`--metric` selects `CD_norm` (the paper's), `CD_m` (metres), or `CD` (raw).

## Steer a different element

`intervention/assets/anchors.json` names one architectural element per scene.
That file is the result of Pass 1, which is why running this code performs Pass 2
only. To steer something else, change the element named there and regenerate that
scene's masks:

```bash
python intervention/make_masks.py --root data --out masks --scene <scene> --overwrite
```

To choose it the way the method does rather than by hand, segment the full
vocabulary (`--all-elements`), rank the elements by Semantic Cross-View
Attention, and write the maximum into `anchors.json`.

## Run on a scene outside the benchmark

Three files decide which scenes exist and how they are read:

| file | holds |
| --- | --- |
| `assets/views.json` | the views to evaluate, per scene, **in order** |
| `assets/anchors.json` | the element to steer, per scene |
| `assets/elements.json` | the vocabulary to segment, per scene |

Add your scene to all three. View order is not cosmetic: view 0 is the reference
frame for both the model and the ground truth, so reordering changes the result.
The scene also needs `archworld/<scene>/{images,depth,sparse}` under `--root`,
with a COLMAP reconstruction whose image names match the view files.

## Add a fourth model

`intervention/run.py` holds a table mapping each model to its pass module and
global-layer range; add an entry, and add `intervention/models/<name>/` with a
`run_pass.py` exposing `main(scene_name, work_dir, model)` and a `loss.py` with
the attention hook.

Copy the closest existing pass rather than writing one from scratch. The three
differ inside the attention objective in ways that are deliberate, not
accidental: VGGT reads the qkv weights directly and so skips the learned q/k
norms and rotary embeddings, averages the objective over batch and heads, and
discards a layer's step if attention did not rise; DA3 and π³ apply the norms and
embeddings, sum over batch and heads, and keep every step; π³ additionally scales
the queries before an attention call that scales them again. Each model's eta
sweep was run against its own convention, so a "cleanup" that unifies them
changes what eta means for that model.

## Modifying the passes

`intervention/models/*/run_pass.py` and `loss.py` are the code that produced the
published numbers. Treat them as a reference implementation. They carry branches
reachable only through environment variables this repository never sets; those
are inert, and removing them means re-proving that nothing else depended on them.

Three things that will bite you if you refactor:

- Imports inside a pass resolve against the repository root, `eval/`, and the
  pass's own directory, all placed on `sys.path` by the header at the top of each
  file. The passes import `loss` flatly and `eval/` modules import each other
  flatly, so dropping any of the three breaks imports.
- Do not add `__init__.py` under `intervention/models/<name>/`. It makes the
  directory importable as a top-level `vggt` or `pi3` and shadows the real
  package.
- The vggt repository ships its own `visual_util.py`. Import ours as
  `eval.visual_util`, never flatly.

## Checking a change

Importing each pass catches most breakage and takes a minute:

```bash
for m in vggt da3 pi3; do
  python -c "
import sys; sys.path[:0] = ['intervention', '.']
import importlib; importlib.import_module('models.$m.run_pass')" && echo "$m ok"
done
```

If you touched anything on the metric path, re-derive a table from an existing
results directory and compare against the paper before trusting it.
