#!/usr/bin/env python
"""Turn a directory of ArchAttend runs into the numbers the paper reports.

    python intervention/aggregate.py results --model da3 --root data

`run.py` writes one metrics file per scene and step size. This reads them all,
applies the paper's conventions, and prints the overall table plus the regional
and architectural breakdowns.

  Before   the baseline, averaged over the step-size sweep. Each pass re-runs
           the baseline, and those runs differ only by the noise of the
           unseeded RANSAC, so averaging removes it rather than crediting it
           to the intervention.
  After    the best score over the sweep, per scene, since the best step size
           is scene-dependent.
  Reported the median across scenes, and within each bucket.

Chamfer distance is reported three ways: raw, in metres (x the scene's metric
scale), and extent-normalized (CD_norm), which is what the paper uses because
it makes a cathedral and a shrine comparable.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = {"vggt": (0, 23), "da3": (9, 23), "pi3": (1, 35)}
EXCLUDE = {"Generalife"}   # dropped from the published tables


def chamfer(entry):
    """CD = (accuracy + completion) / 2, both mean-based, as in the paper."""
    if not entry:
        return None
    acc = entry.get("Acc-mean", entry.get("Acc_mean"))
    comp = entry.get("Comp-mean", entry.get("Comp_mean"))
    if acc is None or comp is None:
        return None
    return 0.5 * (float(acc) + float(comp))


def load(results, model, meta):
    first, last = LAYERS[model]
    name = f"eval_metrics_{first}_{last}.json"
    per_scene = {}
    for step_dir in sorted(Path(results).glob("output_*")):
        for f in sorted(step_dir.glob(f"*/{name}")):
            scene = f.parent.name
            if scene in EXCLUDE:
                continue
            try:
                d = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            base, inter = chamfer(d.get("base")), chamfer(d.get("inter"))
            rec = per_scene.setdefault(scene, {"base": [], "inter": []})
            if base is not None:
                rec["base"].append(base)
            if inter is not None:
                rec["inter"].append(inter)

    rows = []
    for scene, rec in per_scene.items():
        if not rec["base"] or not rec["inter"]:
            continue
        m = meta.get(scene, {})
        scale, extent = m.get("scale"), m.get("extent_m")
        before, after = float(np.mean(rec["base"])), float(min(rec["inter"]))
        row = {"scene": scene, "region": m.get("region"), "arch_type": m.get("arch_type"),
               "n_eta": len(rec["inter"]), "CD_before": before, "CD_after": after}
        if scale:
            row["CD_m_before"], row["CD_m_after"] = before * scale, after * scale
            if extent:
                row["CD_norm_before"] = before * scale * 1000.0 / extent
                row["CD_norm_after"] = after * scale * 1000.0 / extent
        rows.append(row)
    return pd.DataFrame(rows)


def metadata(root):
    csv = Path(root) / "ArchWorld_metadata.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    out = {}
    for r in df.itertuples():
        extent, scale = getattr(r, "Extent", None), getattr(r, "Scale", None)
        out[r.scene] = {
            "region": r.Region if isinstance(r.Region, str) else None,
            "arch_type": getattr(r, "_3", None) if isinstance(getattr(r, "_3", None), str) else None,
            "scale": float(scale) if pd.notna(scale) else None,
            "extent_m": float(extent) if pd.notna(extent) and float(extent) > 0 else None,
        }
    return out


def panel(df, key, b, a):
    rows = []
    for label, sub in df[df[key].notna()].groupby(key):
        rows.append({"bucket": label, "n": len(sub),
                     "before": sub[b].median(), "after": sub[a].median()})
    out = pd.DataFrame(rows).sort_values("before", ascending=False).reset_index(drop=True)
    out["gain %"] = (out.after - out.before) / out.before * 100
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--model", required=True, choices=sorted(LAYERS))
    ap.add_argument("--root", default="data", help="dataset root, for the metadata")
    ap.add_argument("--metric", default="CD_norm", choices=["CD_norm", "CD_m", "CD"])
    ap.add_argument("--csv", type=Path, help="also write the per-scene table here")
    args = ap.parse_args()

    meta = metadata(args.root)
    if not meta:
        print(f"warning: no ArchWorld_metadata.csv under {args.root}; "
              f"only raw CD is available")
    df = load(args.results, args.model, meta)
    if df.empty:
        raise SystemExit(f"no complete scenes in {args.results}")

    b, a = f"{args.metric}_before", f"{args.metric}_after"
    if b not in df.columns:
        raise SystemExit(f"{args.metric} needs the dataset metadata; pass --root, "
                         f"or use --metric CD")
    df = df[df[b].notna() & df[a].notna()]

    mb, ma = df[b].median(), df[a].median()
    print(f"\n{args.model}  {args.metric}  over {len(df)} scenes "
          f"({df.n_eta.min()}-{df.n_eta.max()} step sizes each)")
    print(f"  before {mb:9.2f}")
    print(f"  after  {ma:9.2f}   ({(ma - mb) / mb * 100:+.1f}%, "
          f"improved on {float((df[a] < df[b]).mean()) * 100:.1f}% of scenes)")

    for key in ("region", "arch_type"):
        if df[key].notna().sum() == 0:
            continue
        p = panel(df, key, b, a)
        print(f"\nby {key}")
        print(p.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        sb, sa = p.before.max() - p.before.min(), p.after.max() - p.after.min()
        # A single bucket has no spread to report, and dividing by it would fail.
        pct = f" ({(sa - sb) / sb * 100:+.1f}%)" if sb else ""
        print(f"  worst-to-best spread  {sb:.2f} -> {sa:.2f}{pct}")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
