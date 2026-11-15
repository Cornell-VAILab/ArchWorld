#!/usr/bin/env python
"""Collect the eta=0 VGGT base run (intervention/run.py) into one CSV, next to
swati's ArchAttend sweep and the earlier evaluate.py base, scene by scene.

    python eric/collect_eta0.py            # after the run; safe on partial results

Reads   results/vggt_dev/output_0/<scene>/eval_metrics_0_23.json   ("inter" block;
        at eta=0 the step is zero, so this is untouched VGGT)
        swati/results_vggt_dev/per_scene.csv                       (her eta sweep)
        eric/vggt_base_dev_results.csv                             (evaluate.py base)
Writes  eric/vggt_base_eta0_dev_results.csv

CD = (Acc-mean + Comp-mean)/2; CD_m = CD x Scale; CD_norm = CD x Scale x 1000 / Extent,
exactly as intervention/aggregate.py (its chamfer() and metadata() are reused).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "intervention"))
from aggregate import chamfer, metadata  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--results", type=Path, default=REPO / "results/vggt_dev/output_0")
    ap.add_argument("--scenes", type=Path, default=REPO / "scenes_dev.txt")
    ap.add_argument("--out", type=Path, default=REPO / "eric/vggt_base_eta0_dev_results.csv")
    args = ap.parse_args()

    meta = metadata(args.root)
    scenes = [l.strip() for l in args.scenes.read_text().splitlines() if l.strip()]
    sw = pd.read_csv(REPO / "swati/results_vggt_dev/per_scene.csv")
    sw["cd"] = 0.5 * (sw["Acc-mean"] + sw["Comp-mean"])
    ev = pd.read_csv(REPO / "eric/vggt_base_dev_results.csv").set_index("scene")

    rows = []
    for s in scenes:
        m = meta.get(s, {})
        scale, extent = m.get("scale"), m.get("extent_m")
        norm = (lambda cd: cd * scale * 1000.0 / extent) if scale and extent else (lambda cd: None)
        r = {"scene": s, "region": m.get("region"), "arch_type": m.get("arch_type")}
        f = args.results / s / "eval_metrics_0_23.json"
        e = json.loads(f.read_text()).get("inter") if f.exists() else None
        cd = chamfer(e)
        r["status"] = "ok" if cd is not None else "missing"
        if cd is not None:
            r.update(acc_mean=e["Acc-mean"], acc_med=e["Acc-med"], comp_mean=e["Comp-mean"],
                     comp_med=e["Comp-med"], cd=cd, cd_m=cd * scale if scale else None,
                     cd_norm=norm(cd))
        ss = sw[sw.scene == s]
        if len(ss):
            e1 = ss[ss.eta == 1].cd
            r["swati_cd_norm_eta1"] = norm(float(e1.iloc[0])) if len(e1) else None
            best = ss.loc[ss.cd.idxmin()]
            r["swati_cd_norm_best"] = norm(float(best.cd))
            r["swati_best_eta"] = int(best.eta)
        if s in ev.index:
            r["evalpy_cd_norm"] = ev.loc[s, "cd_norm"]
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, float_format="%.6g")
    ok = df[df.status == "ok"]
    print(f"{len(ok)}/{len(df)} scenes scored -> {args.out}")
    if len(ok):
        cols = ["cd_norm", "swati_cd_norm_eta1", "swati_cd_norm_best", "evalpy_cd_norm"]
        print("median CD_norm over scored scenes:")
        print(ok[cols].median().round(3).to_string())
        both = ok.dropna(subset=["cd_norm", "swati_cd_norm_best"])
        gain = (both.cd_norm - both.swati_cd_norm_best) / both.cd_norm
        print(f"oracle-eta improvement over eta=0 (median, n={len(both)}): {100 * gain.median():.1f}%")
    missing = df[df.status != "ok"].scene.tolist()
    if missing:
        print("missing:", missing)


if __name__ == "__main__":
    main()
