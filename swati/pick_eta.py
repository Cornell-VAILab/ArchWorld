#!/usr/bin/env python
"""Locked inference-time eta picker for ArchAttend (VGGT): guarded-5000 with a depth-drift guard.

Rule (fixed on the 39 dev scenes, not to be re-tuned on test):
    drift = median relative depth change between the eta=5000 and eta=1 predictions
    pick eta=5000 if drift <= TAU, else eta=1000
TAU = 0.00626 (median of the leave-one-out choices on dev).

Needs, per scene, output_1/, output_1000/, output_5000/ with inter_raw.npz (SAVE_RAW_TENSORS=1).
No ground truth is used to pick. If eval_metrics files exist, the pick is also scored.

    python swati/pick_eta.py results/vggt_test --scenes scenes_test.txt --csv swati/results_vggt_test/picks.csv
"""
import argparse, json
from pathlib import Path
import numpy as np

TAU = 0.00626
SAFE, BOLD, REF = 1000, 5000, 1


def depth(npz):
    z = np.load(npz)["depth_map"]
    S = z.shape[1]
    return z.astype(np.float32).reshape(S, *z.shape[2:4])


def drift(d, d0):
    ok = (d0 > 1e-4) & (d > 1e-4)
    return float(np.median(np.abs(d[ok] - d0[ok]) / d0[ok])) if ok.any() else float("inf")


def cd(p):
    if not p.exists():
        return None
    m = json.loads(p.read_text()).get("inter") or {}
    return 0.5 * (m["Acc-mean"] + m["Comp-mean"]) if "Acc-mean" in m and "Comp-mean" in m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--scenes", type=Path, help="scene list; default = every scene with all three runs")
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    run = lambda e, s: a.results / f"output_{e}" / s
    scenes = ([l.strip() for l in a.scenes.read_text().splitlines() if l.strip()] if a.scenes
              else sorted(p.name for p in (a.results / f"output_{REF}").iterdir()))
    out = []
    for s in scenes:
        if not all((run(e, s) / "inter_raw.npz").exists() for e in (REF, SAFE, BOLD)):
            print(f"  {s}: missing runs, skipped"); continue
        dr = drift(depth(run(BOLD, s) / "inter_raw.npz"), depth(run(REF, s) / "inter_raw.npz"))
        pick = BOLD if dr <= a.tau else SAFE
        cds = {e: cd(run(e, s) / "eval_metrics_0_23.json") for e in (SAFE, BOLD)}
        out.append((s, dr, pick, cds[SAFE], cds[BOLD]))
        print(f"  {s[:40]:<40} drift {dr:.4f} -> eta={pick}")

    scored = [o for o in out if o[3] is not None and o[4] is not None]
    print(f"\n{len(out)} scenes picked ({sum(o[2] == BOLD for o in out)} x eta={BOLD}), tau={a.tau}")
    if scored:
        pick_cd = np.array([o[3] if o[2] == SAFE else o[4] for o in scored])
        safe_cd = np.array([o[3] for o in scored]); bold_cd = np.array([o[4] for o in scored])
        print(f"mean CD   picker {pick_cd.mean():.4f} | fixed {SAFE} {safe_cd.mean():.4f} | fixed {BOLD} {bold_cd.mean():.4f}")
        d = pick_cd / safe_cd - 1
        print(f"vs fixed {SAFE}: wins {int(np.sum(d < -1e-12))} ties {int(np.sum(np.abs(d) <= 1e-12))} "
              f"losses {int(np.sum(d > 1e-12))}, mean rel. change {100 * d.mean():+.2f}%")
    if a.csv:
        import csv
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["scene", "drift", "pick", f"cd_{SAFE}", f"cd_{BOLD}"]); w.writerows(out)
        print(f"-> {a.csv}")


if __name__ == "__main__":
    main()
