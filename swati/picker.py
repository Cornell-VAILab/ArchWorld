#!/usr/bin/env python
"""Unsupervised per-scene eta picker with a "do no harm" drift guard.

    python swati/picker.py results/vggt_dev --csv swati/results_vggt_dev/picker_features.csv

For every (scene, eta) it computes, from the model's own outputs only (no GT):
  conf   mean log depth-confidence                               (higher = better)
  mvc    multi-view depth consistency, median relative error      (lower = better)
  ddrift median relative depth change vs the eta=1 prediction     (how far ArchAttend moved it)
  rdrift mean camera rotation change vs eta=1, degrees

Pickers compared (GT only used to score the pick):
  fixed-eta     the same eta for every scene
  conf / mvc    argmax of the proxy
  guarded-conf  start at a safe default eta; take the conf-argmax eta only if its drift <= tau
  guarded-5000  use 5000 if its drift <= tau, else the default
tau is tuned by leave-one-out: for each scene, tau is picked on the other scenes only.

Regret = CD(pick) / CD(best eta for that scene), CD = 0.5*(Acc-mean + Comp-mean). 1.00 = perfect.
"""
import argparse, json
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------- loading
def load_pred(npz):
    z = np.load(npz)
    S = z["depth_map"].shape[1]
    d = z["depth_map"].astype(np.float32).reshape(S, *z["depth_map"].shape[2:4])
    c = z["depth_conf"].astype(np.float32).reshape(S, *z["depth_conf"].shape[2:4])
    E = z["extrinsics"].reshape(-1, 3, 4).astype(np.float64)
    K = z["intrinsics"].reshape(-1, 3, 3).astype(np.float64)
    return d, c, E, K


def cd_of(metrics_path):
    m = json.loads(Path(metrics_path).read_text()).get("inter") or {}
    if "Acc-mean" not in m or "Comp-mean" not in m:
        return None
    return 0.5 * (float(m["Acc-mean"]) + float(m["Comp-mean"]))


# ---------------------------------------------------------------- features
def f_conf(d, c, E, K):
    return float(np.mean(np.log(np.clip(c, 1e-6, None))))


def f_mvc(d, c, E, K, n_pairs=60, n_pix=4000, seed=0):
    rng = np.random.default_rng(seed)
    S, H, W = d.shape
    errs = []
    for _ in range(n_pairs):
        i, j = rng.choice(S, 2, replace=False)
        v = rng.integers(0, H, n_pix); u = rng.integers(0, W, n_pix)
        z = d[i, v, u]; ok = z > 1e-4
        v, u, z = v[ok], u[ok], z[ok]
        pix = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], 0)
        Xc = np.linalg.inv(K[i]) @ pix * z
        Xw = E[i, :, :3].T @ (Xc - E[i, :, 3:])
        Xj = E[j, :, :3] @ Xw + E[j, :, 3:]
        zj = Xj[2]; front = zj > 1e-4
        uvj = K[j] @ Xj
        uj = uvj[0] / np.where(front, zj, 1) - 0.5
        vj = uvj[1] / np.where(front, zj, 1) - 0.5
        inb = front & (uj >= 0) & (uj <= W - 1) & (vj >= 0) & (vj <= H - 1)
        if inb.sum() < 50:
            continue
        dj = d[j, np.round(vj[inb]).astype(int), np.round(uj[inb]).astype(int)]
        errs.append(np.median(np.abs(zj[inb] - dj) / np.maximum(dj, 1e-4)))
    return float(np.median(errs)) if errs else float("nan")


def f_ddrift(pred, ref):
    d, d0 = pred[0], ref[0]
    ok = (d0 > 1e-4) & (d > 1e-4)
    return float(np.median(np.abs(d[ok] - d0[ok]) / d0[ok])) if ok.any() else float("nan")


def f_rdrift(pred, ref):
    R, R0 = pred[2][:, :, :3], ref[2][:, :, :3]
    ang = []
    for a, b in zip(R, R0):
        cosv = (np.trace(a @ b.T) - 1) / 2
        ang.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    return float(np.mean(ang))


# ---------------------------------------------------------------- pickers
def regret(r, e):
    return r["cd"][e] / min(r["cd"].values())


def summarize(name, regs):
    regs = np.asarray(regs)
    return (f"  {name:<28} median {np.median(regs):.3f} | mean {np.mean(regs):.3f} | "
            f"worst {np.max(regs):.2f} | disasters(>1.2) {int(np.sum(regs > 1.2))}")


def guarded_pick(r, default, cand, drift_key, tau):
    if cand == default:
        return default
    return cand if r[drift_key][cand] <= tau else default


def loo(rows, pick_fn, taus):
    """Leave-one-out: for scene i choose tau minimizing mean regret on the others."""
    regs, chosen = [], []
    for i, r in enumerate(rows):
        others = rows[:i] + rows[i + 1:]
        best = min(taus, key=lambda t: np.mean([regret(o, pick_fn(o, t)) for o in others]))
        chosen.append(best)
        regs.append(regret(r, pick_fn(r, best)))
    return regs, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--metrics-name", default="eval_metrics_0_23.json")
    ap.add_argument("--default", type=int, default=1000, help="safe fallback eta")
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()

    etas = sorted(int(p.name.split("_")[1]) for p in a.results.glob("output_*")
                  if p.name.split("_")[1].isdigit() and int(p.name.split("_")[1]) > 0)
    ref_eta = etas[0]
    scenes = sorted({p.parent.name for p in a.results.glob(f"output_{ref_eta}/*/inter_raw.npz")})
    rows = []
    for s in scenes:
        paths = {e: a.results / f"output_{e}" / s for e in etas}
        if not all((p / "inter_raw.npz").exists() and (p / a.metrics_name).exists() for p in paths.values()):
            continue
        cds = {e: cd_of(paths[e] / a.metrics_name) for e in etas}
        if any(v is None for v in cds.values()):
            continue
        preds = {e: load_pred(paths[e] / "inter_raw.npz") for e in etas}
        r = {"scene": s, "cd": cds,
             "conf": {e: f_conf(*preds[e]) for e in etas},
             "mvc": {e: f_mvc(*preds[e]) for e in etas},
             "ddrift": {e: f_ddrift(preds[e], preds[ref_eta]) for e in etas},
             "rdrift": {e: f_rdrift(preds[e], preds[ref_eta]) for e in etas}}
        rows.append(r)
        print(f"  {s[:38]:<38} drift@{etas[-1]}: depth {r['ddrift'][etas[-1]]:.3f}  rot {r['rdrift'][etas[-1]]:.2f}deg"
              f"   regret@{etas[-1]} {regret(r, etas[-1]):.2f}", flush=True)
    if not rows:
        raise SystemExit("no complete scenes")
    n = len(rows); top = etas[-1]; D = a.default
    print(f"\n{n} scenes, etas={etas}, default={D}\n")

    print("Fixed eta:")
    for e in etas:
        print(summarize(f"fixed eta={e}", [regret(r, e) for r in rows]))
    print("\nProxy argmax (no guard):")
    print(summarize("conf argmax", [regret(r, max(etas, key=lambda e: r['conf'][e])) for r in rows]))
    print(summarize("mvc argmin", [regret(r, min(etas, key=lambda e: r['mvc'][e])) for r in rows]))

    print("\nGuarded (tau tuned leave-one-out; GT never used for the held-out scene):")
    for dk in ("ddrift", "rdrift"):
        taus = sorted({0.0, np.inf} | {r[dk][e] for r in rows for e in etas if np.isfinite(r[dk][e])})
        f1 = lambda r, t, dk=dk: guarded_pick(r, D, max(etas, key=lambda e: r["conf"][e]), dk, t)
        f2 = lambda r, t, dk=dk: guarded_pick(r, D, top, dk, t)
        regs, ch = loo(rows, f1, taus)
        print(summarize(f"guarded-conf  [{dk}]", regs) + f"   tau~{np.median(ch):.3g}")
        regs, ch = loo(rows, f2, taus)
        print(summarize(f"guarded-{top} [{dk}]", regs) + f"   tau~{np.median(ch):.3g}")

    # does drift separate the eta=top disasters?
    from scipy.stats import spearmanr
    for dk in ("ddrift", "rdrift"):
        x = [r[dk][top] for r in rows]; y = [regret(r, top) for r in rows]
        print(f"\nspearman({dk}@{top}, regret@{top}) = {spearmanr(x, y)[0]:+.2f}   (positive = drift flags bad runs)")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "eta", "cd", "regret", "conf", "mvc", "ddrift", "rdrift"])
            for r in rows:
                for e in etas:
                    w.writerow([r["scene"], e, r["cd"][e], regret(r, e), r["conf"][e], r["mvc"][e], r["ddrift"][e], r["rdrift"][e]])
        print(f"\nfeature table -> {a.csv}")


if __name__ == "__main__":
    main()
