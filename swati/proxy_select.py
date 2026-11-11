# #!/usr/bin/env python
# """Starter for my part: pick eta per scene with an UNSUPERVISED proxy, then check it
# against ground truth (only for scoring, never for picking).

# Reads the sweep that 03_overnight.sh produces (needs SAVE_RAW_TENSORS=1):
#     results/vggt_dev/output_<eta>/<scene>/{inter_raw.npz, base_raw.npz, eval_metrics_0_23.json}

#     python proxy_select.py results/vggt_dev

# Proxies (all computed from the model's own outputs, no GT):
#   conf   mean log depth-confidence of the prediction            (higher = better)
#   mvc    multi-view depth consistency: unproject view i with its
#          predicted depth+camera, reproject into view j, compare to
#          view j's predicted depth -> median relative error       (lower = better)

# Next step (the actual research): turn the best proxy into an objective and optimize
# eta (or the token update itself) at inference time instead of picking from 5.
# """
# import argparse, json
# from pathlib import Path
# import numpy as np


# def load_pred(npz):
#     z = np.load(npz)
#     d = z["depth_map"].astype(np.float32).reshape(z["depth_map"].shape[1:4])   # S,H,W
#     c = z["depth_conf"].astype(np.float32).reshape(z["depth_conf"].shape[1:4])
#     E = z["extrinsics"].reshape(-1, 3, 4).astype(np.float64)                     # world->cam (OpenCV)
#     K = z["intrinsics"].reshape(-1, 3, 3).astype(np.float64)
#     return d, c, E, K


# def proxy_conf(d, c, E, K):
#     return float(np.mean(np.log(np.clip(c, 1e-6, None))))


# def proxy_mvc(d, c, E, K, n_pairs=40, n_pix=4000, seed=0):
#     """Median |z_reproj - z_j| / z_j over random view pairs. Scale-invariant."""
#     rng = np.random.default_rng(seed)
#     S, H, W = d.shape
#     errs = []
#     for _ in range(n_pairs):
#         i, j = rng.choice(S, 2, replace=False)
#         v = rng.integers(0, H, n_pix); u = rng.integers(0, W, n_pix)
#         z = d[i, v, u]
#         ok = z > 1e-4
#         v, u, z = v[ok], u[ok], z[ok]
#         pix = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], 0)
#         Xc = np.linalg.inv(K[i]) @ pix * z                    # cam i
#         R, t = E[i, :, :3], E[i, :, 3:]
#         Xw = R.T @ (Xc - t)                                   # world
#         Xj = E[j, :, :3] @ Xw + E[j, :, 3:]                   # cam j
#         zj = Xj[2]
#         front = zj > 1e-4
#         uvj = K[j] @ Xj
#         uj = uvj[0] / np.where(front, zj, 1) - 0.5
#         vj = uvj[1] / np.where(front, zj, 1) - 0.5
#         inb = front & (uj >= 0) & (uj <= W - 1) & (vj >= 0) & (vj <= H - 1)
#         if inb.sum() < 50:
#             continue
#         dj = d[j, np.round(vj[inb]).astype(int), np.round(uj[inb]).astype(int)]
#         rel = np.abs(zj[inb] - dj) / np.maximum(dj, 1e-4)
#         errs.append(np.median(rel))
#     return float(np.median(errs)) if errs else float("nan")


# PROXIES = {"conf": (proxy_conf, +1), "mvc": (proxy_mvc, -1)}  # sign: +1 maximize, -1 minimize


# def chamfer(entry):
#     if not entry:
#         return None
#     a, b = entry.get("Acc-mean", entry.get("Acc_mean")), entry.get("Comp-mean", entry.get("Comp_mean"))
#     return None if a is None or b is None else 0.5 * (float(a) + float(b))


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("results", type=Path)
#     ap.add_argument("--metrics-name", default="eval_metrics_0_23.json")
#     args = ap.parse_args()

#     etas = sorted(int(p.name.split("_")[1]) for p in args.results.glob("output_*"))
#     scenes = sorted({p.parent.name for p in args.results.glob("output_*/*/inter_raw.npz")})
#     rows = []
#     for s in scenes:
#         r = {"scene": s, "cd": {}, "proxy": {k: {} for k in PROXIES}}
#         base_cds = []
#         for eta in etas:
#             sd = args.results / f"output_{eta}" / s
#             if not (sd / "inter_raw.npz").exists():
#                 continue
#             m = json.loads((sd / args.metrics_name).read_text()) if (sd / args.metrics_name).exists() else {}
#             r["cd"][eta] = chamfer(m.get("inter"))
#             if chamfer(m.get("base")) is not None:
#                 base_cds.append(chamfer(m.get("base")))
#             pred = load_pred(sd / "inter_raw.npz")
#             for k, (fn, _) in PROXIES.items():
#                 r["proxy"][k][eta] = fn(*pred)
#         if base_cds and any(v is not None for v in r["cd"].values()):
#             r["base"] = float(np.mean(base_cds))
#             rows.append(r)

#     if not rows:
#         raise SystemExit("no complete scenes yet")
#     ratio = lambda r, eta: r["cd"][eta] / r["base"]
#     print(f"{len(rows)} scenes, etas={etas}   (numbers = median CD_inter / CD_base, lower is better)")
#     print(f"  oracle (sees GT)      {np.median([min(ratio(r, e) for e in r['cd']) for r in rows]):.3f}")
#     for eta in etas:
#         vals = [ratio(r, eta) for r in rows if r["cd"].get(eta) is not None]
#         print(f"  fixed eta={eta:<5}      {np.median(vals):.3f}")
#     for k, (_, sign) in PROXIES.items():
#         picks = [max(r["proxy"][k], key=lambda e: sign * r["proxy"][k][e]) for r in rows]
#         vals = [ratio(r, e) for r, e in zip(rows, picks)]
#         hit = np.mean([e == min(r["cd"], key=r["cd"].get) for r, e in zip(rows, picks)])
#         print(f"  proxy={k:<6} picks      {np.median(vals):.3f}   (picked oracle eta on {hit:.0%} of scenes)")
#     # per-scene correlation between proxy and true CD across eta: does the proxy track quality?
#     from scipy.stats import spearmanr
#     for k, (_, sign) in PROXIES.items():
#         rhos = [spearmanr([sign * r["proxy"][k][e] for e in r["cd"]], [-r["cd"][e] for e in r["cd"]])[0]
#                 for r in rows if len(r["cd"]) >= 3]
#         print(f"  spearman(proxy, quality) {k:<5} median rho = {np.nanmedian(rhos):+.2f}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
"""Pick eta per scene with an UNSUPERVISED proxy; score the pick against GT (GT used only for scoring).

    python proxy_select.py results/vggt_dev

Reads results/<out>/output_<eta>/<scene>/{inter_raw.npz, eval_metrics_0_23.json}.
Only scenes with all etas finished are used.

Metric: CD = 0.5*(Acc-mean + Comp-mean), lower is better.
Regret = CD(chosen eta) / CD(oracle eta)  -> 1.00 is perfect.

Proxies (from the model's own outputs, no GT):
  conf  mean log depth-confidence                          (higher = better)
  mvc   multi-view depth consistency, median rel. error     (lower = better)
"""
import argparse, json
from pathlib import Path
import numpy as np


def load_pred(npz):
    z = np.load(npz)
    S = z["depth_map"].shape[1]
    d = z["depth_map"].astype(np.float32).reshape(S, *z["depth_map"].shape[2:4])
    c = z["depth_conf"].astype(np.float32).reshape(S, *z["depth_conf"].shape[2:4])
    E = z["extrinsics"].reshape(-1, 3, 4).astype(np.float64)   # world->cam (OpenCV)
    K = z["intrinsics"].reshape(-1, 3, 3).astype(np.float64)
    return d, c, E, K


def proxy_conf(d, c, E, K):
    return float(np.mean(np.log(np.clip(c, 1e-6, None))))


def proxy_mvc(d, c, E, K, n_pairs=60, n_pix=4000, seed=0):
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


PROXIES = {"conf": (proxy_conf, +1), "mvc": (proxy_mvc, -1)}


def cd(m):
    m = m.get("inter") or {}
    a, b = m.get("Acc-mean"), m.get("Comp-mean")
    return None if a is None or b is None else 0.5 * (float(a) + float(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--metrics-name", default="eval_metrics_0_23.json")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    etas = sorted(int(p.name.split("_")[1]) for p in args.results.glob("output_*"))
    scenes = sorted({p.parent.name for p in args.results.glob("output_*/*/inter_raw.npz")})
    rows = []
    for s in scenes:
        r = {"scene": s, "cd": {}, "proxy": {k: {} for k in PROXIES}}
        for eta in etas:
            sd = args.results / f"output_{eta}" / s
            mf = sd / args.metrics_name
            if not (sd / "inter_raw.npz").exists() or not mf.exists():
                break
            v = cd(json.loads(mf.read_text()))
            if v is None:
                break
            r["cd"][eta] = v
            pred = load_pred(sd / "inter_raw.npz")
            for k, (fn, _) in PROXIES.items():
                r["proxy"][k][eta] = fn(*pred)
        if len(r["cd"]) == len(etas):
            rows.append(r)
            print(f"  {s[:40]:<40} CD by eta: " + " ".join(f"{r['cd'][e]:.4f}" for e in etas), flush=True)

    if not rows:
        raise SystemExit("no scene has all etas finished yet")
    oracle = lambda r: min(r["cd"].values())
    reg = lambda r, e: r["cd"][e] / oracle(r)

    print(f"\n{len(rows)} complete scenes, etas={etas}")
    print("regret = CD(choice)/CD(best eta for that scene); 1.00 = perfect.  [median | mean]")
    print(f"  oracle (sees GT)        1.000 | 1.000")
    for e in etas:
        v = [reg(r, e) for r in rows]
        print(f"  fixed eta={e:<5}         {np.median(v):.3f} | {np.mean(v):.3f}")
    from collections import Counter
    print("  oracle eta histogram:", dict(Counter(min(r['cd'], key=r['cd'].get) for r in rows)))
    from scipy.stats import spearmanr
    for k, (_, sign) in PROXIES.items():
        picks = [max(r["proxy"][k], key=lambda e: sign * r["proxy"][k][e]) for r in rows]
        v = [reg(r, e) for r, e in zip(rows, picks)]
        hit = np.mean([e == min(r["cd"], key=r["cd"].get) for r, e in zip(rows, picks)])
        rhos = [spearmanr([sign * r["proxy"][k][e] for e in etas], [-r["cd"][e] for e in etas])[0] for r in rows]
        print(f"  proxy={k:<5} picks       {np.median(v):.3f} | {np.mean(v):.3f}   "
              f"hit-oracle {hit:.0%}   median spearman {np.nanmedian(rhos):+.2f}   picks={dict(Counter(picks))}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "eta", "cd"] + list(PROXIES))
            for r in rows:
                for e in etas:
                    w.writerow([r["scene"], e, r["cd"][e]] + [r["proxy"][k][e] for k in PROXIES])
        print(f"\nper-(scene,eta) table -> {args.csv}")


if __name__ == "__main__":
    main()