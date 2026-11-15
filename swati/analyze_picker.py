#!/usr/bin/env python
"""Compare unsupervised eta pickers on the saved feature table, with significance tests.

    python swati/analyze_picker.py swati/results_vggt_dev/picker_features.csv

Input: the CSV written by picker.py (scene, eta, cd, regret, conf, mvc, ddrift, rdrift).
Runs in seconds (no model outputs reloaded).

Pickers (tau always tuned leave-one-out, so the held-out scene's GT is never used for its own pick):
  fixed eta=E          same eta everywhere
  guarded-5000         5000 if ddrift(5000) <= tau, else 1000
  largest-safe         largest eta with ddrift(eta) <= tau   (eta=1 is always safe: drift 0)
  guarded-5000 ratio   5000 if ddrift(5000)/ddrift(1000) <= tau, else 1000   (scale-free drift)

Stats vs fixed eta=1000: paired bootstrap 95% CI of the mean regret difference, Wilcoxon signed-rank p,
and per-scene wins / ties / losses.
"""
import argparse, csv
from collections import defaultdict
import numpy as np


def load(path):
    rows = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            e = int(r["eta"])
            for k in ("cd", "regret", "conf", "mvc", "ddrift", "rdrift"):
                rows[r["scene"]].setdefault(k, {})[e] = float(r[k])
    return [dict(scene=s, **v) for s, v in sorted(rows.items())]


def loo(rows, pick, taus):
    regs, picks, chosen = [], [], []
    for i, r in enumerate(rows):
        others = rows[:i] + rows[i + 1:]
        t = min(taus, key=lambda t: (np.mean([o["regret"][pick(o, t)] for o in others]), t))
        e = pick(r, t)
        regs.append(r["regret"][e]); picks.append(e); chosen.append(t)
    return np.array(regs), picks, chosen


def line(name, regs, extra=""):
    return (f"  {name:<26} median {np.median(regs):.3f} | mean {np.mean(regs):.4f} | worst {np.max(regs):.2f}"
            f" | disasters(>1.2) {int(np.sum(regs > 1.2))}{extra}")


def compare(name, regs, base, rng):
    d = regs - base                       # negative = picker better
    boots = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    from scipy.stats import wilcoxon
    nz = d[np.abs(d) > 1e-12]
    p = wilcoxon(nz).pvalue if len(nz) >= 5 else float("nan")
    w, t, l = int(np.sum(d < -1e-12)), int(np.sum(np.abs(d) <= 1e-12)), int(np.sum(d > 1e-12))
    print(f"  {name:<26} mean diff {np.mean(d):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  wilcoxon p={p:.3f}"
          f"  wins/ties/losses {w}/{t}/{l}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--default", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = load(a.features)
    etas = sorted(rows[0]["regret"])
    top, D = etas[-1], a.default
    rng = np.random.default_rng(a.seed)
    print(f"{len(rows)} scenes, etas={etas}, default={D}\n")

    res = {}
    for e in etas:
        res[f"fixed eta={e}"] = np.array([r["regret"][e] for r in rows])
    oracle = np.ones(len(rows))

    dd_taus = sorted({0.0, np.inf} | {r["ddrift"][e] for r in rows for e in etas})
    g5 = lambda r, t: top if r["ddrift"][top] <= t else D
    res["guarded-5000"], p_g5, t_g5 = loo(rows, g5, dd_taus)
    ls = lambda r, t: max(e for e in etas if r["ddrift"][e] <= t)
    res["largest-safe"], p_ls, t_ls = loo(rows, ls, dd_taus)
    ratio = {r["scene"]: r["ddrift"][top] / max(r["ddrift"][D], 1e-9) for r in rows}
    rt_taus = sorted({0.0, np.inf} | set(ratio.values()))
    g5r = lambda r, t: top if ratio[r["scene"]] <= t else D
    res["guarded-5000 ratio"], p_rt, t_rt = loo(rows, g5r, rt_taus)

    print("Regret (1.00 = best eta for that scene):")
    print(line("oracle (uses GT)", oracle))
    for k, v in res.items():
        extra = {"guarded-5000": f"   tau~{np.median(t_g5):.3g}", "largest-safe": f"   tau~{np.median(t_ls):.3g}",
                 "guarded-5000 ratio": f"   tau~{np.median(t_rt):.3g}"}.get(k, "")
        print(line(k, v, extra))

    from collections import Counter
    print("\nWhat the pickers chose:")
    print(f"  guarded-5000       {dict(Counter(p_g5))}")
    print(f"  largest-safe       {dict(Counter(p_ls))}")
    print(f"  guarded-5000 ratio {dict(Counter(p_rt))}")

    base = res[f"fixed eta={D}"]
    print(f"\nVs fixed eta={D} (negative diff = picker better):")
    for k in ("guarded-5000", "largest-safe", "guarded-5000 ratio", f"fixed eta={top}"):
        compare(k, res[k], base, rng)

    print("\nHeadroom: oracle mean 1.000 vs fixed-default mean "
          f"{np.mean(base):.4f}; share of that gap closed:")
    for k in ("guarded-5000", "largest-safe", "guarded-5000 ratio"):
        gap = np.mean(base) - 1.0
        print(f"  {k:<26} {100 * (np.mean(base) - np.mean(res[k])) / gap:5.1f}%")


if __name__ == "__main__":
    main()
