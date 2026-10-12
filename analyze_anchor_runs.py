"""Ancoragem em runs REAIS com replicas (run_phase2.py: random_anchor, dyts_hv_anchor) vs as versoes sem ancora, no CDP n<=150.
(1) hv_norm / ranks vs referencia fixa; (2) MaxMin do extremo (media sobre replicas do maior f2 da frente) vs BKS
(anchors/cdp_bks_symmetry2026_table1.csv; so b02 com BKS) por familia; (3) custo em tempo.

    python src/reference_sets.py build && python analyze_anchor_runs.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import reference_sets as RS  # noqa: E402
import anchor_cdp as A  # noqa: E402

P2 = NORM = PAIRS = None


def load(d, nm, m, k):
    return json.load(open(d / f"{nm}_{m}_rep{k:02d}.json"))


def main():
    global P2, NORM, PAIRS
    problem = sys.argv[1] if len(sys.argv) > 1 else "CDP"
    P2 = ROOT / "results/phase2_aos" / problem / "runs"; NORM = ROOT / "results/capacitated_norm" / problem / "runs"
    PAIRS = [("random_ref", P2, "random_anchor", P2), ("dyts_hv", NORM, "dyts_hv_anchor", P2)]
    ref = RS.load(problem); reps = 3
    names = [r["name"] for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["problem"] == problem and int(r["n"]) <= 150]
    bks = {}
    for r in (csv.DictReader(open(ROOT / "anchors/cdp_bks_symmetry2026_table1.csv")) if problem == "CDP" else []):
        n = A.our_name(r["instance"], set(names))
        if n: bks[n] = float(r["bks"])
    for base, bd, anc, ad in PAIRS:
        ok = [n for n in names if all((bd / f"{n}_{base}_rep{k:02d}.json").exists() and (ad / f"{n}_{anc}_rep{k:02d}.json").exists() for k in range(reps))]
        if len(ok) < 5:
            print(f"{anc}: poucas instancias completas ({len(ok)})"); continue
        hv = lambda d, m: np.array([np.mean([RS.indicators(np.array(load(d, n, m, k)["pareto_front"]), ref[n])["hv_norm"] for k in range(reps)]) for n in ok])
        h0, h1 = hv(bd, base), hv(ad, anc)
        w = wilcoxon(h1 - h0).pvalue if np.any(h1 != h0) else 1.0
        ext = lambda d, m, n: np.mean([max(p[1] for p in load(d, n, m, k)["pareto_front"]) for k in range(reps)])
        t0 = st.median(load(bd, n, base, 0)["elapsed_s"] for n in ok); t1 = st.median(load(ad, n, anc, 0)["elapsed_s"] for n in ok)
        print(f"\n=== {anc} vs {base}  (N={len(ok)} instancias {problem} n<=150, {reps} replicas)")
        print(f"  hv_norm medio: {h0.mean():.4f} -> {h1.mean():.4f} (dif {h1.mean()-h0.mean():+.4f}; melhor em {int((h1>h0).sum())}/{len(ok)}; Wilcoxon p={w:.3g})")
        print(f"  tempo mediano por run: {t0:.1f}s -> {t1:.1f}s ({t1/t0*100:.0f}%)")
        fams = sorted({n.split('_')[0] + '_n' + n.split('_n')[1].split('_')[0] for n in ok if n in bks})
        for fam in fams:
            rr = [n for n in ok if n in bks and n.split('_')[0] + '_n' + n.split('_n')[1].split('_')[0] == fam]
            g0 = st.mean(ext(bd, base, n) / bks[n] for n in rr); g1 = st.mean(ext(ad, anc, n) / bks[n] for n in rr)
            at = sum(ext(ad, anc, n) >= bks[n] - 1e-9 for n in rr)
            print(f"  MaxMin do extremo / BKS  {fam:11s} N={len(rr):2d}: {g0:.3f} -> {g1:.3f}   (atinge BKS: {at}/{len(rr)})")


if __name__ == "__main__":
    main()
