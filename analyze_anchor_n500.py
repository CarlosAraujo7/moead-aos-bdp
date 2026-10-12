"""Ancoragem em runs REAIS com replicas em n=500 (run_phase2.py: random_anchor6k) vs `random` (results/capacitated_norm, mesmas
instancias/replicas): hv_norm contra a referencia fixa, e MaxMin do extremo vs BKS (CDP, Tabela 1 do Symmetry 2026) ou vs valor
publicado (GDP, results.xlsx do pacote Zenodo). Rode antes: python src/reference_sets.py build

    python analyze_anchor_n500.py
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
import anchor_cdp as AC  # noqa: E402
import anchor_gdp as AG  # noqa: E402

REPS = 3


def load(d, nm, m, k):
    return json.load(open(d / f"{nm}_{m}_rep{k:02d}.json"))


def run(problem):
    ref = RS.load(problem)
    norm = ROOT / "results/capacitated_norm" / problem / "runs"; p2 = ROOT / "results/phase2_aos" / problem / "runs"
    names = [r["name"] for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["problem"] == problem and int(r["n"]) == 500]
    ok = [n for n in names if all((norm / f"{n}_random_rep{k:02d}.json").exists() and (p2 / f"{n}_random_anchor6k_rep{k:02d}.json").exists() for k in range(REPS))]
    hv = lambda d, m: np.array([np.mean([RS.indicators(np.array(load(d, n, m, k)["pareto_front"]), ref[n])["hv_norm"] for k in range(REPS)]) for n in ok])
    h0, h1 = hv(norm, "random"), hv(p2, "random_anchor6k")
    w = wilcoxon(h1 - h0).pvalue
    t0 = st.median(load(norm, n, "random", 0)["elapsed_s"] for n in ok); t1 = st.median(load(p2, n, "random_anchor6k", 0)["elapsed_s"] for n in ok)
    print(f"\n=== {problem} n=500: random_anchor6k vs random  (N={len(ok)} instancias, {REPS} replicas)")
    print(f"  hv_norm medio: {h0.mean():.4f} -> {h1.mean():.4f} (dif {h1.mean()-h0.mean():+.4f}; melhor em {int((h1>h0).sum())}/{len(ok)}; Wilcoxon p={w:.3g})")
    print(f"  tempo mediano por run: {t0:.0f}s -> {t1:.0f}s (cargas diferentes; so indicativo)")
    ext = lambda d, m, n: np.mean([max(p[1] for p in load(d, n, m, k)["pareto_front"]) for k in range(REPS)])
    if problem == "CDP":
        table = {r["instance"]: float(r["bks"]) for r in csv.DictReader(open(ROOT / "anchors/cdp_bks_symmetry2026_table1.csv"))}
        target = {}
        for t in table:
            n = AC.our_name(t, set(ok))
            if n: target[n] = table[t]
        label = "BKS"
    else:
        pub = AG.published(); target = {n: pub[n]["disp"] for n in ok if n in pub}; label = "publicado"
    for fam in ("GKD-c", "MDG-b"):
        rr = [n for n in ok if n in target and n.startswith(fam)]
        if rr:
            a0 = st.mean(ext(norm, "random", n) / target[n] for n in rr); a1 = st.mean(ext(p2, "random_anchor6k", n) / target[n] for n in rr)
            at = sum(ext(p2, "random_anchor6k", n) >= target[n] - 1e-9 for n in rr)
            print(f"  MaxMin do extremo / {label}  {fam:6s} N={len(rr):2d}: {a0:.3f} -> {a1:.3f}   (atinge/supera: {at}/{len(rr)})")


if __name__ == "__main__":
    for p in ("CDP", "GDP"):
        run(p)
