"""Efeito da ancoragem de MaxMin (src/anchor.py) sobre o piloto de n=500 do CDP (results/capacitated_norm/CDP).

Calcula o ponto ancora de cada instancia (busca tabu, 6000 it x 3 reinicios, semente fixa; cache em
anchors/cdp_n500_pilot_anchor_points.json) e o acrescenta as frentes JA GRAVADAS (sem reexecutar nada), em tres cenarios:
  0) como gravado;  A) ancora acrescentada a TODOS os metodos (comparacao justa: extremos equalizados);
  B) ancora so nos 7 MOEA/D-AOS (o que o metodo ancorado entregaria; baselines sem ancora).
Reporta ranks medios (hv_norm, frente agrupada), AOS vs random (Wilcoxon, Holm) e MaxMin vs BKS.

    python analyze_anchor_effect.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from anchor import maxmin_anchor  # noqa: E402
from capacitated_problem import CDPInstance  # noqa: E402
from experiment import ALL_MODES, build_ref_point, compute_indicators, compute_reference_front  # noqa: E402
import anchor_cdp as A  # noqa: E402

AOS = ALL_MODES[:7]
CACHE = ROOT / "anchors" / "cdp_n500_pilot_anchor_points.json"
NORM = ROOT / "results" / "capacitated_norm" / "CDP" / "runs"


def fr(p):
    return np.array(json.load(open(p))["pareto_front"])


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def main():
    cat = {r["name"]: r["filepath"] for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv"))}
    names = sorted({f.name.rsplit("_hypervolume_rep00.json", 1)[0] for f in NORM.glob("*_n500_*_hypervolume_rep00.json")})
    names = [n for n in names if all((NORM / f"{n}_{m}_rep00.json").exists() for m in ALL_MODES)]
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    for n in names:
        if n not in cache:
            inst = CDPInstance(ROOT / "instances" / cat[n]); inst.name = n
            r = maxmin_anchor(inst, np.random.default_rng(12345), iters=6000, restarts=3)
            cache[n] = list(r[1]) if r else None
            print(f"ancora {n}: {cache[n]}", flush=True)
            CACHE.parent.mkdir(exist_ok=True); json.dump(cache, open(CACHE, "w"))
    scen = {"0) como gravado": lambda m: False, "A) ancora em TODOS": lambda m: True, "B) ancora so nos AOS": lambda m: m in AOS}
    for label, use in scen.items():
        H = []
        for n in names:
            a = np.array([cache[n]]) if cache[n] else np.zeros((0, 2))
            fs = {m: (np.vstack([fr(NORM / f"{n}_{m}_rep00.json"), a]) if use(m) and len(a) else fr(NORM / f"{n}_{m}_rep00.json")) for m in ALL_MODES}
            allf = list(fs.values()); rf, rp = compute_reference_front(allf), build_ref_point(allf)
            H.append([compute_indicators(fs[m], rf, rp)["hv_norm"] for m in ALL_MODES])
        H = np.array(H); R = np.array([rankdata(-r) for r in H]); jr = ALL_MODES.index("random")
        print(f"\n=== {label}  (N={len(H)} instancias CDP n=500)")
        for j in np.argsort(R.mean(0)):
            print(f"    {ALL_MODES[j]:16s} rank medio {R.mean(0)[j]:5.2f}   hv_norm medio {H.mean(0)[j]:.4f}")
        ps, nm, dm = [], [], []
        for m in AOS:
            if m == "random": continue
            d = H[:, ALL_MODES.index(m)] - H[:, jr]; nm.append(m); dm.append(d.mean()); ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
        adj = holm(np.array(ps))
        print("    AOS vs random: " + "  ".join(f"{a_[:10]} {d_:+.3f}(p={p_:.2f})" for a_, d_, p_ in zip(nm, dm, adj)))
    # MaxMin vs BKS (instancias b02 com BKS)
    table = {r["instance"]: float(r["bks"]) for r in csv.DictReader(open(ROOT / "anchors" / "cdp_bks_symmetry2026_table1.csv"))}
    rows = []
    for n in names:
        bks = [v for k, v in table.items() if A.our_name(k, {n}) == n]
        if bks and cache[n]:
            moea = max(A.f2max(NORM / f"{n}_{m}_rep00.json") for m in AOS)
            rows.append((n, bks[0], moea, cache[n][1]))
    print(f"\n=== MaxMin vs BKS nas {len(rows)} instancias n=500 b02 do piloto que tem BKS")
    for fam in ("GKD-c", "MDG-b"):
        rr = [r for r in rows if r[0].startswith(fam)]
        if rr:
            print(f"    {fam}: N={len(rr)}  MOEA/D-AOS normalizado {st.mean(r[2]/r[1] for r in rr):.3f} do BKS | ancora tabu {st.mean(r[3]/r[1] for r in rr):.3f} do BKS")


if __name__ == "__main__":
    main()
