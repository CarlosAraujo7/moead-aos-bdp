"""E5: efeito de relaxar o orcamento do GDP (run_e5.py) sobre o ranking dos 12 metodos, regime normalizado, 3 replicas.

Em cada regime (original = results/capacitated_norm/GDP; relaxado = results/e5_budget_relaxed_norm/GDP) a referencia de cada instancia e a frente
nao-dominada da uniao dos 12 metodos x 3 replicas DAQUELE regime (ponto de referencia 1% abaixo do pior valor), de modo que os dois regimes sao
rankeados internamente (as solucoes relaxadas podem violar o orcamento original e nao seriam comparaveis numa referencia unica). Para cada metodo:
rank medio por instancia em cada regime e Wilcoxon pareado sobre os ranks por instancia (relaxado vs original), Holm sobre os 12 metodos.

    python analyze_e5.py
Saida: terminal + results/analysis/e5_budget_relaxed_norm.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from experiment import ALL_MODES  # noqa: E402
from metrics import hypervolume, pareto_front  # noqa: E402
from analyze_phase2 import holm  # noqa: E402

ORIG = ROOT / "results/capacitated_norm/GDP/runs"
RELX = ROOT / "results/e5_budget_relaxed_norm/GDP/runs"
REPS = 3
LAB = {"hypervolume": "UCB-HV", "thompson_hv": "TS-HV", "dyts_hv": "GTS-HV", "scalar": "UCB-Scal", "thompson_scalar": "TS-Scal",
       "dyts_scalar": "GTS-Scal", "random": "Uniform", "nsga2": "NSGA-II", "grasp": "GRASP-MO", "grasp_pr": "GRASP-PR",
       "vns_pareto": "VNS-Pareto", "vns_tchebycheff": "VNS-Tcheb."}


def load_front(d, name, mode, k):
    return np.array(json.load(open(d / f"{name}_{mode}_rep{k:02d}.json"))["pareto_front"], float).reshape(-1, 2)


def hv_matrix(d, name):
    fronts = {(m, k): load_front(d, name, m, k) for m in ALL_MODES for k in range(REPS)}
    allp = np.vstack(list(fronts.values()))
    rf = pareto_front(allp)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    rp = lo - 0.01 * np.maximum(hi - lo, [1.0, 1e-9])
    hr = hypervolume(rf, rp)
    return np.array([np.mean([hypervolume(pareto_front(fronts[(m, k)]), rp) / hr for k in range(REPS)]) for m in ALL_MODES])


def main():
    cat = {r["name"]: r for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["problem"] == "GDP"}
    names = sorted({f.name.split("_" + ALL_MODES[0] + "_rep")[0] for f in RELX.glob(f"*_{ALL_MODES[0]}_rep00.json")})
    ok = [n for n in names if all((RELX / f"{n}_{m}_rep{k:02d}.json").exists() and (ORIG / f"{n}_{m}_rep{k:02d}.json").exists()
                                  for m in ALL_MODES for k in range(REPS))]
    out = {}
    for lab, lo_n, hi_n in (("n<=150", 0, 150), ("n=500", 500, 500)):
        nn = [n for n in ok if lo_n <= int(cat[n]["n"]) <= hi_n]
        if len(nn) < 5:
            print(f"\n=== GDP {lab}: {len(nn)} instancias completas (poucas)"); continue
        Ro, Rr = [], []
        Ho, Hr = [], []
        for n in nn:
            ho, hr = hv_matrix(ORIG, n), hv_matrix(RELX, n)
            Ho.append(ho); Hr.append(hr)
            Ro.append(rankdata(-ho)); Rr.append(rankdata(-hr))
        Ro, Rr = np.array(Ro), np.array(Rr)
        ps = []
        for j in range(len(ALL_MODES)):
            d = Rr[:, j] - Ro[:, j]
            ps.append(wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
        pa = holm(ps)
        print(f"\n=== GDP {lab}: N={len(nn)} instancias, {REPS} replicas; rank medio (de 12) original -> relaxado (negativo = melhora); Wilcoxon sobre ranks, Holm x12")
        res = {}
        for j in np.argsort(Ro.mean(axis=0)):
            m = ALL_MODES[j]
            res[m] = dict(rank_orig=float(Ro[:, j].mean()), rank_relaxed=float(Rr[:, j].mean()), p_holm=float(pa[j]),
                          hv_orig=float(np.mean([h[j] for h in Ho])), hv_relaxed=float(np.mean([h[j] for h in Hr])))
            print(f"  {LAB[m]:11s} {Ro[:, j].mean():5.2f} -> {Rr[:, j].mean():5.2f} ({Rr[:, j].mean() - Ro[:, j].mean():+5.2f})  p_holm={pa[j]:.4f}{' *' if pa[j] < 0.05 else ''}")
        best_o, best_r = Ro.mean(axis=0).min(), Rr.mean(axis=0).min()
        print(f"  melhor rank medio: original {best_o:.2f}; relaxado {best_r:.2f}")
        out[lab] = dict(N=len(nn), methods=res)
    (ROOT / "results/analysis/e5_budget_relaxed_norm.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
