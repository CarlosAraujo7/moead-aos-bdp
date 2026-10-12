"""E21: escala completa no MDP (run_e21.py): MDG-b n=2000 (20 instancias) e MDG-c n=3000 (amostra), baselines ajustados e MOEA/D hibrido a tempo igual.

Referencia por instancia = uniao nao-dominada de TODOS os runs disponiveis da instancia (E21, E10 [MOEA/D-Uniform, UCB-HV, GTS-HV, 1x/4x], E6); hv_norm,
IGD+ e epsilon+ contra ela (replica 0). Na familia MDG-c (MaxMin = 0 em quase todo subconjunto) o hv_norm e degenerado: usa-se o MaxSum maximo de cada
metodo como fracao do melhor MaxSum conhecido. Wilcoxon + Holm e A12 contra o MOEA/D-Uniform, pareados por instancia.

    python analyze_e21.py [--metric hv_norm|igd_plus]
Saida: terminal + results/analysis/e21_xl[_igd_plus].json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import reference_sets as RS  # noqa: E402
from metrics import pareto_front  # noqa: E402
from analyze_phase2 import a12, holm  # noqa: E402

E21 = ROOT / "results" / "e21_xl" / "runs"
E10 = ROOT / "results" / "e10_budget_regime" / "MDP" / "runs"
E6 = ROOT / "results" / "e6_scalability" / "runs"
LAB = {"uniform": "MOEA/D-Uniform", "t_ts": "Tabu (tuned at n=500)", "t_vns": "VNS (tuned at n=500)", "t_grasp": "GRASP (tuned at n=500)",
       "t_nsga2": "NSGA-II (tuned at n=500)", "m_hyb_vns50": "MOEA/D seeded by VNS (50% of T)", "ucb_hv": "MOEA/D-UCB-HV", "gts_hv": "MOEA/D-GTS-HV"}


def front(f):
    return np.asarray(json.load(open(f))["pareto_front"], float).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus"], default="hv_norm")
    a = ap.parse_args()
    hb = a.metric == "hv_norm"
    rows = {r["name"]: r for r in csv.DictReader(open(ROOT / "instances" / "catalog.csv"))}
    names = sorted({f.name.split("_uniform_rep00")[0] for f in E21.glob("*_uniform_rep00.json")})
    res = {}
    for fam in ("MDG-b", "MDG-c"):
        ins = [n for n in names if rows[n]["family"] == fam]
        table, used = [], []
        for n in ins:
            fr = {m: E21 / f"{n}_{m}_rep00.json" for m in LAB if m in ("uniform", "t_ts", "t_vns", "t_grasp", "t_nsga2", "m_hyb_vns50")}
            if not all(p.exists() for p in fr.values()):
                continue
            fronts = {m: front(p) for m, p in fr.items()}
            if fam == "MDG-b":                            # AOS do E10 (rep 0), se existir
                for m, v in (("ucb_hv", "hypervolume"), ("gts_hv", "dyts_hv")):
                    f = E10 / f"{n}_{v}_rep00.json"
                    if f.exists():
                        fronts[m] = front(f)
            extra = [front(f) for f in list(E10.glob(f"{n}_*_rep0*.json")) + list(E6.glob(f"{n}_*_rep0*.json"))]
            allp = np.vstack(list(fronts.values()) + extra)
            if fam == "MDG-b":
                lo, hi = allp.min(axis=0), allp.max(axis=0)
                ref = {"front": pareto_front(allp).tolist(), "ref_point": (lo - RS.MARGIN * np.maximum(hi - lo, [1.0, 1e-9])).tolist()}
                if RS.hypervolume(np.asarray(ref["front"]), np.asarray(ref["ref_point"])) <= 0:
                    continue
                vals = {m: RS.indicators(f, ref)[a.metric] for m, f in fronts.items()}
                vals = {m: (v if hb else -v) for m, v in vals.items()}
            else:
                best = allp[:, 0].max()
                vals = {m: float(f[:, 0].max() / best) for m, f in fronts.items()}
            table.append(vals)
            used.append(n)
        if len(used) < 3:
            print(f"{fam}: {len(used)} instancias completas")
            continue
        modes = [m for m in LAB if all(m in t for t in table)]
        M = np.array([[t[m] for m in modes] for t in table])
        k0 = modes.index("uniform")
        ps, info = [], {}
        for j, m in enumerate(modes):
            d = M[:, j] - M[:, k0]
            p = wilcoxon(d).pvalue if (m != "uniform" and np.any(d != 0)) else 1.0
            ps.append(p)
            info[m] = dict(mean=float(M[:, j].mean()), diff=float(d.mean()), a12=a12(M[:, j], M[:, k0]), better=int((d > 0).sum()), p_raw=float(p))
        idx = [j for j, m in enumerate(modes) if m != "uniform"]
        for j, pa in zip(idx, holm([ps[j] for j in idx])):
            info[modes[j]]["p_holm"] = float(pa)
        res[fam] = dict(N=len(used), metric=("maxsum_fraction" if fam == "MDG-c" else a.metric), methods=info, instances=used)
        print(f"\n=== {fam} [{res[fam]['metric']}]: N={len(used)} instancias (replica 0)")
        for m in sorted(modes, key=lambda m: -info[m]["mean"]):
            i = info[m]
            ph = f"{i['p_holm']:.4f}" if "p_holm" in i else "   -"
            print(f"   {LAB[m]:36s} {i['mean']:.4f}  vs uniform {i['diff']:+.4f} (A12 {i['a12']:.2f}, melhor em {i['better']:2d}/{len(used)}, p_Holm {ph})")
    suffix = "" if a.metric == "hv_norm" else f"_{a.metric}"
    (ROOT / "results" / "analysis" / f"e21_xl{suffix}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
