"""Ancora com a literatura (PLANO A1): extremo MaxMin da nossa frente vs valor publicado do GDP.

Fonte do valor publicado: results.xlsx do pacote Zenodo 18233882 (Lozano-Osorio et al. 2022, VNS hibrido,
limite de 60 s), 120 instancias n=500 (familia GKD-c). Nosso lado: maior 2o objetivo (MaxMin) entre os pontos
das frentes de Pareto gravadas (nao armazenamos as solucoes, entao a viabilidade nao e re-verificada aqui).

    python anchor_gdp.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import openpyxl

ROOT = Path(__file__).parent
XLSX = ROOT / "a pedidos" / "isaaclo97-Generalized-Dispersion-Problem-2c0fa37" / "solution" / "results.xlsx"
MODES = ["hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar", "random",
         "nsga2", "grasp", "grasp_pr", "vns_pareto", "vns_tchebycheff"]
AOS = MODES[:7]


def published() -> dict[str, dict]:
    ws = openpyxl.load_workbook(XLSX, data_only=True)["DATA"]
    out = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0]:
            out[r[0].replace(".txt", "")] = {"disp": float(r[1]), "time": r[2], "cost": r[3], "maxcost": r[4], "cap": r[5], "mincap": r[6]}
    return out


def best_f2(path: Path) -> float | None:
    if not path.exists():
        return None
    return max(p[1] for p in json.load(open(path))["pareto_front"])


def main():
    pub = published()
    srcs = {"catalogo (historico)": ROOT / "results/capacitated/GDP/runs",
            "normalizado (capacitated_norm)": ROOT / "results/capacitated_norm/GDP/runs"}
    ours_names = sorted(n for n in pub if (srcs["catalogo (historico)"] / f"{n}_hypervolume_rep00.json").exists())
    print(f"{len(pub)} instancias publicadas; {len(ours_names)} estao no nosso catalogo GDP (todas n=500, GKD-c)\n")
    for label, d in srcs.items():
        rows = []
        for n in ours_names:
            per = {m: best_f2(d / f"{n}_{m}_rep00.json") for m in MODES}
            per = {m: v for m, v in per.items() if v is not None}
            if len(per) < 12:
                continue
            rows.append((n, max(per.values()), per))
        if not rows:
            print(f"-- {label}: sem instancias completas ainda\n")
            continue
        ratio_best = [b / pub[n]["disp"] for n, b, _ in rows]
        print(f"-- {label}: N={len(rows)} instancias completas")
        print(f"   melhor MaxMin nosso (qualquer metodo) / publicado: media {st.mean(ratio_best):.3f}  mediana {st.median(ratio_best):.3f}  min {min(ratio_best):.3f}  max {max(ratio_best):.3f}")
        print(f"   instancias em que igualamos ou superamos o publicado: {sum(r >= 0.999 for r in ratio_best)}/{len(rows)}")
        print("   razao media por metodo (MaxMin do extremo de cada metodo / publicado):")
        for m in MODES:
            rr = [per[m] / pub[n]["disp"] for n, _, per in rows]
            print(f"      {m:16s} {st.mean(rr):.3f}")
        print()


if __name__ == "__main__":
    main()


def published_point_analysis(label: str, runs_dir: Path) -> None:
    """Posiciona o ponto (MaxSum, MaxMin) da solucao publicada em relacao a uniao das nossas frentes."""
    import sys

    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    import validate_published_solutions as V
    from capacitated_problem import GDPInstance
    from metrics import hypervolume, pareto_front

    res = {}
    for f in sorted((V.PKG / "solution" / "solutions").glob("*.txt")):
        inst_file = ROOT / "instances" / "GDP (Const.)" / f"{f.stem}.txt"
        if not inst_file.exists() or not all((runs_dir / f"{f.stem}_{m}_rep00.json").exists() for m in MODES):
            continue
        nodes, *_ = V.parse_solution(f)
        inst = GDPInstance(inst_file)
        pt = np.array(inst.evaluate(np.array(sorted(nodes), dtype=np.intp)), dtype=float)
        ours = pareto_front(np.vstack([np.array(json.load(open(runs_dir / f"{f.stem}_{m}_rep00.json"))["pareto_front"]) for m in MODES]))
        dominated = bool(np.any(np.all(ours >= pt, axis=1) & np.any(ours > pt, axis=1)))
        n_dom = int(np.sum(np.all(pt >= ours, axis=1) & np.any(pt > ours, axis=1)))
        both = np.vstack([ours, pt[None, :]])
        lo = both.min(axis=0); hi = both.max(axis=0)
        rp = lo - 0.01 * np.maximum(hi - lo, [1.0, 1e-9])
        hv_ours = hypervolume(ours, rp); hv_all = hypervolume(pareto_front(both), rp)
        fam = f.stem.split("_")[0]
        res.setdefault((fam, f.stem[-3:]), []).append((dominated, n_dom, len(ours), 1.0 - hv_ours / hv_all))
    print(f"\n-- ponto publicado vs uniao das nossas 12 frentes [{label}]")
    print(f"   {'familia/k':12s} N   publicado dominado por nos   pontos nossos que ele domina   perda de HV da nossa uniao ao incluir o ponto dele")
    for (fam, k), v in sorted(res.items()):
        print(f"   {fam+' '+k:12s} {len(v):2d}  {sum(x[0] for x in v):2d}/{len(v):2d}                        "
              f"{sum(x[1] for x in v):3d}/{sum(x[2] for x in v):3d}                         media {st.mean(x[3] for x in v)*100:5.1f}%  max {max(x[3] for x in v)*100:5.1f}%")


if __name__ == "__main__":
    published_point_analysis("catalogo historico", ROOT / "results/capacitated/GDP/runs")


def anchor_effect_gdp(iters: int = 6000, restarts: int = 3) -> None:
    """Aplica src/anchor.maxmin_anchor as instancias GDP n=500 do piloto (results/capacitated_norm/GDP) que tem valor
    publicado e compara o MaxMin da ancora com o publicado; confere viabilidade (nosso criterio e o estrito deles: custo < K1)."""
    import sys

    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from anchor import maxmin_anchor
    from capacitated_problem import GDPInstance

    pub = published()
    norm = ROOT / "results/capacitated_norm/GDP/runs"
    cache = ROOT / "anchors" / "gdp_n500_pilot_anchor_points.json"
    out = json.load(open(cache)) if cache.exists() else {}
    names = [n for n in pub if all((norm / f"{n}_{m}_rep00.json").exists() for m in MODES)]
    for n in names:
        if n in out:
            continue
        inst = GDPInstance(ROOT / "instances" / "GDP (Const.)" / f"{n}.txt")
        r = maxmin_anchor(inst, np.random.default_rng(12345), iters=iters, restarts=restarts)
        if r is None:
            out[n] = None
            continue
        sol, f = r
        out[n] = {"maxsum": f[0], "maxmin": f[1], "cost": float(inst.cost[sol].sum()), "K1": float(inst.max_budget),
                  "cap": float(inst.capacity[sol].sum()), "B": float(inst.min_capacity), "size": int(len(sol))}
        print(f"ancora {n}: MaxMin {f[1]:.2f} (publicado {pub[n]['disp']:.2f}) custo {out[n]['cost']:.1f}/{out[n]['K1']:.0f}", flush=True)
        cache.parent.mkdir(exist_ok=True)
        json.dump(out, open(cache, "w"))
    rows = [(n, pub[n]["disp"], max(best_f2(norm / f"{n}_{m}_rep00.json") for m in MODES[:7]), out[n]) for n in names if out.get(n)]
    print(f"\n-- ancora (tabu MaxMin) vs MOEA/D-AOS normalizado vs publicado, GDP n=500 piloto (N={len(rows)})")
    for fam in ("GKD-c", "MDG-b"):
        rr = [r for r in rows if r[0].startswith(fam)]
        if rr:
            print(f"   {fam}: N={len(rr)}  MOEA/D-AOS {st.mean(r[2]/r[1] for r in rr):.3f}  ancora {st.mean(r[3]['maxmin']/r[1] for r in rr):.3f} do publicado")
    print(f"   viaveis (custo < K1 estrito e cap >= B): {sum(r[3]['cost'] < r[3]['K1'] and r[3]['cap'] >= r[3]['B'] - 1e-9 for r in rows)}/{len(rows)}")
