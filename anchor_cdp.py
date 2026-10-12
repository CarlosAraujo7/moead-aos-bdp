"""Ancora com a literatura para o CDP (PLANO A1): extremo MaxMin da nossa frente vs melhores solucoes conhecidas (BKS).

BKS: Tabela 1 de Gomez, Chen, Calvet, Ammouriova & Juan (2026), Symmetry 18(1):110, doi:10.3390/sym18010110
(anchors/cdp_bks_symmetry2026_table1.csv; 50 instancias com demanda b02; BKS agrega Gurobi/Marti et al. 2021/Gomez et al.).
Nosso lado: maior 2o objetivo (MaxMin) entre os pontos das frentes gravadas; as solucoes nao sao armazenadas, entao a
viabilidade nao e re-verificada aqui (o avaliador reproduz as dispersoes publicadas do GDP, ver validate_published_solutions.py).

    python anchor_cdp.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).parent
MODES = ["hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar", "random",
         "nsga2", "grasp", "grasp_pr", "vns_pareto", "vns_tchebycheff"]


def our_name(table_name: str, existing: set[str]) -> str | None:
    if table_name.startswith("SOM-a"):
        cand = table_name.replace("_m", "_b02_m", 1)
        return cand if cand in existing else None
    cand = [n for n in existing if n.startswith(table_name + "_m")]
    return cand[0] if len(cand) == 1 else None


def f2max(path: Path):
    if not path.exists():
        return None
    return max(p[1] for p in json.load(open(path))["pareto_front"])


def main():
    table = list(csv.DictReader(open(ROOT / "anchors" / "cdp_bks_symmetry2026_table1.csv")))
    base = ROOT / "results" / "capacitated" / "CDP" / "runs"
    names = {f.name.rsplit("_rep00.json", 1)[0] for f in base.glob("*_hypervolume_rep00.json")}
    names = {n[: -len("_hypervolume")] for n in names}
    other = {  # fontes extras (mesma restricao do CDP): modo -> pasta
        "normalizado (ablation_normalized)": ROOT / "results/ablation_normalized/CDP/runs",
        "normalizado (capacitated_norm)": ROOT / "results/capacitated_norm/CDP/runs",
    }
    rows = []
    for r in table:
        n = our_name(r["instance"], names)
        if n is None:
            continue
        bks = float(r["bks"])
        per_hist = {m: f2max(base / f"{n}_{m}_rep00.json") for m in MODES}
        best_hist = max(v for v in per_hist.values() if v is not None)
        best_any = best_hist
        for d in other.values():
            for f in d.glob(f"{n}_*_rep*.json"):
                v = f2max(f)
                if v is not None:
                    best_any = max(best_any, v)
        rows.append((r["instance"], n, bks, float(r["gurobi_ub"]), best_hist, best_any, per_hist))
    print(f"{len(rows)} instancias CDP (b02) com BKS na Tabela 1 do artigo e no nosso catalogo\n")
    fams = sorted({r[0].split("_")[0] + ("_n" + r[0].split("_n")[1].split("_")[0]) for r in rows})
    print(f"{'familia/n':12s} {'N':>3s} | catalogo historico: gap medio vs BKS  #atinge BKS | melhor de todas as fontes: gap medio  #atinge BKS  | gap medio vs UB Gurobi (historico)")
    for fam in fams:
        rr = [r for r in rows if (r[0].split("_")[0] + "_n" + r[0].split("_n")[1].split("_")[0]) == fam]
        gh = [(r[4] - r[2]) / r[2] * 100 for r in rr]; ga = [(r[5] - r[2]) / r[2] * 100 for r in rr]
        gu = [(r[4] - r[3]) / r[3] * 100 for r in rr]
        print(f"{fam:12s} {len(rr):3d} | {st.mean(gh):+8.2f}%  {sum(g >= -1e-9 for g in gh):3d}/{len(rr):2d}               | {st.mean(ga):+8.2f}%  {sum(g >= -1e-9 for g in ga):3d}/{len(rr):2d}                    | {st.mean(gu):+8.2f}%")
    print("\nPor metodo (catalogo historico), gap medio do extremo MaxMin de cada metodo vs BKS, por familia:")
    print(f"{'metodo':16s} " + " ".join(f"{f:>10s}" for f in fams))
    for m in MODES:
        cells = []
        for fam in fams:
            rr = [r for r in rows if (r[0].split("_")[0] + "_n" + r[0].split("_n")[1].split("_")[0]) == fam]
            cells.append(f"{st.mean((r[6][m] - r[2]) / r[2] * 100 for r in rr):+9.1f}%")
        print(f"{m:16s} " + " ".join(cells))


if __name__ == "__main__":
    main()
