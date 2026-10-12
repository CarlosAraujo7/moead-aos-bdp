"""Ponta de MaxMin do CDP contra a literatura ATUALIZADA (Symmetry 2026 Tabela 1 + Lu et al. 2023 Tabelas A.1/A.2), com e sem ancora,
em runs reais com replicas (results/capacitated_norm e results/phase2_aos). Cobre b02 e b03 (Lu et al. publicam as duas).

Para cada grupo familia/n/demanda: razao media (MaxMin do extremo da frente, media das replicas) / referencia, sendo
  best     = max(BKS do Symmetry, melhor valor de Lu et al.)  (melhor valor publicado conhecido por nos)
  sbts_avg = media de 40 execucoes do SBTS_long (300 s) / sbts_short_avg = idem do SBTS_short (10 s), Xeon 2,1 GHz, C++ (Lu et al.)
             (desempenho TIPICO, nao o melhor de 40 execucoes)
e o numero de instancias em que a razao >= 1 (para `best`: tolerancia 1e-9; para `sbts_avg`: >= media).

    python analyze_anchor_lit.py
Saida: tabela no terminal + results/analysis/anchor_lit_cdp.json
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import anchors_lit as L

ROOT = Path(__file__).parent
NORM = ROOT / "results/capacitated_norm/CDP/runs"
P2 = ROOT / "results/phase2_aos/CDP/runs"
# (rotulo, n_min, n_max, reps, [(rotulo_metodo, pasta, modo)])
SETUPS = [
    ("n<=150", 0, 150, 3, [("random (sem ancora)", P2, "random_ref"), ("random + ancora 2000x3", P2, "random_anchor"),
                           ("dyts_hv (sem ancora)", NORM, "dyts_hv"), ("dyts_hv + ancora 2000x3", P2, "dyts_hv_anchor")]),
    ("n=500", 500, 500, 3, [("random (sem ancora)", NORM, "random"), ("random + ancora 6000x3", P2, "random_anchor6k")]),
]


def ext(d, name, mode, reps):
    vals = []
    for k in range(reps):
        f = d / f"{name}_{mode}_rep{k:02d}.json"
        if not f.exists():
            return None
        vals.append(max(p[1] for p in json.load(open(f))["pareto_front"]))
    return st.mean(vals)


def group(name):
    fam = name.split("_")[0]
    n = name.split("_n")[1].split("_")[0]
    tag = name.split("_")[-2]
    return f"{fam} n{n} {tag}"


def main():
    cat = [r["name"] for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["problem"] == "CDP"]
    ref = L.all_references()
    out = {}
    for label, lo, hi, reps, methods in SETUPS:
        names = [r["name"] for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv"))
                 if r["problem"] == "CDP" and lo <= int(r["n"]) <= hi]
        print(f"\n=== CDP {label}: {len(names)} instancias, {reps} replicas (MaxMin do extremo / referencia da literatura)")
        groups = sorted({group(n) for n in names})
        print(f"{'grupo':22s} {'N':>2s} | " + " | ".join(f"{m[0]:32s}" for m in methods))
        print(f"{'':22s} {'':>2s} | " + " | ".join(f"{'/best (>=best) /sh10 /lg300 /sota':32s}" for _ in methods))
        for g in groups:
            nn = [n for n in names if group(n) == g]
            cells = []
            for mlabel, d, mode in methods:
                vals = [(ext(d, n, mode, reps), ref[n]) for n in nn]
                vals = [(e, r) for e, r in vals if e is not None]
                if not vals:
                    cells.append(f"{'n/d':32s}")
                    continue
                rb = st.mean(e / r["best"] for e, r in vals)
                rs = st.mean(e / r["sbts_avg"] for e, r in vals); rsh = st.mean(e / r["sbts_short_avg"] for e, r in vals); rso = st.mean(e / r["sota_avg"] for e, r in vals)
                ab = sum(e >= r["best"] - 1e-9 for e, r in vals)
                out.setdefault(label, {}).setdefault(g, {})[mlabel] = dict(N=len(vals), ratio_best=rb, ratio_sbts_avg=rs, ratio_sbts_short_avg=rsh, ratio_sota_avg=rso, reach_best=ab)
                cells.append(f"{rb:5.3f} ({ab:2d}/{len(vals):2d}) {rsh:5.3f} {rs:5.3f} {rso:5.3f}")
            print(f"{g:22s} {len(nn):2d} | " + " | ".join(cells))
        # agregados por familia (b02+b03 juntos)
        print("  -- por familia (b02 + b03):")
        fams = sorted({" ".join(g.split()[:2]) for g in groups})
        for fm in fams:
            nn = [n for n in names if " ".join(group(n).split()[:2]) == fm]
            cells = []
            for mlabel, d, mode in methods:
                vals = [(ext(d, n, mode, reps), ref[n]) for n in nn]
                vals = [(e, r) for e, r in vals if e is not None]
                if not vals:
                    cells.append(f"{'n/d':32s}")
                    continue
                rb = st.mean(e / r["best"] for e, r in vals)
                rs = st.mean(e / r["sbts_avg"] for e, r in vals); rsh = st.mean(e / r["sbts_short_avg"] for e, r in vals); rso = st.mean(e / r["sota_avg"] for e, r in vals)
                ab = sum(e >= r["best"] - 1e-9 for e, r in vals)
                out.setdefault(label, {}).setdefault("FAM " + fm, {})[mlabel] = dict(N=len(vals), ratio_best=rb, ratio_sbts_avg=rs, ratio_sbts_short_avg=rsh, ratio_sota_avg=rso, reach_best=ab)
                cells.append(f"{rb:5.3f} ({ab:2d}/{len(vals):2d}) {rsh:5.3f} {rs:5.3f} {rso:5.3f}")
            print(f"{fm:22s} {len(nn):2d} | " + " | ".join(cells))
    (ROOT / "results/analysis/anchor_lit_cdp.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nsalvo: results/analysis/anchor_lit_cdp.json")



MODES = ["hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar", "random",
         "nsga2", "grasp", "grasp_pr", "vns_pareto", "vns_tchebycheff"]


def best_of_12(d, name, reps):
    """Maior (sobre os 12 metodos) da media sobre replicas do extremo MaxMin; ignora metodos sem todos os arquivos."""
    vals = [v for v in (ext(d, name, m, reps) for m in MODES) if v is not None]
    return max(vals) if vals else None


def regimes():
    """Melhor dos 12 metodos, regime historico (catalogo bruto, 1 replica) e regime normalizado (3 replicas), vs referencia."""
    ref = L.all_references()
    rows = [r for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["problem"] == "CDP"]
    hist = ROOT / "results/capacitated/CDP/runs"
    print("\n=== Melhor dos 12 metodos: extremo MaxMin / melhor valor publicado (b02+b03 juntos)")
    print(f"{'familia':12s} {'N':>2s} | {'historico (bruto, 1 rep)':26s} | {'normalizado (rep00)':26s} | {'normalizado (media 3 reps)':26s} | uniforme normalizado (media 3 reps)")
    res = {}
    for fm in sorted({(r['family'], r['n']) for r in rows}):
        nn = [r["name"] for r in rows if (r["family"], r["n"]) == fm]
        cells = []
        for d, reps in ((hist, 1), (NORM, 1), (NORM, 3)):
            vals = [(best_of_12(d, n, reps), ref[n]) for n in nn]
            vals = [(e, r) for e, r in vals if e is not None]
            rb = st.mean(e / r["best"] for e, r in vals); ab = sum(e >= r["best"] - 1e-9 for e, r in vals)
            cells.append(f"{rb:5.3f} ({ab:2d}/{len(vals):2d} >= best)")
        vals = [(ext(NORM, n, "random", 3), ref[n]) for n in nn]
        vals = [(e, r) for e, r in vals if e is not None]
        cells.append(f"{st.mean(e / r['best'] for e, r in vals):5.3f}")
        res["%s n%s" % fm] = cells
        print(f"{fm[0] + ' n' + fm[1]:12s} {len(nn):2d} | " + " | ".join(f"{c:26s}" for c in cells))
    return res


if __name__ == "__main__":
    main()
    regimes()
