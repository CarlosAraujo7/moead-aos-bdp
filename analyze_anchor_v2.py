"""E7: ancoragem com motor SBTS-like (`random_sbts`, `random_both`, `dyts_hv_both`) vs sem ancora e vs a ancora historica (tabu de
destruicao/reconstrucao: `random_anchor` n<=150, `random_anchor6k` n=500), com replicas, em runs reais (results/phase2_aos).

Para cada recorte: N instancias completas; hv_norm medio (referencia fixa), Wilcoxon pareado contra a base e contra a ancora historica
(A12 = P(variante melhor)); MaxMin do extremo / melhor valor da literatura (CDP: anchors_lit.py; GDP: valor publicado do pacote
Zenodo = heuristica de 60 s), por familia; tempo mediano por run. Rode antes: python src/reference_sets.py build

    python analyze_anchor_v2.py --problem CDP|GDP
"""
from __future__ import annotations

import argparse
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
from analyze_phase2 import a12  # noqa: E402

REPS = 3
NORM = {"CDP": ROOT / "results/capacitated_norm/CDP/runs", "GDP": ROOT / "results/capacitated_norm/GDP/runs"}
P2 = {"CDP": ROOT / "results/phase2_aos/CDP/runs", "GDP": ROOT / "results/phase2_aos/GDP/runs"}
SETUPS = [  # (rotulo, n_min, n_max, base (modo, pasta-chave), ancora historica, novas)
    ("n<=150", 0, 150, ("random_ref", "p2"), "random_anchor", ["random_sbts", "random_both", "dyts_hv_both"]),
    ("n=500", 500, 500, ("random", "norm"), "random_anchor6k", ["random_sbts", "random_both"]),
]


def load(d, nm, m, k):
    return json.load(open(d / f"{nm}_{m}_rep{k:02d}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["CDP", "GDP"], default="CDP")
    a = ap.parse_args()
    prob = a.problem
    ref = RS.load(prob)
    lit = None
    if prob == "CDP":
        import anchors_lit as L
        lit = {n: v["best"] for n, v in L.all_references().items()}
    else:
        import anchor_gdp as AG
        lit = {n: v["disp"] for n, v in AG.published().items()}
    out = {}
    for label, lo, hi, (base, bkey), v1, news in SETUPS:
        names = [r["name"] for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv"))
                 if r["problem"] == prob and lo <= int(r["n"]) <= hi]
        dirs = {base: (NORM if bkey == "norm" else P2)[prob], v1: P2[prob], **{v: P2[prob] for v in news}}
        meths = [base, v1] + news
        ok = [n for n in names if all((dirs[m] / f"{n}_{m}_rep{k:02d}.json").exists() for m in meths for k in range(REPS))]
        present = [m for m in meths if all((dirs[m] / f"{n}_{m}_rep{k:02d}.json").exists() for n in names for k in range(REPS))]
        newp = [v for v in news if v in present]
        if len(ok) < 5 or not newp:
            # tenta so com as variantes que ja estao completas
            ok = [n for n in names if all((dirs[m] / f"{n}_{m}_rep{k:02d}.json").exists() for m in [base, v1] + newp for k in range(REPS))] if newp else []
        if len(ok) < 5 or not newp:
            print(f"\n=== {prob} {label}: ainda sem variantes completas ({len(ok)} instancias)"); continue
        meths = [base, v1] + newp
        hv = {m: np.array([np.mean([RS.indicators(np.array(load(dirs[m], n, m, k)["pareto_front"]), ref[n])["hv_norm"] for k in range(REPS)]) for n in ok]) for m in meths}
        ext = lambda m, n: np.mean([max(p[1] for p in load(dirs[m], n, m, k)["pareto_front"]) for k in range(REPS)])
        print(f"\n=== {prob} {label}: N={len(ok)} instancias, {REPS} replicas")
        res = {}
        for m in meths:
            t = st.median(load(dirs[m], n, m, 0)["elapsed_s"] for n in ok)
            row = dict(hv=float(hv[m].mean()), time_median=t)
            if m != base:
                for ref_m in (base, v1):
                    if ref_m == m:
                        continue
                    d = hv[m] - hv[ref_m]
                    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                    row[f"vs_{ref_m}"] = dict(diff=float(d.mean()), better=int((d > 0).sum()), p=float(p), a12=a12(hv[m], hv[ref_m]))
            fams = {}
            for n in ok:
                if n in lit:
                    fams.setdefault(n.split("_")[0] + (" n" + n.split("_n")[1].split("_")[0]), []).append(ext(m, n) / lit[n])
            row["ext_ratio"] = {f: float(np.mean(v)) for f, v in fams.items()}
            row["ext_ratio_N"] = {f: len(v) for f, v in fams.items()}
            res[m] = row
            s = f"  {m:16s} hv_norm {row['hv']:.4f}  tempo med {t:6.1f}s  "
            for ref_m in (base, v1):
                k = f"vs_{ref_m}"
                if k in row:
                    s += f"| vs {ref_m}: {row[k]['diff']:+.4f} ({row[k]['better']}/{len(ok)}, p={row[k]['p']:.3g}) "
            print(s)
            print("      extremo MaxMin / " + ("melhor valor da literatura" if prob == "CDP" else "valor publicado (60 s)") + ": " +
                  ", ".join(f"{f} {r:.3f} (N={row['ext_ratio_N'][f]})" for f, r in sorted(row["ext_ratio"].items())))
        out[label] = res
    Path(ROOT / f"results/analysis/anchor_v2_{prob}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
