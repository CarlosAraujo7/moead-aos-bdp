"""E13: o tamanho da solucao como dimensao escondida (run_e13.py). Resume, por celula, o Tabu ajustado SEM e COM insercoes dominantes.

    python analyze_e13.py
Saida: terminal + results/analysis/e13_size_growth.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "e13_size_growth"


def main():
    res = {}
    for cell in ("CDP-S", "CDP-L", "GDP-S", "GDP-L"):
        rows = [json.load(open(f)) for f in sorted((OUT / cell).glob("*.json"))] if (OUT / cell).exists() else []
        if not rows:
            continue
        # instancias com referencia degenerada (hv_norm explode) ficam fora
        rows = [r for r in rows if r["nogrow"]["hv_norm"] < 5 and r["grow"]["hv_norm"] < 5]
        g = lambda tag, k: np.array([r[tag][k] for r in rows])
        res[cell] = dict(N=len(rows),
                         hv_nogrow=float(g("nogrow", "hv_norm").mean()), hv_grow=float(g("grow", "hv_norm").mean()),
                         maxsum_nogrow=float(g("nogrow", "maxsum_ratio").mean()), maxsum_grow=float(np.median(g("grow", "maxsum_ratio"))),
                         maxmin_nogrow=float(g("nogrow", "maxmin_ratio").mean()), maxmin_grow=float(g("grow", "maxmin_ratio").mean()),
                         size_nogrow=float(np.median(g("nogrow", "median_size"))), size_grow=float(np.median(g("grow", "median_size"))),
                         max_size_grow=float(np.median(g("grow", "max_size"))),
                         ref_dominated_nogrow=float(g("nogrow", "ref_dominated").mean()), ref_dominated_grow=float(g("grow", "ref_dominated").mean()),
                         better=int((g("grow", "hv_norm") > g("nogrow", "hv_norm")).sum()))
        r = res[cell]
        print(f"{cell}: N={r['N']}  hv_norm sem/com insercao {r['hv_nogrow']:.3f}/{r['hv_grow']:.3f} (melhor em {r['better']}/{r['N']});  max MaxSum/ref {r['maxsum_nogrow']:.2f}/{r['maxsum_grow']:.2f} (mediana);  "
              f"tamanho mediano {r['size_nogrow']:.0f}/{r['size_grow']:.0f} (max {r['max_size_grow']:.0f});  pontos da referencia dominados {r['ref_dominated_nogrow']:.2f}/{r['ref_dominated_grow']:.2f}")
    (ROOT / "results" / "analysis" / "e13_size_growth.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
