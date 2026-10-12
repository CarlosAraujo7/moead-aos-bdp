"""Fracao media de selecao por operador, por variante de seletor (CDP e GDP, n <= 150, 3 replicas; `operator_counts` = selecoes totais).

Serve para as observacoes qualitativas da Secao 6.3/6.4 e do Suplemento S1 (quais operadores cada seletor prefere). Le os runs de
results/capacitated_norm (catalogo) e results/phase2_aos (variantes). Saida: terminal + results/analysis/op_shares.json

    python analyze_op_shares.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from operators import OPERATOR_NAMES  # noqa: E402

CATALOG = ["random", "hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar"]
PHASE2 = ["ucb_insert", "ts_insert", "dyts_insert", "ucb_srel_cost", "fd_replace", "fd_scalar", "fd_hv", "ts_replace", "frr_default", "frr_c03", "px_random",
          "px_ucbhv", "px_ucbins", "px_frr"]


def main():
    out = {}
    for prob in ("CDP", "GDP"):
        names = [r["name"] for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")) if r["problem"] == prob and int(r["n"]) <= 150]
        out[prob] = {}
        print(f"\n=== {prob} (N={len(names)}, 3 replicas): fracao media de selecoes por operador")
        print(f"   {'variante':16s} " + " ".join(f"{n[:9]:>9s}" for n in OPERATOR_NAMES) + "   (extras: restart, half, noop)")
        for v in CATALOG + PHASE2:
            d = ROOT / "results" / ("capacitated_norm" if v in CATALOG else "phase2_aos") / prob / "runs"
            sh = []
            for nm in names:
                for k in range(3):
                    f = d / f"{nm}_{v}_rep{k:02d}.json"
                    if not f.exists():
                        sh = None
                        break
                    c = np.array(json.load(open(f))["operator_counts"], float)
                    sh.append(c / c.sum())
                if sh is None:
                    break
            if not sh:
                continue
            m = np.mean(sh, axis=0)
            out[prob][v] = m.tolist()
            ex = f"   extras {np.round(m[8:], 3).tolist()}" if len(m) > 8 else ""
            print(f"   {v:16s} " + " ".join(f"{x:9.3f}" for x in m[:8]) + ex)
    (ROOT / "results" / "analysis" / "op_shares.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
