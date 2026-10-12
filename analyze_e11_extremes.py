"""E11: onde esta a diferenca entre o MOEA/D e os metodos de trajetoria? Cobertura dos EXTREMOS da frente (MaxSum e MaxMin).

Por celula e metodo (tempo T; ajustados + MOEA/D padrao): media sobre replicas, por instancia, de max f1 / (max f1 da referencia congelada) e de
max f2 / (max f2 da referencia), mais a fracao de instancias em que o extremo atinge >= 0.999 do da referencia e o tamanho mediano da frente.
Instancias com extremo de referencia nulo (MaxMin = 0 em toda a frente) ficam fora da razao de MaxMin.

    python analyze_e11_extremes.py
Saida: terminal + results/analysis/e11_extremes.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import e11_common as C  # noqa: E402
import reference_sets as RS  # noqa: E402
from analyze_e11 import LAB, path  # noqa: E402

MODES = ["uniform", "t_moead", "def_ts", "def_vns", "t_nsga2", "t_grasp", "t_ts", "t_vns"]


def main():
    allc = C.cells()
    out = {}
    for cell in ("MDP-S", "MDP-L", "CDP-S", "CDP-L", "GDP-S", "GDP-L"):
        prob = allc[cell]["problem"]
        ref = RS.load(prob)
        names = [r["name"] for r in allc[cell]["test"]]
        res = {}
        for mode in MODES:
            r1, r2, sz, n_ok = [], [], [], 0
            hit1, hit2 = [], []
            for nm in names:
                rf = np.asarray(ref[nm]["front"], float).reshape(-1, 2)
                m1, m2 = rf[:, 0].max(), rf[:, 1].max()
                a1, a2, ss = [], [], []
                for k in (0, 1, 2):
                    p = path(cell, prob, nm, mode, k)
                    if not p.exists():
                        a1 = None
                        break
                    f = np.array(json.load(open(p))["pareto_front"], float).reshape(-1, 2)
                    a1.append(f[:, 0].max() / m1)
                    a2.append(f[:, 1].max() / m2 if m2 > 1e-9 else np.nan)
                    ss.append(len(f))
                if a1 is None:
                    continue
                n_ok += 1
                r1.append(np.mean(a1)); r2.append(np.nanmean(a2) if m2 > 1e-9 else np.nan); sz.append(np.mean(ss))
            if n_ok:
                r1, r2 = np.array(r1), np.array(r2)
                res[mode] = dict(N=n_ok, maxsum=float(np.mean(r1)), maxsum_hit=float(np.mean(r1 >= 0.999)),
                                 maxmin=float(np.nanmean(r2)), maxmin_hit=float(np.nanmean(r2[~np.isnan(r2)] >= 0.999)),
                                 front_size=float(np.median(sz)))
        out[cell] = res
        print(f"\n=== {cell} (extremos relativos a referencia congelada; media sobre instancias; 'hit' = fracao >= 0.999)")
        print(f"   {'metodo':26s} {'N':>3s} {'MaxSum':>7s} {'hit':>5s} {'MaxMin':>7s} {'hit':>5s} {'|PF| med':>8s}")
        for mode, d in res.items():
            print(f"   {LAB[mode]:26s} {d['N']:3d} {d['maxsum']:7.3f} {d['maxsum_hit']:5.2f} {d['maxmin']:7.3f} {d['maxmin_hit']:5.2f} {d['front_size']:8.1f}")
    (ROOT / "results" / "analysis" / "e11_extremes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
