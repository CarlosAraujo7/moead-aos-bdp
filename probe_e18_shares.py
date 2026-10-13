"""Sonda do E18: participacao de selecao por operador de UCB-HV e GTS-HV com o pool de 10 operadores nas celulas de n=500 (os runs de L do E18 nao
gravaram `operator_counts`). Mesmas instancias de teste (as 10 primeiras de cada celula, espacadas), replica 0, mesmo T e mesma semente do E18, problema
livre e limitado (rho = 2). Saida: results/e18_size/shares_L.json.

    python probe_e18_shares.py [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import e11_common as C  # noqa: E402
import run_e18 as R  # noqa: E402

OPS = ["RandomSwap1x1", "BestSwap1x1", "RandomSwap2x2", "GraspGreedy", "BiObjBestSwap", "MaxMinSwap", "LargePerturbation", "SubsetCrossover", "BiObjGrow", "BottleneckShrink"]


def _work(args):
    cell, prob, row, exp, method, T = args
    inst = C.load_instance(prob, row)
    if R.RHO[exp]:
        inst.set_max_size(R.RHO[exp])
    seed = C.seed_for("e18", row["name"], 0)
    _, _, _, counts = R._run(method, cell, inst, seed, T)
    c = np.array(counts, float)
    return dict(cell=cell, instance=row["name"], exp=exp, method=method, shares=(c / c.sum()).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    allc = C.cells()
    tasks = []
    for cell in ("CDP-L", "GDP-L"):
        prob = allc[cell]["problem"]
        test = allc[cell]["test"]
        for row in test[:: max(1, len(test) // 10)][:10]:
            T = C.e4_time(prob, row["name"], 0)
            for exp in ("free", "bound2"):
                for m in ("moead_size_ucbhv", "moead_size_gtshv"):
                    tasks.append((cell, prob, row, exp, m, T))
    out = ROOT / "results" / "e18_size" / "shares_L.json"
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_work, tasks, chunksize=1):
            res.append(r)
    agg = {}
    for r in res:
        agg.setdefault(f"{r['exp']}/{r['cell']}/{r['method']}", []).append(r["shares"])
    summary = {k: dict(N=len(v), shares=dict(zip(OPS, np.mean(v, axis=0).tolist()))) for k, v in agg.items()}
    out.write_text(json.dumps(dict(runs=res, summary=summary), indent=1))
    for k, v in summary.items():
        top = sorted(v["shares"].items(), key=lambda t: -t[1])[:4]
        print(k, "N", v["N"], " ".join(f"{n[:10]}={s:.2f}" for n, s in top), flush=True)
    print("FIM PROBE", flush=True)


if __name__ == "__main__":
    main()
