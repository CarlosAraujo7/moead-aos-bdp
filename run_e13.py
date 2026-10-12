"""E13 -- o tamanho da solucao como dimensao escondida: insercao dominante no CDP/GDP.

No CDP/GDP o MaxSum e monotono no tamanho do conjunto (distancias >= 0): inserir um elemento a distancia >= MaxMin de todos os membros nao
reduz o MaxMin e sobe o MaxSum, ou seja, DOMINA a solucao. Os metodos do estudo (MOEA/D, NSGA-II, GRASP, VNS e o TS/VNS do E11 sem insercoes)
trabalham no regime de tamanho dirigido pela capacidade: constroem ate atingir a capacidade, trocam 1-por-1 e so reparam o necessario. Esta sonda
roda o Tabu/VNS ajustado do E11 com a busca local de MaxSum aceitando insercoes dominantes (`add_moves=True`) e mede quanto a frente de referencia
congelada (e a do MOEA/D) fica abaixo do que e alcancavel crescendo o conjunto.

Por celula (CDP-S, GDP-S, CDP-L, GDP-L): 12 instancias de teste espacadas, replica 0, limite de tempo T (o do MOEA/D-Uniform, via E4).
Saida: results/e13_size_growth/<celula>/<instancia>.json (FORA da referencia). Analise: analyze_e13.py.

    python run_e13.py [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import e11_common as C  # noqa: E402

OUT = ROOT / "results" / "e13_size_growth"
CELLS = ["CDP-S", "GDP-S", "CDP-L", "GDP-L"]


def _work(args):
    cell, prob, row, T, params_ts = args
    out = OUT / cell / f"{row['name']}.json"
    if out.exists():
        return str(out)
    from bo_trajectory import BOTrajConfig, BOTrajectory
    import reference_sets as RS
    inst = C.load_instance(prob, row)
    ref = RS.load(prob)[row["name"]]
    res = {}
    for tag, add in (("nogrow", False), ("grow", True)):
        alg = BOTrajectory(inst, BOTrajConfig(method="tabu", seed=C.seed_for("e13", row["name"], tag), time_limit_s=T, add_moves=add, **params_ts))
        t0 = time.perf_counter()
        r = alg.run()
        el = time.perf_counter() - t0
        fr = np.asarray(r.pareto_front, float).reshape(-1, 2)
        sizes = [int(len(s)) for s in alg.arch_s]
        rf = np.asarray(ref["front"], float)
        # fracao dos pontos da referencia dominados por algum ponto desta frente
        dom = np.mean([bool(np.any((fr[:, 0] >= p[0]) & (fr[:, 1] >= p[1]) & ((fr[:, 0] > p[0]) | (fr[:, 1] > p[1])))) for p in rf])
        res[tag] = dict(hv_norm=C.hv_of(fr, ref), maxsum_ratio=float(fr[:, 0].max() / rf[:, 0].max()), maxmin_ratio=float(fr[:, 1].max() / max(rf[:, 1].max(), 1e-12)),
                        median_size=float(np.median(sizes)), max_size=int(max(sizes)), ref_dominated=float(dom), elapsed_s=el, pareto_size=int(len(fr)))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(cell=cell, instance=row["name"], n=inst.n, T=T, params=params_ts, **res)))
    tmp.replace(out)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    allc = C.cells()
    tasks = []
    for cell in CELLS:
        prob = allc[cell]["problem"]
        f = C.OUT / "tuning" / cell / "tuned_ts.json"
        if not f.exists():
            print(f"[aviso] sem ajuste do TS em {cell}", flush=True)
            continue
        params = json.load(open(f))["best"]
        test = allc[cell]["test"]
        for row in test[:: max(1, len(test) // 12)][:12]:
            T = C.e4_time(prob, row["name"], 0)
            if T is not None:
                tasks.append((T, (cell, prob, row, T, params)))
    tasks.sort(key=lambda t: -t[0])
    print(f"{len(tasks)} instancias", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(_work, [t[1] for t in tasks], chunksize=1), 1):
            if i % 10 == 0:
                print(f"   [{i}/{len(tasks)}]", flush=True)
    print("FIM E13", flush=True)


if __name__ == "__main__":
    main()
