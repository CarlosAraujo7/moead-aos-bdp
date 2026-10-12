"""E22 -- os dois metodos de Colmenar, Marti & Duarte (2018) que faltavam (SPEA2 e Iterated Greedy), com a configuracao do irace deles, contra o MOEA/D.

Colmenar et al. comparam seis metodos (NSGA-II, SPEA2, GRASP, Iterated Greedy, Tabu Search, VNS); o E11 tem NSGA-II, GRASP, Tabu e VNS. Aqui SPEA2 (src/spea2.py)
e Iterated Greedy (bo_trajectory.py, method="ig", Algoritmo 8 do artigo) rodam com os parametros da Tabela 6 do artigo (populacao 340 / mutacao 0,14 /
cruzamento 0,15; alpha 0,95, CAltBwS, destruicao de 55,39%), com o limite de tempo T do MOEA/D-Uniform padrao (o mesmo do E11; inicializacao contada) nas
instancias de TESTE do E11, 3 replicas, regime normalizado (so o MOEA/D usa normalizacao). Sem ajuste proprio: sao os valores publicados.

Saida: results/e22_colmenar_rest/<celula>/runs/<instancia>_<modo>_rep<kk>.json (FORA da referencia). Analise: analyze_e22.py.

    python run_e22.py [--cells CDP-S GDP-S MDP-S CDP-L GDP-L MDP-L] [--workers 8] [--reps 3]
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
import run_e11 as R11  # noqa: E402

OUT = ROOT / "results" / "e22_colmenar_rest"
MODES = {"d_spea2": "spea2", "d_ig": "ig"}


def _work(args):
    cell, prob, row, rep, T = args
    d = OUT / cell / "runs"
    todo = [m for m in MODES if not (d / f"{row['name']}_{m}_rep{rep:02d}.json").exists()]
    if not todo:
        return 0
    d.mkdir(parents=True, exist_ok=True)
    inst = C.load_instance(prob, row)
    seed = C.seed_for("e22", row["name"], rep)
    for m in todo:
        fr, el, it = C.run_method(MODES[m], C.DEFAULT[MODES[m]][0], inst, seed, T)
        out = d / f"{row['name']}_{m}_rep{rep:02d}.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(instance=row["name"], family=row["family"], n=inst.n, cell=cell, mode=m, rep=rep, seed=seed, time_target_s=T,
                                       elapsed_s=el, iterations=it, pareto_size=int(len(fr)), pareto_front=np.asarray(fr).tolist())))
        tmp.replace(out)
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["CDP-S", "GDP-S", "MDP-S", "CDP-L", "GDP-L", "MDP-L"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    allc = C.cells()
    tasks = []
    for cell in a.cells:
        prob = allc[cell]["problem"]
        for row in allc[cell]["test"]:
            for rep in range(a.reps):
                T = R11.target_time(cell, prob, row["name"], rep)
                if T is not None:
                    tasks.append((T, (cell, prob, row, rep, T)))
    tasks.sort(key=lambda t: -t[0])
    print(f"E22: {len(tasks)} tarefas (instancia x replica; {len(MODES)} metodos cada)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(_work, [t[1] for t in tasks], chunksize=1), 1):
            if i % 50 == 0 or i == len(tasks):
                print(f"   [{i}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    print("FIM E22", flush=True)


if __name__ == "__main__":
    main()
