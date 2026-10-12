"""E18 -- o tamanho do conjunto como dimensao explorada: (A) operadores que aumentam/reduzem o conjunto e (B) a variante do CDP/GDP com
limite de cardinalidade.

Motivacao (CLAUDE.md §2s, E13): no CDP/GDP o MaxSum e monotono no tamanho do conjunto; inserir um elemento a distancia >= MaxMin dos membros
DOMINA. O pool de 8 operadores troca 1-por-1 e so cresce o conjunto por deriva do reparo (sondado em 11 out: o arquivo final do MOEA/D tem 2-3x o
tamanho da solucao gulosa minima no CDP). Este experimento torna a dimensao explicita:

  exp "free"  : problema ORIGINAL (tamanho livre). MOEA/D com o pool de 8 (moead) e com +2 operadores de tamanho (moead_size: biobj_grow,
                bottleneck_shrink); Tabu/VNS de Colmenar ajustados (E11) sem/com insercoes dominantes (ts, ts_add, vns, vns_add); versoes ajustadas do
                MOEA/D (moead_t, moead_size_t).
  exp "bound2": variante com LIMITE DE CARDINALIDADE |S| <= ceil(2.0 * k_ref), k_ref = tamanho da solucao gulosa viavel por eficiencia
                (CapacitatedInstance.set_max_size). Problema bem posto: o MaxSum deixa de ser ilimitado. Mesmos metodos + NSGA-II e GRASP ajustados.
  exp "bound15": idem com rho = 1.5 (so celulas S; sensibilidade).

Todos os metodos rodam com limite de tempo T = tempo do MOEA/D-Uniform padrao (E4), contando a inicializacao (mesmo protocolo do E11), 3 replicas
(mesmas instancias de TESTE do E11). Saida: results/e18_size/<exp>/<celula>/runs/<instancia>_<metodo>_rep<kk>.json (FORA da referencia congelada).
Analise: analyze_e18.py.

    python run_e18.py --exp free|bound2|bound15 [--cells CDP-S GDP-S ...] [--workers 8] [--reps 3]
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

OUT = ROOT / "results" / "e18_size"
RHO = {"free": None, "bound2": 2.0, "bound15": 1.5}
METHODS = {
    "free": ["moead", "moead_size", "moead_t", "moead_size_t", "ts", "ts_add", "vns", "vns_add"],
    "bound2": ["moead", "moead_size", "moead_t", "moead_size_t", "ts", "ts_add", "vns", "vns_add", "nsga2", "grasp"],
    "bound15": ["moead", "moead_size", "moead_t", "moead_size_t", "ts", "ts_add", "vns", "vns_add", "nsga2", "grasp"],
}
SIZE_OPS = ("biobj_grow", "bottleneck_shrink")


def tuned(cell: str, m: str) -> dict:
    f = C.OUT / "tuning" / cell / f"tuned_{m}.json"
    return json.load(open(f))["best"] if f.exists() else C.DEFAULT[m][0]


def _run(method: str, cell: str, inst, seed: int, T: float):
    """Devolve (frente Nx2, tempo, tamanhos dos membros do arquivo ou None)."""
    from experiment import budget_for
    t0 = time.perf_counter()
    sizes = None
    if method.startswith("moead"):
        from moead_aos import MOEAD_AOS, MOEADConfig
        p = tuned(cell, "moead") if method.endswith("_t") else C.DEFAULT["moead"][0]
        b = budget_for(inst.n)
        r = MOEAD_AOS(inst, MOEADConfig(n_weights=int(p.get("n_weights") or b["T"]), neighborhood_size=int(p.get("nbr") or b["nbr"]), max_gen=10**7,
                                        operator_selection="random", reward_mode="scalar", normalize_objectives=True, grasp_alpha=p["grasp_alpha"],
                                        seed=seed, time_limit_s=T, extra_operators=(SIZE_OPS if "size" in method else ()))).run()
        fr, sizes = np.asarray(r.pareto_front, float).reshape(-1, 2), r.archive_sizes
    elif method in ("ts", "ts_add", "vns", "vns_add"):
        from bo_trajectory import BOTrajConfig, BOTrajectory
        base = "ts" if method.startswith("ts") else "vns"
        params = dict(tuned(cell, base))
        alg = BOTrajectory(inst, BOTrajConfig(method=("tabu" if base == "ts" else "vns"), seed=seed, time_limit_s=T,
                                              add_moves=method.endswith("_add"), **params))
        r = alg.run()
        fr, sizes = np.asarray(r.pareto_front, float).reshape(-1, 2), [int(len(s)) for s in alg.arch_s]
    else:
        fr, _, _ = C.run_method(method, tuned(cell, method), inst, seed, T)
    return fr, time.perf_counter() - t0, sizes


def _work(args):
    exp, cell, prob, row, rep, T = args
    d = OUT / exp / cell / "runs"
    todo = [m for m in METHODS[exp] if not (d / f"{row['name']}_{m}_rep{rep:02d}.json").exists()]
    if not todo:
        return 0
    d.mkdir(parents=True, exist_ok=True)
    inst = C.load_instance(prob, row)
    ms = inst.set_max_size(RHO[exp]) if RHO[exp] else None
    seed = C.seed_for("e18", row["name"], rep)
    for m in todo:
        fr, el, sizes = _run(m, cell, inst, seed, T)
        out = d / f"{row['name']}_{m}_rep{rep:02d}.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(instance=row["name"], family=row["family"], n=inst.n, exp=exp, cell=cell, mode=m, rep=rep, seed=seed,
                                       time_target_s=T, elapsed_s=el, max_size=ms, pareto_size=int(len(fr)), pareto_front=fr.tolist(),
                                       archive_sizes=sizes)))
        tmp.replace(out)
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(RHO))
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    allc = C.cells()
    cells = a.cells or (["CDP-S", "GDP-S"] if a.exp == "bound15" else ["CDP-S", "GDP-S", "CDP-L", "GDP-L"])
    tasks = []
    for cell in cells:
        prob = allc[cell]["problem"]
        for row in allc[cell]["test"]:
            for rep in range(a.reps):
                T = C.e4_time(prob, row["name"], rep)
                if T is None:
                    continue
                tasks.append((T, (a.exp, cell, prob, row, rep, T)))
    tasks.sort(key=lambda t: -t[0])
    print(f"E18 {a.exp}: {len(tasks)} tarefas (instancia x replica; {len(METHODS[a.exp])} metodos cada)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(_work, [t[1] for t in tasks], chunksize=1), 1):
            if i % 50 == 0 or i == len(tasks):
                print(f"   [{i}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    print(f"FIM E18 {a.exp}", flush=True)


if __name__ == "__main__":
    main()
