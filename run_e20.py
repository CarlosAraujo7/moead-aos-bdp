"""E20 -- MOEA/D memetico/hibrido: repara a fraqueza do MOEA/D no MDP (E11: Tabu/VNS/GRASP ajustados ficam +0,09 a +0,14 acima em hv_norm).

Variantes (todas MOEA/D-Uniform normalizado, orcamento = tempo T do MOEA/D-Uniform padrao [mesmo T do E11], inicializacao contada):
  m_ls     : pool de 8 operadores + `tcheb_ls` (busca local escalarizada ate o otimo local ou 10 trocas; operators.TchebLocalSearch);
  m_hyb    : semeadura por Tabu (parametros ajustados no E11) durante 30% de T, depois o MOEA/D usa o arquivo do Tabu como populacao inicial
             (MOEADConfig.seed_traj_frac);
  m_hyb_ls : as duas coisas.
O MOEA/D-Uniform padrao, o MOEA/D ajustado, Tabu, VNS, GRASP e NSGA-II (ajustados e padrao) JA existem no E11, mesmas instancias de teste, replicas e
referencia CONGELADA; a analise (analyze_e20.py) compara tudo contra elas.

Saida: results/e20_hybrid/<celula>/runs/<instancia>_<modo>_rep<kk>.json (FORA da referencia).

    python run_e20.py [--cells MDP-S MDP-L CDP-S ...] [--workers 8] [--reps 3] [--modes m_ls,m_hyb,m_hyb_ls]
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

OUT = ROOT / "results" / "e20_hybrid"
MODES = {"m_ls": dict(extra=("tcheb_ls",), seed=0.0, traj="ts"), "m_hyb": dict(extra=(), seed=0.3, traj="ts"),
         "m_hyb_ls": dict(extra=("tcheb_ls",), seed=0.3, traj="ts"),
         "m_hyb_vns30": dict(extra=(), seed=0.3, traj="vns"), "m_hyb_vns50": dict(extra=(), seed=0.5, traj="vns"),
         "m_hyb_ts50": dict(extra=(), seed=0.5, traj="ts")}


def _work(args):
    cell, prob, row, rep, T, modes = args
    d = OUT / cell / "runs"
    todo = [m for m in modes if not (d / f"{row['name']}_{m}_rep{rep:02d}.json").exists()]
    if not todo:
        return 0
    d.mkdir(parents=True, exist_ok=True)
    from experiment import budget_for
    from moead_aos import MOEAD_AOS, MOEADConfig
    inst = C.load_instance(prob, row)
    b = budget_for(inst.n)
    tuned_params = {t: json.load(open(C.OUT / "tuning" / cell / f"tuned_{t}.json"))["best"] for t in ("ts", "vns")}
    seed = C.seed_for("e20", row["name"], rep)
    for m in todo:
        spec = MODES[m]
        cfg = MOEADConfig(n_weights=b["T"], neighborhood_size=b["nbr"], max_gen=10**7, operator_selection="random", reward_mode="scalar",
                          normalize_objectives=True, grasp_alpha=C.DEFAULT["moead"][0]["grasp_alpha"], seed=seed, time_limit_s=T,
                          extra_operators=spec["extra"], seed_traj_frac=spec["seed"], seed_traj_params=(dict(tuned_params[spec["traj"]], method=("tabu" if spec["traj"] == "ts" else "vns")) if spec["seed"] > 0 else None))
        t0 = time.perf_counter()
        r = MOEAD_AOS(inst, cfg).run()
        el = time.perf_counter() - t0
        fr = np.asarray(r.pareto_front, float).reshape(-1, 2)
        out = d / f"{row['name']}_{m}_rep{rep:02d}.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(instance=row["name"], family=row["family"], n=inst.n, cell=cell, mode=m, rep=rep, seed=seed, time_target_s=T,
                                       elapsed_s=el, pareto_size=int(len(fr)), pareto_front=fr.tolist(), archive_sizes=r.archive_sizes,
                                       operator_counts=[float(x) for x in r.operator_counts])))
        tmp.replace(out)
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["MDP-S", "MDP-L"])
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    modes = a.modes.split(",")
    allc = C.cells()
    tasks = []
    for cell in a.cells:
        prob = allc[cell]["problem"]
        for row in allc[cell]["test"]:
            for rep in range(a.reps):
                T = R11.target_time(cell, prob, row["name"], rep)
                if T is not None:
                    tasks.append((T, (cell, prob, row, rep, T, modes)))
    tasks.sort(key=lambda t: -t[0])
    print(f"E20: {len(tasks)} tarefas (instancia x replica; modos {modes})", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(_work, [t[1] for t in tasks], chunksize=1), 1):
            if i % 25 == 0 or i == len(tasks):
                print(f"   [{i}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    print("FIM E20", flush=True)


if __name__ == "__main__":
    main()
