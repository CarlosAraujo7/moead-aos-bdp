"""E8: ablacao de operadores COM REPLICAS em CDP e GDP (n <= 150): operador unico (op0..op7) e leave-one-out do pool (no0..no7).

Estende a ablacao de 1 out (CDP, 1 replica; run_ablation_ops.py) para CDP + GDP, 3 replicas, mesmas sementes do regime principal. Regime
normalizado, selecao uniforme. A referencia de comparacao e o pool completo `random` ja existente em results/capacitated_norm (mesmas instancias
e replicas). Saida: results/e8_operator_ablation/<PROBLEMA>/runs/<inst>_<tag>_rep<k>.json (FORA da referencia congelada). Idempotente.
Analise: analyze_e8.py.

    python run_e8.py [--workers 8] [--reps 3] [--problems CDP,GDP]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "e8_operator_ablation"
K = 8


def _work(args):
    problem, name, family, filepath, tag, rep = args
    out = OUT / problem / "runs" / f"{name}_{tag}_rep{rep:02d}.json"
    if out.exists():
        return name, tag, rep, "skip"
    from capacitated_problem import CDPInstance, GDPInstance
    from experiment import budget_for, seed_for
    from moead_aos import MOEAD_AOS, MOEADConfig
    from operators import OPERATOR_NAMES

    k = int(tag[2:])
    loo = tag.startswith("no")
    inst = (CDPInstance if problem == "CDP" else GDPInstance)(ROOT / "instances" / filepath)
    inst.family, inst.name = family, name
    b = budget_for(inst.n)
    cfg = MOEADConfig(n_weights=b["T"], neighborhood_size=b["nbr"], max_gen=b["max_gen"], operator_selection="random", reward_mode="scalar",
                      seed=seed_for(name, tag, rep), normalize_objectives=True,
                      fixed_operator=None if loo else k, excluded_operators=(k,) if loo else ())
    t0 = time.perf_counter()
    res = MOEAD_AOS(inst, cfg).run()
    el = time.perf_counter() - t0
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"instance": name, "family": family, "n": inst.n, "mode": tag, "operator": OPERATOR_NAMES[k], "rep": rep,
                               "seed": cfg.seed, "elapsed_s": round(el, 4), "pareto_size": len(res.pareto_front),
                               "pareto_front": res.pareto_front.tolist()}))
    tmp.replace(out)
    return name, tag, rep, f"{el:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--problems", default="CDP,GDP")
    ap.add_argument("--max-n", type=int, default=150)
    a = ap.parse_args()
    cat = list(csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")))
    tasks = []
    for k in range(a.reps):                                   # rep-major: resultados parciais ja sao utilizaveis
        for prob in a.problems.split(","):
            (OUT / prob / "runs").mkdir(parents=True, exist_ok=True)
            for r in sorted((r for r in cat if r["problem"] == prob and int(r["n"]) <= a.max_n), key=lambda r: -int(r["n"])):
                for pre in ("op", "no"):
                    for j in range(K):
                        tasks.append((prob, r["name"], r["family"], r["filepath"], f"{pre}{j}", k))
    print(f"{len(tasks)} tarefas", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, tag, k, st in ex.map(_work, tasks, chunksize=2):
            done += 1
            if done % 100 == 0 or (st != "skip" and done < 5):
                print(f"[{done}/{len(tasks)}] {name} {tag} rep{k} {st}", flush=True)
    print("FIM E8", flush=True)


if __name__ == "__main__":
    main()
