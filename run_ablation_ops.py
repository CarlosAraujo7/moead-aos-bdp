"""Ablacao de operador unico (CLAUDE.md, secao 2g): MOEA/D com UM operador fixo, regime normalizado.

Testa se o pool diverso importa mais que a estrategia de selecao (hipotese da secao 10) e
identifica quais operadores carregam o desempenho. Comparar com os 7 modos normalizados de
results/ablation_normalized/ (inclui `random` = pool uniforme). Grava em
results/ablation_single_operator/<PROBLEMA>/runs/<inst>_op<k>_rep00.json. Retomavel.

    python run_ablation_ops.py --problem CDP --max-n 150 --workers 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))


def _work(args):
    problem, name, family, filepath, k, outdir, loo = args
    tag = f"no{k}" if loo else f"op{k}"
    out = Path(outdir) / f"{name}_{tag}_rep00.json"
    if out.exists():
        return name, k, "skip"
    import time
    from capacitated_problem import CDPInstance, GDPInstance
    from experiment import budget_for, seed_for
    from moead_aos import MOEAD_AOS, MOEADConfig
    from operators import OPERATOR_NAMES

    inst = (CDPInstance if problem == "CDP" else GDPInstance)(ROOT / "instances" / filepath)
    inst.family, inst.name = family, name
    b = budget_for(inst.n)
    cfg = MOEADConfig(n_weights=b["T"], neighborhood_size=b["nbr"], max_gen=b["max_gen"],
                      operator_selection="random", reward_mode="scalar",
                      seed=seed_for(name, tag, 0), normalize_objectives=True,
                      fixed_operator=None if loo else k,
                      excluded_operators=(k,) if loo else ())
    t0 = time.perf_counter()
    res = MOEAD_AOS(inst, cfg).run()
    el = time.perf_counter() - t0
    out.write_text(json.dumps({
        "instance": name, "family": family, "n": inst.n, "mode": tag,
        "operator": OPERATOR_NAMES[k], "rep": 0, "elapsed_s": round(el, 4),
        "pareto_size": len(res.pareto_front), "pareto_front": res.pareto_front.tolist(),
        "normalize_objectives": True,
    }))
    return name, k, f"{el:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["CDP", "GDP"], default="CDP")
    ap.add_argument("--max-n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--loo", action="store_true", help="leave-one-out: pool uniforme sem o operador k (em vez de operador unico k)")
    a = ap.parse_args()
    from operators import OPERATOR_NAMES
    rows = [r for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv"))
            if r["problem"] == a.problem and int(r["n"]) <= a.max_n]
    outdir = ROOT / "results" / ("ablation_pool_loo" if a.loo else "ablation_single_operator") / a.problem / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = [(a.problem, r["name"], r["family"], r["filepath"], k, str(outdir), a.loo)
             for r in sorted(rows, key=lambda r: -int(r["n"])) for k in range(len(OPERATOR_NAMES))]
    print(f"{len(rows)} instancias {a.problem} x {len(OPERATOR_NAMES)} operadores = {len(tasks)} tarefas", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, k, status in ex.map(_work, tasks, chunksize=1):
            done += 1
            if done % 10 == 0 or status != "skip":
                print(f"[{done}/{len(tasks)}] {name} k={k} {status}", flush=True)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
