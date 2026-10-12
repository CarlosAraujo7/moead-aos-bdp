"""E5: ablacao do orcamento do GDP (restricao de custo relaxada a infinito) no regime NORMALIZADO, com 3 replicas, 12 metodos.

Substitui a ablacao antiga (regime bruto, 1 replica; results/ablation_gdp_relaxed). Para o GDP n<=150 (todas as 120 instancias) e uma amostra de
10 instancias de n=500. A condicao original e results/capacitated_norm/GDP (mesmas instancias, replicas e sementes: run_one usa
seed_for(instancia, modo, replica), logo as duas condicoes compartilham numeros aleatorios).
Grava em results/e5_budget_relaxed_norm/GDP/runs/ -- FORA de reference_sets.source_dirs (as solucoes relaxadas podem violar o orcamento).
Comparar com analyze_e5.py (referencia propria de cada regime). Retomavel.

    python run_e5.py [--workers 8] [--reps 3]
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
OUT = ROOT / "results" / "e5_budget_relaxed_norm" / "GDP" / "runs"


def _work(args):
    name, family, filepath, mode, rep = args
    out = OUT / f"{name}_{mode}_rep{rep:02d}.json"
    if out.exists():
        return name, mode, rep, "skip"
    from capacitated_problem import GDPInstance
    from experiment import ExperimentConfig, MODE_LABEL, run_one, seed_for

    inst = GDPInstance(ROOT / "instances" / filepath)
    inst.family, inst.name = family, name
    inst.max_budget = float("inf")          # unica diferenca em relacao ao GDP original
    pf, elapsed, hv_history, op_counts, op_qvals = run_one(inst, ExperimentConfig(normalize_objectives=True), mode, rep)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "instance": name, "family": family, "n": inst.n, "mode": mode, "label": MODE_LABEL.get(mode, mode), "rep": rep,
        "seed": seed_for(name, mode, rep), "elapsed_s": round(elapsed, 4), "pareto_size": len(pf),
        "pareto_front": pf.tolist(), "budget_relaxed": True, "normalize_objectives": True}))
    tmp.replace(out)
    return name, mode, rep, f"{elapsed:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    from experiment import ALL_MODES
    cat = [r for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")) if r["problem"] == "GDP"]
    small = [r for r in cat if int(r["n"]) <= 150]
    # n=500: [::8] pega so a combinacao (b02, k02) (a mais apertada); 7 out: [3::8] acrescenta 10 com (b03, k03) (a mais folgada)
    big = [r for r in cat if int(r["n"]) == 500][::8] + [r for r in cat if int(r["n"]) == 500][3::8]
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(int(r["n"]), (r["name"], r["family"], r["filepath"], m, k)) for k in range(a.reps) for r in small + big for m in ALL_MODES]
    tasks.sort(key=lambda t: -t[0])
    tasks = [t[1] for t in tasks]
    print(f"{len(small)} + {len(big)} instancias GDP x {len(ALL_MODES)} modos x {a.reps} replicas = {len(tasks)} tarefas", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, mode, rep, st in ex.map(_work, tasks, chunksize=1):
            done += 1
            if done % 100 == 0:
                print(f"[{done}/{len(tasks)}] {name} {mode} rep{rep} {st}", flush=True)
    print("FIM E5", flush=True)


if __name__ == "__main__":
    main()
