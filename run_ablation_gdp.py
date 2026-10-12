"""Ablacao CDP-vs-GDP (CLAUDE.md, secao 10): GDP com a restricao de orcamento de custo relaxada.

Hipotese testada: a 2a restricao do GDP (orcamento) penaliza metodos construtivos nao
adaptativos (GRASP-MO/GRASP-PR). Se for isso, ao tornar o orcamento nao-ativo
(max_budget = inf, restando so a capacidade minima, como no CDP) GRASP deve voltar
ao grupo do topo. A condicao original ja existe em results/capacitated/GDP (rep00).

Grava em results/ablation_gdp_relaxed/runs/ (nao toca no catalogo). Retomavel: pula
(instancia, modo) ja gravados.

    python run_ablation_gdp.py --max-n 150 --workers 10
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

OUT = ROOT / "results" / "ablation_gdp_relaxed" / "runs"


def _work(args):
    name, family, filepath, mode = args
    out = OUT / f"{name}_{mode}_rep00.json"
    if out.exists():
        return name, mode, "skip"
    from capacitated_problem import GDPInstance
    from experiment import ExperimentConfig, MODE_LABEL, run_one, seed_for

    inst = GDPInstance(ROOT / "instances" / filepath)
    inst.family, inst.name = family, name
    inst.max_budget = float("inf")          # <- a unica diferenca em relacao ao GDP original
    pf, elapsed, hv_history, op_counts, op_qvals = run_one(inst, ExperimentConfig(), mode, 0)
    out.write_text(json.dumps({
        "instance": name, "family": family, "n": inst.n, "mode": mode,
        "label": MODE_LABEL.get(mode, mode), "rep": 0,
        "seed": seed_for(name, mode, 0), "elapsed_s": round(elapsed, 4),
        "pareto_size": len(pf), "pareto_front": pf.tolist(),
        "budget_relaxed": True,
    }))
    return name, mode, f"{elapsed:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--reverse", action="store_true", help="percorre as tarefas em ordem inversa (2o processo)")
    a = ap.parse_args()

    from experiment import ALL_MODES
    rows = [r for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv"))
            if r["problem"] == "GDP" and int(r["n"]) <= a.max_n]
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(r["name"], r["family"], r["filepath"], m) for r in rows for m in ALL_MODES]
    if a.reverse:
        tasks.reverse()
    print(f"{len(rows)} instancias GDP (n<={a.max_n}) x {len(ALL_MODES)} modos = {len(tasks)} tarefas", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, mode, status in ex.map(_work, tasks, chunksize=1):
            done += 1
            if done % 25 == 0 or status != "skip":
                print(f"[{done}/{len(tasks)}] {name} {mode} {status}", flush=True)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
