"""E14: mecanismo do efeito do orcamento de custo apertado (GDP) sobre a selecao adaptativa de operadores.

Hipotese do manuscrito (§6.11): com orcamento de custo apertado, a etapa de reparo descarta muitos filhos dos operadores cegos a custo, de modo que as
taxas de sucesso dos operadores diferem mais do que com o orcamento folgado, e ha mais a aprender. Teste: repetir runs JA EXISTENTES do regime
normalizado (mesmas sementes -> mesmas fronteiras, conferido pela analise) com contadores por operador (`MOEADConfig.track_ops`, nao altera o
resultado): aplicacoes, filhos inviaveis ANTES do reparo, filhos que melhoram algum subproblema da vizinhanca, filhos inseridos no arquivo.
Grupos (10 instancias cada, 5 de cada familia, b02 e b03 misturados): GDP n=500 k02 (apertado), GDP n=500 k03 (folgado), CDP n=500, e os
equivalentes com n=150 (GKD-b). Metodos: random (uniforme), hypervolume (UCB-HV), dyts_hv (GTS-HV); 3 replicas.
Saida: results/e14_mechanism/<PROBLEMA>/runs (FORA da referencia congelada). Idempotente. Analise: analyze_e14.py.

    python run_e14.py [--workers 8] [--reps 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "e14_mechanism"
SEL = {"random": ("random", "scalar"), "hypervolume": ("ucb", "hypervolume"), "dyts_hv": ("dyts", "hypervolume")}


def groups(cat):
    """{grupo: [linhas do catalogo]} -- 10 instancias por grupo, escolhidas por passo dentro da lista ordenada por nome (mistura b02/b03 e familias)."""
    def pick(rows, k=10):
        rows = sorted(rows, key=lambda r: r["name"])
        step = len(rows) / k
        return [rows[int(i * step)] for i in range(k)]
    g = {}
    for prob in ("GDP", "CDP"):
        for n in (500, 150):
            rows = [r for r in cat if r["problem"] == prob and int(r["n"]) == n]
            if prob == "GDP":
                g[f"GDP-{n}-k02"] = pick([r for r in rows if "_k02" in r["name"]])
                g[f"GDP-{n}-k03"] = pick([r for r in rows if "_k03" in r["name"]])
            else:
                g[f"CDP-{n}"] = pick(rows)
    return g


def _work(args):
    problem, name, family, filepath, mode, rep = args
    out = OUT / problem / "runs" / f"{name}_{mode}_rep{rep:02d}.json"
    if out.exists():
        return name, mode, rep, "skip"
    from capacitated_problem import CDPInstance, GDPInstance
    from experiment import budget_for, seed_for
    from moead_aos import MOEAD_AOS, MOEADConfig

    sel, rw = SEL[mode]
    inst = (CDPInstance if problem == "CDP" else GDPInstance)(ROOT / "instances" / filepath)
    inst.family, inst.name = family, name
    b = budget_for(inst.n)
    cfg = MOEADConfig(n_weights=b["T"], neighborhood_size=b["nbr"], max_gen=b["max_gen"], operator_selection=sel, reward_mode=rw,
                      seed=seed_for(name, mode, rep), normalize_objectives=True, track_ops=True)
    t0 = time.perf_counter()
    res = MOEAD_AOS(inst, cfg).run()
    el = time.perf_counter() - t0
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"instance": name, "family": family, "n": inst.n, "mode": mode, "rep": rep, "seed": cfg.seed,
                               "elapsed_s": round(el, 4), "pareto_size": len(res.pareto_front), "pareto_front": res.pareto_front.tolist(),
                               "operator_counts": res.operator_counts.tolist(), "operator_stats": res.operator_stats}))
    tmp.replace(out)
    return name, mode, rep, f"{el:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    cat = list(csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")))
    g = groups(cat)
    tasks = []
    for gname, rows in g.items():
        cost = 130 if "-500" in gname else 20
        for k in range(a.reps):
            for r in rows:
                for m in SEL:
                    tasks.append((cost, (r["problem"], r["name"], r["family"], r["filepath"], m, k)))
    for p in ("CDP", "GDP"):
        (OUT / p / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "groups.json").write_text(json.dumps({k: [r["name"] for r in v] for k, v in g.items()}, indent=1))
    tasks.sort(key=lambda t: -t[0])
    tasks = [t[1] for t in tasks]
    print(f"{len(tasks)} tarefas", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, m, k, st in ex.map(_work, tasks, chunksize=1):
            done += 1
            if done % 20 == 0 or (st != "skip" and done < 5):
                print(f"[{done}/{len(tasks)}] {name} {m} rep{k} {st}", flush=True)
    print("FIM E14", flush=True)


if __name__ == "__main__":
    main()
