"""E10: a seleção adaptativa ajuda quando o orcamento e curto em relacao ao tamanho da instancia? (CLAUDE.md §2s, E6)

Hipotese: em n<=500 (CDP/GDP) a recompensa HV satura cedo e o uniforme empata com os bandits; em MDG-b n=2000 (600 geracoes) a busca esta longe da
convergencia e UCB-HV/GTS-HV vencem. Teste controlado: variar o orcamento de geracoes (multiplicador) e comparar UCB-HV e GTS-HV com o uniforme
NO MESMO orcamento.
  A) CDP n=500 (40 instancias), multiplicadores 1/4 e 1/2 (x1 ja existe em results/capacitated_norm, mesmas sementes);
  B) GDP n=500 (40 das 80 instancias, uma em duas = TODAS COM k02, orcamento de custo apertado), multiplicadores 1/4 e 1/2;
  B2) GDP n=500: as outras 40 (todas k03), multiplicadores 1/4 e 1/2;
  C) MDP MDG-b n=2000: 10 instancias (inclui as 3 do E6), orcamento padrao (x1; os runs do E6 sao reaproveitados por copia);
  D) MDP MDG-b n=2000: as 3 instancias do E6, 4x as geracoes.
Variantes (todas regime normalizado): <prefixo>random, <prefixo>hypervolume (UCB-HV), <prefixo>dyts_hv (GTS-HV); prefixo "q25_", "q50_", "x4_" ou "" (x1).
Saida: results/e10_budget_regime/<PROBLEMA>/runs (FORA da referencia congelada). Idempotente. Analise: analyze_e10.py.

    python run_e10.py [--workers 8] [--parts A,B,C,D]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "e10_budget_regime"
SEL = {"random": ("random", "scalar"), "hypervolume": ("ucb", "hypervolume"), "dyts_hv": ("dyts", "hypervolume")}
MULT = {"": 1.0, "q25_": 0.25, "q50_": 0.5, "x4_": 4.0}
MDGB_IDX = [0, 2, 4, 6, 8, 10, 12, 15, 17, 19]   # 10 das 20 instancias MDG-b n=2000 (inclui as 3 do E6: 0, 10, 19)
E6_IDX = [0, 10, 19]


def _work(args):
    problem, name, family, filepath, variant, rep = args
    out = OUT / problem / "runs" / f"{name}_{variant}_rep{rep:02d}.json"
    if out.exists():
        return name, variant, rep, "skip"
    from experiment import budget_for, seed_for
    from moead_aos import MOEAD_AOS, MOEADConfig

    prefix = next((p for p in ("q25_", "q50_", "x4_") if variant.startswith(p)), "")
    mode = variant[len(prefix):]
    sel, rw = SEL[mode]
    if problem == "MDP":
        from problem import MDPInstance
        inst = MDPInstance(ROOT / "instances" / filepath)
    else:
        from capacitated_problem import CDPInstance, GDPInstance
        inst = (CDPInstance if problem == "CDP" else GDPInstance)(ROOT / "instances" / filepath)
    inst.family, inst.name = family, name
    b = budget_for(inst.n)
    gens = max(1, int(round(b["max_gen"] * MULT[prefix])))
    cfg = MOEADConfig(n_weights=b["T"], neighborhood_size=b["nbr"], max_gen=gens, operator_selection=sel, reward_mode=rw,
                      seed=seed_for(name, mode, rep), normalize_objectives=True)   # semente = a do regime x1 (numeros aleatorios comuns)
    t0 = time.perf_counter()
    res = MOEAD_AOS(inst, cfg).run()
    el = time.perf_counter() - t0
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"instance": name, "family": family, "n": inst.n, "mode": variant, "rep": rep, "seed": cfg.seed,
                               "elapsed_s": round(el, 4), "max_gen": gens, "pareto_size": len(res.pareto_front),
                               "pareto_front": res.pareto_front.tolist(), "operator_counts": res.operator_counts.tolist()}))
    tmp.replace(out)
    return name, variant, rep, f"{el:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--parts", default="A,B,C,D")
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    cat_c = list(csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")))
    cat_m = list(csv.DictReader(open(ROOT / "instances" / "catalog.csv")))
    tasks = []   # (custo estimado, tarefa)

    def add(problem, rows, prefixes, cost):
        for k in range(a.reps):
            for r in rows:
                for p in prefixes:
                    for m in SEL:
                        tasks.append((cost, (problem, r["name"], r["family"], r["filepath"], p + m, k)))

    if "A" in parts:
        add("CDP", [r for r in cat_c if r["problem"] == "CDP" and int(r["n"]) == 500], ["q25_", "q50_"], 60)
    if "B" in parts:
        add("GDP", [r for r in cat_c if r["problem"] == "GDP" and int(r["n"]) == 500][::2], ["q25_", "q50_"], 70)
    if "B2" in parts:   # B2 (7 out): as outras 40 instancias GDP n=500 (a parte B pegou so as k02, orcamento de custo apertado)
        add("GDP", [r for r in cat_c if r["problem"] == "GDP" and int(r["n"]) == 500][1::2], ["q25_", "q50_"], 70)
    mdgb = [r for r in cat_m if r["family"] == "MDG-b" and int(r["n"]) == 2000]
    if "C" in parts:
        (OUT / "MDP" / "runs").mkdir(parents=True, exist_ok=True)
        e6 = ROOT / "results" / "e6_scalability" / "runs"
        for i in E6_IDX:                       # reaproveita os runs x1 do E6 (mesma semente, mesma configuracao)
            for m in SEL:
                for k in range(a.reps):
                    src = e6 / f"{mdgb[i]['name']}_{m}_rep{k:02d}.json"
                    dst = OUT / "MDP" / "runs" / src.name
                    if src.exists() and not dst.exists():
                        shutil.copy(src, dst)
        add("MDP", [mdgb[i] for i in MDGB_IDX], [""], 900)
    if "D" in parts:
        add("MDP", [mdgb[i] for i in E6_IDX], ["x4_"], 3600)
    for p in ("CDP", "GDP", "MDP"):
        (OUT / p / "runs").mkdir(parents=True, exist_ok=True)
    tasks.sort(key=lambda t: -t[0])
    tasks = [t[1] for t in tasks]
    print(f"{len(tasks)} tarefas (partes {sorted(parts)})", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, v, k, st in ex.map(_work, tasks, chunksize=1):
            done += 1
            if done % 25 == 0 or st != "skip" and done < 5:
                print(f"[{done}/{len(tasks)}] {name} {v} rep{k} {st}", flush=True)
    print("FIM E10", flush=True)


if __name__ == "__main__":
    main()
