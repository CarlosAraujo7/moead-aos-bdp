"""E4: baselines a TEMPO IGUAL ao MOEA/D uniforme, no regime normalizado, CDP/GDP, com replicas.

Para cada (instancia, replica), NO MESMO PROCESSO (mesma carga): (1) roda o MOEA/D-Uniform normalizado (orcamento padrao) e mede o tempo T;
(2) roda NSGA-II, VNS-Pareto (sem parada antecipada) e VNS-Tchebycheff (normalizado) com limite de tempo de parede T (max_gen/max_iter
praticamente infinitos). GRASP-MO/PR ficam de fora: no orcamento padrao ja gastam tempo comparavel ou maior que o MOEA/D.
Grava em results/e4_equal_time/<P>/runs/<inst>_<modo>_rep<kk>.json com modo em {e4_uniform, e4_nsga2, e4_vns_pareto, e4_vns_tcheb}.

    python run_e4.py [--workers 8] [--reps 3] [--only CDP|GDP]
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
MODES = ["e4_uniform", "e4_nsga2", "e4_vns_pareto", "e4_vns_tcheb", "e4_grasp", "e4_grasp_pr"]
# (problema, n minimo, n maximo, passo de amostragem sobre o catalogo ordenado)
# (problema, n minimo, n maximo, passo, deslocamento). n<=150: TODAS as instancias. n=500: passo 4 com DOIS deslocamentos: no catalogo cada instancia-base
# tem 4 combinacoes (CDP: b02/b03 x ...; GDP: b02k02, b02k03, b03k02, b03k03 em ordem), entao o passo 4 sozinho pega SO a primeira combinacao
# (b02, e k02 no GDP = as mais apertadas). 7 out: acrescentado o deslocamento 1 (CDP: indice impar = b03) e 3 (GDP, b03k03) para cobrir tambem as folgadas.
SLICES = [("GDP", 0, 150, 1, 0), ("GDP", 500, 500, 4, 0), ("GDP", 500, 500, 4, 3), ("CDP", 0, 150, 1, 0), ("CDP", 500, 500, 4, 0), ("CDP", 500, 500, 4, 1)]


def _write(out: Path, d: dict):
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(out)


def _work(args):
    problem, name, family, filepath, rep, outdir = args
    outdir = Path(outdir)
    outs = {m: outdir / f"{name}_{m}_rep{rep:02d}.json" for m in MODES}
    if all(o.exists() for o in outs.values()):
        return name, rep, "skip"
    from capacitated_problem import CDPInstance, GDPInstance
    from experiment import ExperimentConfig, budget_for, run_one, seed_for
    from nsga2 import NSGA2, NSGA2Config
    from vns import BVNS, BVNSConfig
    from grasp_mo import GRASPMO, GRASPMOConfig

    inst = (CDPInstance if problem == "CDP" else GDPInstance)(ROOT / "instances" / filepath)
    inst.family, inst.name = family, name
    b = budget_for(inst.n)
    cfg = ExperimentConfig(normalize_objectives=True)
    if outs["e4_uniform"].exists():       # reaproveita o MOEA/D-Uniform (e o tempo-alvo) de uma execucao anterior
        T = json.load(open(outs["e4_uniform"]))["time_target_s"]
        base = {"instance": name, "family": family, "n": inst.n, "rep": rep, "time_target_s": T}
    else:
        pf, el, *_ = run_one(inst, cfg, "random", rep)
        T = el
        base = {"instance": name, "family": family, "n": inst.n, "rep": rep, "time_target_s": T}
        _write(outs["e4_uniform"], {**base, "mode": "e4_uniform", "elapsed_s": el, "pareto_size": len(pf), "pareto_front": pf.tolist()})
    done = {}

    def run_mode(m):
        t0 = time.perf_counter()
        if m == "e4_nsga2":
            r = NSGA2(inst, NSGA2Config(pop_size=b["nsga2_pop"], max_gen=10**7, seed=seed_for(name, m, rep), time_limit_s=T)).run()
            it = r.n_gen_executed
        elif m == "e4_vns_pareto":
            r = BVNS(inst, BVNSConfig(max_iter=10**8, max_no_improve=10**9, k_max=3, variant="pareto", n_weights=b["T"],
                                      seed=seed_for(name, m, rep), time_limit_s=T)).run()
            it = r.n_iter_executed
        elif m == "e4_vns_tcheb":
            r = BVNS(inst, BVNSConfig(max_iter=10**8, max_no_improve=10**9, k_max=3, variant="tchebycheff", n_weights=b["T"],
                                      normalize_objectives=True, seed=seed_for(name, m, rep), time_limit_s=T)).run()
            it = r.n_iter_executed
        else:   # e4_grasp / e4_grasp_pr
            r = GRASPMO(inst, GRASPMOConfig(max_iter=10**7, alpha=cfg.grasp_alpha, path_relink=(m == "e4_grasp_pr"),
                                            seed=seed_for(name, m, rep), time_limit_s=T)).run()
            it = r.n_gen_executed
        done[m] = (r.pareto_front, time.perf_counter() - t0, int(it))

    for m in MODES[1:]:
        if not outs[m].exists():
            run_mode(m)
    for m, (pfb, elb, it) in done.items():
        if outs[m].exists():
            continue
        _write(outs[m], {**base, "mode": m, "elapsed_s": elb, "iterations": it, "pareto_size": len(pfb), "pareto_front": pfb.tolist()})
    return name, rep, f"T={T:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", choices=["CDP", "GDP"], default=None)
    a = ap.parse_args()
    cat = list(csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")))
    tasks = []
    for prob, lo, hi, step, off in SLICES:
        if a.only and prob != a.only:
            continue
        rows = [r for r in cat if r["problem"] == prob and lo <= int(r["n"]) <= hi][off::step]
        outdir = ROOT / "results" / "e4_equal_time" / prob / "runs"
        outdir.mkdir(parents=True, exist_ok=True)
        for k in range(a.reps):
            for r in rows:
                tasks.append((int(r["n"]), (prob, r["name"], r["family"], r["filepath"], k, str(outdir))))
    tasks.sort(key=lambda t: (-t[0]))   # grandes primeiro (evita cauda longa)
    tasks = [t[1] for t in tasks]
    print(f"{len(tasks)} tarefas (instancia x replica); cada uma = MOEA/D + 3 baselines a tempo igual", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, rep, st in ex.map(_work, tasks, chunksize=1):
            done += 1
            if done % 10 == 0:
                print(f"[{done}/{len(tasks)}] {name} rep{rep} {st}", flush=True)
    print("FIM E4", flush=True)


if __name__ == "__main__":
    main()
