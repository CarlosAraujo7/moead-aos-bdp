"""Teste do src/bo_trajectory.py (TS/VNS no estilo de Colmenar et al. 2018) em CDP, GDP e MDP.

Confere: (1) toda solucao do arquivo e viavel (is_feasible; no MDP tem exatamente m elementos, sem repeticao);
(2) objetivos do arquivo == inst.evaluate; (3) frente nao-dominada; (4) limite de tempo respeitado; (5) determinismo com a semente
(sem limite de tempo, max_iter fixo); (6) qualidade relativa (hv_norm contra a referencia congelada) para ter ordem de grandeza.

    python test_bo_trajectory.py [segundos]
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import reference_sets as RS
from bo_trajectory import BOTrajConfig, BOTrajectory
from capacitated_problem import CDPInstance, GDPInstance
from experiment import load_instances
from metrics import pareto_front

SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0


def load(problem, name):
    if problem == "MDP":
        row = next(r for r in csv.DictReader(open(ROOT / "instances/catalog.csv")) if r["name"] == name)
        return next(i for i in load_instances(ROOT / "instances/catalog.csv", ROOT / "instances", row["family"], int(row["n"]), int(row["n"])) if i.name == name)
    row = next(r for r in csv.DictReader(open(ROOT / "instances/catalog_capacitated.csv")) if r["name"] == name and r["problem"] == problem)
    inst = (CDPInstance if problem == "CDP" else GDPInstance)(ROOT / "instances" / row["filepath"])
    inst.family, inst.name = row["family"], name
    return inst


CASES = [("CDP", "GKD-b_11_n50_b02_m5"), ("CDP", "GKD-b_41_n150_b03_m15"), ("GDP", "GKD-b_11_n50_b02_m5_k02"), ("MDP", "SOM-a_10_n25_m7"),
         ("MDP", "GKD-c_10_n500_m50")]
bad = 0
for problem, name in CASES:
    try:
        inst = load(problem, name)
    except StopIteration:
        print("instancia ausente:", problem, name); continue
    ref = RS.load(problem)[name]
    for method in ("tabu", "vns"):
        for ls in ("pr", "alt", "dom"):
            cfg = BOTrajConfig(method=method, local_search=ls, seed=7, time_limit_s=SEC if ls == "pr" else 1.0)
            t0 = time.perf_counter()
            alg = BOTrajectory(inst, cfg)
            res = alg.run()
            dt = time.perf_counter() - t0
            # (1)-(3)
            errs = []
            for S, f in zip(alg.arch_s, alg.arch_f):
                if len(set(S.tolist())) != len(S):
                    errs.append("repetido")
                if problem == "MDP":
                    if len(S) != inst.m:
                        errs.append("tamanho")
                elif not inst.is_feasible(S):
                    errs.append("inviavel")
                fe = inst.evaluate(S)
                if abs(fe[0] - f[0]) > 1e-6 * max(1, abs(fe[0])) or abs(fe[1] - f[1]) > 1e-9:
                    errs.append("objetivo")
            pf = pareto_front(res.pareto_front)
            if len(pf) != len(res.pareto_front):
                errs.append("dominada no arquivo")
            if dt > (SEC if ls == "pr" else 1.0) + 2.0:
                errs.append(f"tempo {dt:.1f}s")
            hv = RS.indicators(res.pareto_front, ref)["hv_norm"]
            flag = "OK " if not errs else "ERR"
            bad += bool(errs)
            print(f"{flag} {problem} {name[:26]:26s} {method:5s} ls={ls:3s} {dt:5.1f}s iters={res.n_iter_executed:5d} |PF|={len(res.pareto_front):3d} hv_norm={hv:.3f} {set(errs) or ''}", flush=True)

# (5) determinismo
inst = load("CDP", "GKD-b_11_n50_b02_m5")
a = BOTrajectory(inst, BOTrajConfig(method="tabu", seed=3, max_iter=15)).run().pareto_front
b = BOTrajectory(inst, BOTrajConfig(method="tabu", seed=3, max_iter=15)).run().pareto_front
print("determinismo (mesma semente, max_iter fixo):", "OK" if np.array_equal(a, b) else "FALHOU")
bad += not np.array_equal(a, b)
print("RESULTADO:", "OK" if bad == 0 else f"{bad} problema(s)")
sys.exit(0 if bad == 0 else 1)
