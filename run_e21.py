"""E21 -- escala completa no MDP: todas as 20 instancias MDG-b n=2000 e uma amostra do MDG-c n=3000, baselines ajustados e MOEA/D hibrido a tempo igual.

O E6 tinha so 3 instancias MDG-b n=2000 e 2 MDG-c n=3000 (6 metodos padrao) e o E10 10 instancias MDG-b com MOEA/D-Uniform/UCB-HV/GTS-HV. Aqui, por instancia
(1 replica; T = tempo do MOEA/D-Uniform padrao normalizado na mesma instancia, reaproveitado do E6/E10 quando existe, senao rodado aqui; inicializacao contada):
  uniform (MOEA/D-Uniform padrao; define T), t_ts / t_vns / t_grasp / t_nsga2 (baselines com os parametros ajustados do E11 na celula MDP-L, n=500: nao
  reajustados para n=2000/3000), m_hyb_vns50 (MOEA/D semeado pelo VNS durante 50% de T; E20).
MDG-c n=3000: 8 instancias espacadas (a familia e degenerada: MaxMin = 0 em quase todo subconjunto; a analise usa o MaxSum).
Saida: results/e21_xl/runs/<instancia>_<modo>_rep00.json (FORA da referencia). Analise: analyze_e21.py.

    python run_e21.py [--workers 8] [--families MDG-b,MDG-c]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import e11_common as C  # noqa: E402

OUT = ROOT / "results" / "e21_xl"
MODES = ["t_ts", "t_vns", "t_grasp", "t_nsga2", "m_hyb_vns50"]


def known_uniform(name: str):
    """T e a frente do MOEA/D-Uniform padrao ja existentes (E6/E10), se houver."""
    for d in (ROOT / "results" / "e10_budget_regime" / "MDP" / "runs", ROOT / "results" / "e6_scalability" / "runs"):
        f = d / f"{name}_random_rep00.json"
        if f.exists():
            return json.load(open(f))
    return None


def _write(out: Path, d: dict):
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(out)


def _work(args):
    row = args
    name = row["name"]
    d = OUT / "runs"
    todo = [m for m in MODES if not (d / f"{name}_{m}_rep00.json").exists()]
    if not todo and (d / f"{name}_uniform_rep00.json").exists():
        return 0
    d.mkdir(parents=True, exist_ok=True)
    inst = C.load_instance("MDP", row)
    u = d / f"{name}_uniform_rep00.json"
    if not u.exists():
        k = known_uniform(name)
        if k is not None:
            T, fr = float(k["elapsed_s"]), k["pareto_front"]
        else:
            pf, T = C.moead_uniform_time(inst, 0)
            fr = pf.tolist()
        _write(u, dict(instance=name, mode="uniform", rep=0, time_target_s=T, elapsed_s=T, pareto_size=len(fr), pareto_front=fr))
    T = json.load(open(u))["time_target_s"]
    params = {m: json.load(open(C.OUT / "tuning" / "MDP-L" / f"tuned_{m}.json"))["best"] for m in ("ts", "vns", "grasp", "nsga2")}
    seed = C.seed_for("e21", name, 0)
    for m in todo:
        t0 = time.perf_counter()
        if m == "m_hyb_vns50":
            from experiment import budget_for
            from moead_aos import MOEAD_AOS, MOEADConfig
            b = budget_for(inst.n)
            r = MOEAD_AOS(inst, MOEADConfig(n_weights=b["T"], neighborhood_size=b["nbr"], max_gen=10**7, operator_selection="random", reward_mode="scalar",
                                            normalize_objectives=True, grasp_alpha=C.DEFAULT["moead"][0]["grasp_alpha"], seed=seed, time_limit_s=T,
                                            seed_traj_frac=0.5, seed_traj_params=dict(params["vns"], method="vns"))).run()
            fr = np.asarray(r.pareto_front, float).reshape(-1, 2)
        else:
            fr, _, _ = C.run_method(m[2:], params[m[2:]], inst, seed, T)
        _write(d / f"{name}_{m}_rep00.json", dict(instance=name, family=row["family"], n=inst.n, mode=m, rep=0, seed=seed, time_target_s=T,
                                                  elapsed_s=time.perf_counter() - t0, pareto_size=int(len(fr)), pareto_front=np.asarray(fr).tolist()))
    return len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--families", default="MDG-b,MDG-c")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / "instances" / "catalog.csv")))
    sel = []
    for fam in a.families.split(","):
        rs = [r for r in rows if r["family"] == fam and int(r["n"]) >= 2000]
        if fam == "MDG-c":
            rs = rs[:: max(1, len(rs) // 8)][:8]
        sel += rs
    tasks = []
    for r in sel:
        k = known_uniform(r["name"])
        est = (k["elapsed_s"] if k else (900 if r["family"] == "MDG-b" else 2000)) * 6
        tasks.append((est, r))
    tasks.sort(key=lambda t: -t[0])
    print(f"E21: {len(tasks)} instancias ({len(MODES)} metodos + uniform cada)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(_work, [t[1] for t in tasks], chunksize=1), 1):
            print(f"   [{i}/{len(tasks)}] {time.time() - t0:.0f}s", flush=True)
    print("FIM E21", flush=True)


if __name__ == "__main__":
    main()
