"""Verificacao de reprodutibilidade (G5): reexecuta runs ja gravados com a semente gravada e compara a fronteira de Pareto BIT A BIT.

Cobre os 12 metodos do catalogo normalizado (CDP, GDP, MDP; instancias pequenas) e as variantes de `run_phase2.py` (ancoras, DYTS fiel,
recompensas insert/replace, selecao contextual, hiperparametros, orcamento maior). Variantes ciente-de-custo dependem do relogio e nao sao
reprodutiveis por construcao (nao entram). Saida: uma linha por caso e `REGRESSAO OK`/`FALHOU`; codigo de saida 0/1.

    python regression_check.py [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

CASES_NORM = [("CDP", "GKD-b_11_n50_b02_m5"), ("GDP", "GKD-b_11_n50_b02_m5_k02"), ("MDP", "SOM-a_10_n25_m7")]
MODES = ["hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar", "random",
         "nsga2", "grasp", "grasp_pr", "vns_pareto", "vns_tchebycheff"]
PHASE2 = ["ucb_insert", "ts_insert", "dyts_insert", "ucb_insert_mix50", "random_ref", "random_anchor", "dyts_hv_anchor", "random_sbts", "random_both",
          "fd_replace", "fd_hv", "ts_replace", "c_ucbins_w3p3", "c_tsins_w3p3", "h_ucbhv_c20", "h_tshv_w20", "b2_random"]


def _norm_case(args):
    problem, name, mode = args
    ref_dir = ROOT / ("results_norm" if problem == "MDP" else f"results/capacitated_norm/{problem}") / "runs"
    ref = ref_dir / f"{name}_{mode}_rep00.json"
    if not ref.exists():
        return problem, name, mode, "sem referencia gravada"
    from experiment import ExperimentConfig, load_capacitated_instances, load_instances, run_one
    cat = ROOT / "instances" / ("catalog.csv" if problem == "MDP" else "catalog_capacitated.csv")
    row = next(r for r in csv.DictReader(open(cat)) if r["name"] == name)
    if problem == "MDP":
        inst = load_instances(cat, ROOT / "instances", row["family"], int(row["n"]), int(row["n"]))
        inst = next(i for i in inst if i.name == name)
    else:
        inst = next(i for i in load_capacitated_instances(cat, ROOT / "instances", problem, row["family"], int(row["n"]), int(row["n"])) if i.name == name)
    pf, *_ = run_one(inst, ExperimentConfig(normalize_objectives=True), mode, 0)
    same = pf.tolist() == json.load(open(ref))["pareto_front"]
    return problem, name, mode, "IDENTICO" if same else "DIFERENTE"


def _phase2_case(args):
    problem, name, variant = args
    ref = ROOT / "results" / "phase2_aos" / problem / "runs" / f"{name}_{variant}_rep00.json"
    if not ref.exists():
        return problem, name, variant, "sem referencia gravada"
    import run_phase2 as R
    row = next(r for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")) if r["name"] == name)
    with tempfile.TemporaryDirectory() as tmp:
        R._work((problem, name, row["family"], row["filepath"], variant, 0, tmp))
        same = json.load(open(Path(tmp) / f"{name}_{variant}_rep00.json"))["pareto_front"] == json.load(open(ref))["pareto_front"]
    return problem, name, variant, "IDENTICO" if same else "DIFERENTE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(_norm_case, (p, n, m)) for p, n in CASES_NORM for m in MODES]
    jobs += [(_phase2_case, (p, n, v)) for p, n in CASES_NORM[:2] for v in PHASE2]
    ok = bad = skip = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(f, args) for f, args in jobs]
        for fu in futs:
            p, n, m, st = fu.result()
            print(f"{p:3s} {n:28s} {m:18s} {st}", flush=True)
            ok += st == "IDENTICO"; bad += st == "DIFERENTE"; skip += st.startswith("sem")
    print(f"\n{ok} identicos, {bad} diferentes, {skip} sem referencia gravada (variantes que nao existem para esse problema)")
    print("REGRESSAO", "OK" if bad == 0 else "FALHOU")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
