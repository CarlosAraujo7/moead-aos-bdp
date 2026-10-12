"""Ablacao de normalizacao de objetivos no Tchebycheff do MOEA/D-AOS (CLAUDE.md, secao 2e).

O catalogo historico usa Tchebycheff SEM normalizar (MaxSum ~1e4-1e5 contra MaxMin ~1e0-1e2).
Aqui os 7 modos MOEA/D-AOS rodam com normalize_objectives=True; os 5 baselines nao dependem
da opcao e sao reaproveitados do catalogo. Grava em results/ablation_normalized/<PROBLEMA>/runs/
(nao toca no catalogo). Retomavel: pula (instancia, modo) ja gravados.

    python run_ablation_norm.py --problem CDP --max-n 150 --workers 6
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

AOS_MODES = ["hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar", "random"]


def _work(args):
    problem, name, family, filepath, mode, outdir = args
    out = Path(outdir) / f"{name}_{mode}_rep00.json"
    if out.exists():
        return name, mode, "skip"
    from capacitated_problem import CDPInstance, GDPInstance
    from experiment import ExperimentConfig, MODE_LABEL, run_one, seed_for

    inst = (CDPInstance if problem == "CDP" else GDPInstance)(ROOT / "instances" / filepath)
    inst.family, inst.name = family, name
    cfg = ExperimentConfig(normalize_objectives=True)
    pf, elapsed, hv_history, op_counts, op_qvals = run_one(inst, cfg, mode, 0)
    out.write_text(json.dumps({
        "instance": name, "family": family, "n": inst.n, "mode": mode,
        "label": MODE_LABEL.get(mode, mode), "rep": 0,
        "seed": seed_for(name, mode, 0), "elapsed_s": round(elapsed, 4),
        "pareto_size": len(pf), "pareto_front": pf.tolist(),
        "hv_history": hv_history, "operator_counts": op_counts, "operator_q_values": op_qvals,
        "normalize_objectives": True,
    }))
    return name, mode, f"{elapsed:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["CDP", "GDP"], default="CDP")
    ap.add_argument("--max-n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv"))
            if r["problem"] == a.problem and int(r["n"]) <= a.max_n]
    outdir = ROOT / "results" / "ablation_normalized" / a.problem / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    # instancias grandes primeiro dentro de cada n evita cauda longa no fim
    tasks = [(a.problem, r["name"], r["family"], r["filepath"], m, str(outdir))
             for r in sorted(rows, key=lambda r: -int(r["n"])) for m in AOS_MODES]
    print(f"{len(rows)} instancias {a.problem} (n<={a.max_n}) x {len(AOS_MODES)} modos = {len(tasks)} tarefas", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, mode, status in ex.map(_work, tasks, chunksize=1):
            done += 1
            if done % 10 == 0 or status != "skip":
                print(f"[{done}/{len(tasks)}] {name} {mode} {status}", flush=True)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
