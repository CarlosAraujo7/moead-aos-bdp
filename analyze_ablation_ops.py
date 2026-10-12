"""Analise da ablacao de operador unico (run_ablation_ops.py).

Por instancia: frente de referencia agrupada de 20 metodos = 8 operadores unicos (normalizados)
+ 7 MOEA/D-AOS normalizados (inclui random) + 5 baselines do catalogo; hv_norm e ranks.
Wilcoxon pareado (hv_norm) de cada operador unico contra `random` (pool uniforme), Holm.

    python analyze_ablation_ops.py --problem CDP --max-n 150
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from experiment import build_ref_point, compute_indicators, compute_reference_front  # noqa: E402
from operators import OPERATOR_NAMES  # noqa: E402

AOS = ["hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar", "random"]
BASE = ["nsga2", "grasp", "grasp_pr", "vns_pareto", "vns_tchebycheff"]
K = len(OPERATOR_NAMES)


def fr(p: Path):
    return np.array(json.load(open(p))["pareto_front"])


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["CDP", "GDP"], default="CDP")
    ap.add_argument("--max-n", type=int, default=150)
    ap.add_argument("--loo", action="store_true", help="analisar leave-one-out (pool sem o operador k)")
    a = ap.parse_args()
    pre = "no" if a.loo else "op"
    cat = list(csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")))
    names = [r["name"] for r in cat if r["problem"] == a.problem and int(r["n"]) <= a.max_n]
    base = ROOT / "results" / "capacitated" / a.problem / "runs"
    norm = ROOT / "results" / "ablation_normalized" / a.problem / "runs"
    ops = ROOT / "results" / ("ablation_pool_loo" if a.loo else "ablation_single_operator") / a.problem / "runs"
    labels = [f"{pre}{k}:{OPERATOR_NAMES[k]}" for k in range(K)] + AOS + BASE
    H = []
    for nm in names:
        paths = [ops / f"{nm}_{pre}{k}_rep00.json" for k in range(K)] + [norm / f"{nm}_{m}_rep00.json" for m in AOS] \
                + [base / f"{nm}_{m}_rep00.json" for m in BASE]
        if not all(p.exists() for p in paths):
            continue
        fs = [fr(p) for p in paths]
        rf, rp = compute_reference_front(fs), build_ref_point(fs)
        H.append([compute_indicators(f, rf, rp)["hv_norm"] for f in fs])
    if not H:
        print("sem instancias completas"); return
    H = np.array(H); R = np.array([rankdata(-r) for r in H])
    jr = labels.index("random")
    print(f"{a.problem}  N={len(H)} instancias, {len(labels)} metodos\n")
    ps = [wilcoxon(H[:, j] - H[:, jr]).pvalue if np.any(H[:, j] != H[:, jr]) else 1.0 for j in range(K)]
    adj = holm(np.array(ps))
    print(f"{'metodo':34s} rank medio  hv_norm medio  | {'pool sem o operador' if a.loo else 'operador unico'} vs random (dif. hv, p_holm)")
    for j in np.argsort(R.mean(0)):
        extra = ""
        if j < K:
            d = (H[:, j] - H[:, jr]).mean()
            extra = f"| {d:+.4f}  p={adj[j]:.4f} {'*' if adj[j] < 0.05 else 'ns'}"
        print(f"{labels[j]:34s} {R.mean(0)[j]:8.2f}   {H.mean(0)[j]:8.4f}   {extra}")


if __name__ == "__main__":
    main()
