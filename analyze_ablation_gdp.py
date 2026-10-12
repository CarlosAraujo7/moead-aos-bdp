"""Analise da ablacao GDP com orcamento relaxado (run_ablation_gdp.py).

Compara, por instancia (hv_norm, frente de referencia agrupada dentro de cada condicao),
os ranks medios dos 12 metodos em: GDP original (results/capacitated/GDP rep00),
GDP relaxado (results/ablation_gdp_relaxed) e, como referencia, CDP (n<=max_n).

    python analyze_ablation_gdp.py --max-n 150
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from experiment import ALL_MODES, build_ref_point, compute_indicators, compute_reference_front  # noqa: E402

AOS = ["hypervolume", "thompson_hv", "dyts_hv", "scalar", "thompson_scalar", "dyts_scalar", "random"]


def _front(path: Path):
    return np.array(json.load(open(path))["pareto_front"])


def _ranks(fronts: dict) -> dict:
    allf = list(fronts.values())
    rf, rp = compute_reference_front(allf), build_ref_point(allf)
    hv = {m: compute_indicators(f, rf, rp)["hv_norm"] for m, f in fronts.items()}
    order = sorted(hv, key=lambda m: -hv[m])  # empates: ordem estavel (aproximacao simples)
    # rank medio para empates
    ranks = {}
    for m in hv:
        ties = [k for k in hv if abs(hv[k] - hv[m]) < 1e-12]
        ranks[m] = np.mean([order.index(k) + 1 for k in ties])
    return ranks, hv


def collect(problem_dir: Path, names: list[str]):
    out = {}
    for nm in names:
        fs = {m: problem_dir / f"{nm}_{m}_rep00.json" for m in ALL_MODES}
        if all(p.exists() for p in fs.values()):
            out[nm] = _ranks({m: _front(p) for m, p in fs.items()})[0]
    return out


def summarize(label: str, R: dict):
    M = np.array([[R[i][m] for m in ALL_MODES] for i in R])
    avg = M.mean(axis=0)
    aos_avg = np.mean([avg[ALL_MODES.index(m)] for m in AOS])
    print(f"\n{label}  (N={len(R)} instancias)   rank medio do grupo AOS = {aos_avg:.2f}")
    for m, a in sorted(zip(ALL_MODES, avg), key=lambda t: t[1]):
        print(f"   {m:16s} {a:5.2f}")
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=150)
    a = ap.parse_args()
    cat = list(csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")))
    gdp = [r["name"] for r in cat if r["problem"] == "GDP" and int(r["n"]) <= a.max_n]
    cdp = [r["name"] for r in cat if r["problem"] == "CDP" and int(r["n"]) <= a.max_n]

    relaxed = collect(ROOT / "results/ablation_gdp_relaxed/runs", gdp)
    done = sorted(relaxed)
    orig = collect(ROOT / "results/capacitated/GDP/runs", done)  # mesmas instancias, pareado
    cdpR = collect(ROOT / "results/capacitated/CDP/runs", cdp)

    Mo = summarize("GDP ORIGINAL (orcamento ativo)", orig)
    Mr = summarize("GDP RELAXADO (orcamento = inf)", relaxed)
    summarize("CDP (referencia, instancias diferentes)", cdpR)

    print("\n=== Mudanca de rank medio, original -> relaxado (negativo = melhorou) ===")
    for m in ALL_MODES:
        j = ALL_MODES.index(m)
        d = Mr[:, j] - Mo[:, j]
        try:
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = float("nan")
        print(f"   {m:16s} {d.mean():+6.2f}   Wilcoxon pareado p={p:.4f}")


if __name__ == "__main__":
    main()
