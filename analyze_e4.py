"""E4: baselines a tempo igual (run_e4.py) vs MOEA/D-Uniform no MESMO processo/tempo. hv_norm (ou IGD+/eps+) contra a referencia fixa
(rode antes: python src/reference_sets.py build), media sobre replicas, Wilcoxon pareado + Holm, A12. Tambem mostra os baselines no
orcamento padrao (results/capacitated_norm) nas mesmas instancias, para ver quanto o tempo extra ajuda.

    python analyze_e4.py [--metric hv_norm]
Saida: terminal + results/analysis/e4_equal_time_<metric>.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import reference_sets as RS  # noqa: E402
from analyze_phase2 import a12, holm  # noqa: E402

E4 = ["e4_uniform", "e4_nsga2", "e4_vns_pareto", "e4_vns_tcheb", "e4_grasp", "e4_grasp_pr"]
STD = {"e4_nsga2": "nsga2", "e4_vns_pareto": "vns_pareto", "e4_vns_tcheb": "vns_tchebycheff", "e4_grasp": "grasp", "e4_grasp_pr": "grasp_pr"}
LAB = {"e4_uniform": "MOEA/D-Uniform", "e4_nsga2": "NSGA-II (equal time)", "e4_vns_pareto": "VNS-Pareto (equal time)",
       "e4_vns_tcheb": "VNS-Tcheb. (equal time)", "e4_grasp": "GRASP-MO (equal time)", "e4_grasp_pr": "GRASP-PR (equal time)", "nsga2": "NSGA-II (default)", "vns_pareto": "VNS-Pareto (default)",
       "vns_tchebycheff": "VNS-Tcheb. (default)", "grasp": "GRASP-MO (default)", "grasp_pr": "GRASP-PR (default)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus", "eps_plus"], default="hv_norm")
    a = ap.parse_args()
    hb = a.metric == "hv_norm"
    out = {}
    for prob in ("CDP", "GDP"):
        d = ROOT / "results" / "e4_equal_time" / prob / "runs"
        if not d.exists():
            continue
        ref = RS.load(prob)
        norm = ROOT / "results" / "capacitated_norm" / prob / "runs"
        insts = sorted({f.name.split("_e4_uniform_rep")[0] for f in d.glob("*_e4_uniform_rep*.json")})
        for lo, hi, lab in ((0, 150, "n<=150"), (500, 500, "n=500")):
            names = []
            for n in insts:
                j = json.load(open(next(d.glob(f"{n}_e4_uniform_rep*.json"))))
                if lo <= j["n"] <= hi and all((d / f"{n}_{m}_rep{k:02d}.json").exists() for m in E4 for k in range(3)):
                    names.append(n)
            if len(names) < 5:
                print(f"\n=== {prob} {lab}: {len(names)} instancias completas (poucas)"); continue
            cols = E4 + list(STD.values())
            def val(n, m, k):
                f = (d if m in E4 else norm) / f"{n}_{m}_rep{k:02d}.json"
                return RS.indicators(np.array(json.load(open(f))["pareto_front"]), ref[n])[a.metric] if f.exists() else np.nan
            M = np.array([[np.nanmean([val(n, m, k) for k in range(3)]) for m in cols] for n in names])
            ok = ~np.isnan(M).any(axis=1)
            M = M[ok]; nm = [n for n, o in zip(names, ok) if o]
            sign = -1.0 if hb else 1.0
            R = np.array([rankdata(sign * r) for r in M]); avg = R.mean(axis=0)
            # tempos
            tgt = st.median(json.load(open(d / f"{n}_e4_uniform_rep00.json"))["elapsed_s"] for n in nm)
            t_std = {m: st.median(json.load(open(norm / f"{n}_{m}_rep00.json"))["elapsed_s"] for n in nm) for m in STD.values()}
            print(f"\n=== {prob} {lab}: N={len(nm)} instancias x 3 replicas; tempo mediano do MOEA/D-Uniform (= limite dos baselines a tempo igual) {tgt:.1f}s")
            print("    tempo mediano dos baselines no orcamento padrao (capacitated_norm; cargas diferentes, so indicativo): " +
                  ", ".join(f"{k} {v:.1f}s" for k, v in t_std.items()))
            res = {}
            pr = {m: (wilcoxon(M[:, cols.index(m)] - M[:, 0]).pvalue if np.any(M[:, cols.index(m)] != M[:, 0]) else 1.0) for m in STD}
            ph = dict(zip(STD, holm([pr[m] for m in STD])))
            for j, m in sorted(enumerate(cols), key=lambda x: avg[x[0]]):
                row = dict(rank=float(avg[j]), mean=float(M[:, j].mean()))
                s = f"  {LAB[m]:26s} rank {avg[j]:5.2f}  {a.metric} {M[:, j].mean():.4f}"
                if m in STD:
                    dd = M[:, j] - M[:, 0]; p = ph[m]
                    s += f"  | vs Uniform: {dd.mean() * (1 if hb else -1):+.4f}  A12={a12(M[:, j] if hb else -M[:, j], M[:, 0] if hb else -M[:, 0]):.2f}  p_holm={p:.3g}"
                    row.update(diff_vs_uniform=float(dd.mean()), p=float(p))
                    j2 = cols.index(STD[m])
                    gain = M[:, j] - M[:, j2]
                    s += f"  | ganho do tempo extra: {gain.mean() * (1 if hb else -1):+.4f}"
                res[m] = row
                print(s)
            out[f"{prob} {lab}"] = dict(N=len(nm), target_time=tgt, methods=res)
    (ROOT / "results" / "analysis" / f"e4_equal_time_{a.metric}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
