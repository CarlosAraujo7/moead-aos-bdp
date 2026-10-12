"""E20: MOEA/D memetico/hibrido (run_e20.py) contra o MOEA/D-Uniform e os baselines ajustados do E11 (mesmas instancias de teste, replicas e
referencia CONGELADA). Por celula: hv_norm (media sobre replicas por instancia), diferenca pareada para o MOEA/D-Uniform padrao e para o melhor
baseline ajustado, Wilcoxon + Holm e A12.

    python analyze_e20.py [--cells MDP-S,MDP-L] [--metric hv_norm|igd_plus]
Saida: terminal + results/analysis/e20_hybrid[_igd_plus].json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import e11_common as C  # noqa: E402
import reference_sets as RS  # noqa: E402
from analyze_e11 import path as e11_path  # noqa: E402
from analyze_phase2 import a12, holm  # noqa: E402

OUT = ROOT / "results" / "e20_hybrid"
NEW = {"m_ls": "MOEA/D + local search", "m_hyb": "MOEA/D seeded by Tabu (30% of T)", "m_hyb_ls": "MOEA/D seeded by Tabu + local search",
       "m_hyb_ts50": "MOEA/D seeded by Tabu (50% of T)", "m_hyb_vns30": "MOEA/D seeded by VNS (30% of T)", "m_hyb_vns50": "MOEA/D seeded by VNS (50% of T)"}
OLD = {"uniform": "MOEA/D-Uniform (default)", "t_moead": "MOEA/D-Uniform (tuned)", "def_ts": "Tabu (Colmenar irace)", "def_vns": "VNS (Colmenar irace)",
       "t_ts": "Tabu (tuned)", "t_vns": "VNS (tuned)", "t_grasp": "GRASP (tuned)", "t_nsga2": "NSGA-II (tuned)", "t_sms": "SMS-EMOA (tuned)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="MDP-S,MDP-L")
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus"], default="hv_norm")
    a = ap.parse_args()
    hb = a.metric == "hv_norm"
    allc = C.cells()
    res = {}
    for cell in a.cells.split(","):
        if not (OUT / cell / "runs").exists():
            continue
        prob = allc[cell]["problem"]
        ref = RS.load(prob)
        names = [r["name"] for r in allc[cell]["test"]]
        cache = {}

        def val(name, mode, reps=(0, 1, 2)):
            v = []
            for k in reps:
                p = (OUT / cell / "runs" / f"{name}_{mode}_rep{k:02d}.json") if mode in NEW else e11_path(cell, prob, name, mode, k)
                if not p.exists():
                    return None
                if p not in cache:
                    cache[p] = np.array(json.load(open(p))["pareto_front"], float).reshape(-1, 2)
                x = RS.indicators(cache[p], ref[name])[a.metric]
                v.append(x if hb else -x)
            return float(np.mean(v))

        modes = list(NEW) + [m for m in OLD if m != "uniform"]
        rows, used = [], []
        for n in names:
            vals = [val(n, m) for m in ["uniform"] + modes]
            if any(x is None for x in vals):
                continue
            if hb and max(RS.indicators(cache[p], ref[n])["hv_norm"] for p in cache if n in p.name) > 5.0:
                continue
            rows.append(vals)
            used.append(n)
        if len(used) < 5:
            print(f"{cell}: {len(used)} instancias completas")
            continue
        M = np.array(rows)
        allm = ["uniform"] + modes
        best_old = max((m for m in OLD if m != "uniform"), key=lambda m: M[:, allm.index(m)].mean())
        ps, info = [], {}
        for m in modes:
            d = M[:, allm.index(m)] - M[:, 0]
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            ps.append(p)
            db = M[:, allm.index(m)] - M[:, allm.index(best_old)]
            pb = wilcoxon(db).pvalue if np.any(db != 0) else 1.0
            info[m] = dict(mean=float(M[:, allm.index(m)].mean()), diff_uniform=float(d.mean()), a12_vs_uniform=a12(M[:, allm.index(m)], M[:, 0]),
                           better_than_uniform=int((d > 0).sum()), p_raw=float(p), diff_best_baseline=float(db.mean()), p_vs_best_baseline=float(pb))
        for m, pa in zip(modes, holm(ps)):
            info[m]["p_holm_vs_uniform"] = float(pa)
        res[cell] = dict(N=len(used), uniform_mean=float(M[:, 0].mean()), best_baseline=best_old, methods=info)
        print(f"\n=== {cell} [{a.metric}]: N={len(used)}; MOEA/D-Uniform (default) {M[:, 0].mean():.3f}; melhor baseline antigo: {OLD[best_old]} {M[:, allm.index(best_old)].mean():.3f}")
        for m in modes:
            i = info[m]
            lab = NEW.get(m) or OLD[m]
            ph = f"{i['p_holm_vs_uniform']:.4f}"
            print(f"   {lab:40s} {i['mean']:.3f}  vs uniform {i['diff_uniform']:+.3f} (A12 {i['a12_vs_uniform']:.2f}, melhor em {i['better_than_uniform']:3d}/{len(used)}, p_Holm {ph})"
                  f"   vs {OLD[best_old].split(' (')[0]} {i['diff_best_baseline']:+.3f} (p {i['p_vs_best_baseline']:.3f})")
    suffix = "" if a.metric == "hv_norm" else f"_{a.metric}"
    (ROOT / "results" / "analysis" / f"e20_hybrid{suffix}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
