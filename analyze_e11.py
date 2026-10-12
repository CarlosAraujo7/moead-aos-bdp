"""E11: baselines AJUSTADOS e Tabu/VNS de Colmenar vs MOEA/D-Uniform (run_e11_tune.py, run_e11.py).

Por celula (CDP-S/L, GDP-S/L, MDP-S/L; S = n<=150, L = n=500): media sobre replicas por instancia (hv_norm contra a referencia CONGELADA, ou -IGD+),
diferenca pareada para a referencia da tabela, Wilcoxon + Holm (sobre os metodos da tabela) e A12. Tabelas por celula:
  at_T      : metodos padrao e ajustados a tempo igual (T) vs MOEA/D-Uniform PADRAO (o do experimento principal);
  at_T_vs_tm: baselines ajustados vs MOEA/D-Uniform AJUSTADO (mesmo esforco de ajuste), a T;
  at_5T     : ajustados com 5T (so tarefas com execucao longa) vs MOEA/D-Uniform padrao a T; at_5T_vs_tm: vs MOEA/D ajustado tambem a 5T.
Tambem: ganho do ajuste (ajustado - padrao) e razao tempo gasto / limite.

    python analyze_e11.py [--metric hv_norm|igd_plus]
Saida: terminal + results/analysis/e11_strong_baselines[_igd_plus].json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import e11_common as C  # noqa: E402
import reference_sets as RS  # noqa: E402
from analyze_phase2 import a12, holm  # noqa: E402

LAB = {"uniform": "MOEA/D-Uniform (default)", "t_moead": "MOEA/D-Uniform (tuned)", "l_moead": "MOEA/D-Uniform (tuned, 5T)",
       "def_nsga2": "NSGA-II (default)", "def_grasp": "GRASP-MO (default)", "def_grasp_pr": "GRASP-PR (default)",
       "def_ts": "Tabu (Colmenar irace)", "def_vns": "VNS (Colmenar irace)",
       "t_nsga2": "NSGA-II (tuned)", "t_grasp": "GRASP (tuned)", "t_ts": "Tabu (tuned)", "t_vns": "VNS (tuned)",
       "l_nsga2": "NSGA-II (tuned, 5T)", "l_grasp": "GRASP (tuned, 5T)", "l_ts": "Tabu (tuned, 5T)", "l_vns": "VNS (tuned, 5T)",
       "def_sms": "SMS-EMOA (default)", "t_sms": "SMS-EMOA (tuned)", "l_sms": "SMS-EMOA (tuned, 5T)"}
DEF_T = ["def_nsga2", "def_grasp", "def_grasp_pr", "def_ts", "def_vns"]
TUN_T = ["t_nsga2", "t_grasp", "t_ts", "t_vns"]
TUN_L = ["l_nsga2", "l_grasp", "l_ts", "l_vns"]


def path(cell, prob, name, mode, rep):
    r11 = C.OUT / cell / "runs"
    e4 = ROOT / "results" / "e4_equal_time" / prob / "runs"
    if mode == "uniform":
        return (r11 / f"{name}_d_uniform_rep{rep:02d}.json") if prob == "MDP" else (e4 / f"{name}_e4_uniform_rep{rep:02d}.json")
    if mode in ("def_nsga2", "def_grasp", "def_grasp_pr"):
        base = mode.split("def_")[1]
        return (r11 / f"{name}_d_{base}_rep{rep:02d}.json") if prob == "MDP" else (e4 / f"{name}_e4_{base}_rep{rep:02d}.json")
    if mode in ("def_ts", "def_vns", "def_sms"):
        return r11 / f"{name}_d_{mode[4:]}_rep{rep:02d}.json"
    return r11 / f"{name}_{mode}_rep{rep:02d}.json"      # t_*, l_*


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus"], default="hv_norm")
    ap.add_argument("--cells", default="CDP-S,GDP-S,MDP-S,CDP-L,GDP-L,MDP-L")
    a = ap.parse_args()
    hb = a.metric == "hv_norm"
    allc = C.cells()
    out = {}
    cache = {}

    def front(p):
        if p not in cache:
            cache[p] = np.array(json.load(open(p))["pareto_front"], float).reshape(-1, 2)
        return cache[p]

    for cell in a.cells.split(","):
        prob = allc[cell]["problem"]
        ref = RS.load(prob)
        names = [r["name"] for r in allc[cell]["test"]]
        reps_all = [0, 1, 2]
        large = cell.endswith("-L")

        def val(name, mode, reps):
            v = []
            for k in reps:
                p = path(cell, prob, name, mode, k)
                if not p.exists():
                    return None
                x = RS.indicators(front(p), ref[name])[a.metric]
                v.append(x if hb else -x)
            return float(np.mean(v))

        res = {}
        # Instancias cuja referencia congelada e degenerada (MaxMin = 0 em toda a frente) explodem o hv_norm quando algum metodo acha MaxMin > 0
        # (area de referencia ~ 0). Regra: excluir da celula toda instancia com hv_norm > 5 em qualquer execucao do E11 (e informar).
        excluded = []
        for n in names:
            big = False
            for f in (C.OUT / cell / "runs").glob(f"{n}_*_rep*.json"):
                if RS.indicators(front(f), ref[n])["hv_norm"] > 5.0:
                    big = True
                    break
            if big:
                excluded.append(n)
        names = [n for n in names if n not in excluded]
        res["excluded_degenerate_reference"] = excluded
        if excluded:
            print(f"{chr(10)}[{cell}] instancias excluidas (referencia degenerada, hv_norm > 5): {excluded}")

        def table(tag, ref_mode, modes, reps_of):
            rows, used = [], []
            for n in names:
                reps = reps_of(n)
                vals = [val(n, m, reps) for m in [ref_mode] + modes]
                if any(x is None for x in vals):
                    continue
                rows.append(vals)
                used.append(n)
            if len(used) < 3:
                print(f"\n=== {cell} {tag}: {len(used)} instancias completas (poucas)")
                return
            M = np.array(rows)
            ps, info = [], {}
            for j, m in enumerate(modes, start=1):
                d = M[:, j] - M[:, 0]
                p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else 1.0
                ps.append(p)
                info[m] = dict(mean=float(M[:, j].mean()), diff=float(d.mean()), a12=a12(M[:, j], M[:, 0]), better=int((d > 0).sum()), p_raw=float(p))
            for m, pa in zip(modes, holm(ps)):
                info[m]["p_holm"] = float(pa)
            res[tag] = dict(N=len(used), reference=ref_mode, reference_mean=float(M[:, 0].mean()), methods=info, instances=used)
            print(f"\n=== {cell} {tag} [{a.metric}]: N={len(used)}; referencia {LAB[ref_mode]} {M[:, 0].mean():.3f}")
            for m in modes:
                i = info[m]
                print(f"   {LAB[m]:28s} {i['mean']:.3f}  dif {i['diff']:+.3f}  A12 {i['a12']:.2f}  melhor em {i['better']:3d}/{len(used)}  p_Holm {i['p_holm']:.4f}")

        has_sms = any((C.OUT / cell / "runs").glob("*_t_sms_rep*.json"))      # SMS-EMOA entra nas tabelas so quando ja foi rodado na celula
        DEF_T_c = DEF_T + (["def_sms"] if has_sms else [])
        TUN_T_c = TUN_T + (["t_sms"] if has_sms else [])
        TUN_L_c = TUN_L + (["l_sms"] if has_sms else [])
        all_reps = lambda n: reps_all
        long_reps = (lambda n: [0]) if large else (lambda n: [0])     # 5T: so a replica 0 (ver run_e11.py)
        table("at_T", "uniform", ["t_moead"] + DEF_T_c + TUN_T_c, all_reps)
        table("at_T_vs_tm", "t_moead", TUN_T_c + ["def_ts", "def_vns"], all_reps)
        # 5T: instancias com execucao longa; comparacao na replica 0 (referencias a T tambem na replica 0)
        table("at_5T", "uniform", ["l_moead"] + TUN_L_c, long_reps)
        table("at_5T_vs_tm", "l_moead", TUN_L_c, long_reps)
        # ganho do ajuste (ajustado - padrao), a tempo igual
        gain = {}
        for dm, tm in (("uniform", "t_moead"), ("def_nsga2", "t_nsga2"), ("def_grasp_pr", "t_grasp"), ("def_ts", "t_ts"), ("def_vns", "t_vns")) + ((("def_sms", "t_sms"),) if has_sms else ()):
            x, y = [], []
            for n in names:
                a_, b_ = val(n, dm, reps_all), val(n, tm, reps_all)
                if a_ is not None and b_ is not None:
                    x.append(a_); y.append(b_)
            if len(x) >= 5:
                d = np.array(y) - np.array(x)
                gain[tm] = dict(N=len(x), default=float(np.mean(x)), tuned=float(np.mean(y)), gain=float(d.mean()),
                                p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
        res["tuning_gain"] = gain
        if gain:
            print(f"\n   [{cell}] ganho do ajuste:", {k: f"{v['default']:.3f}->{v['tuned']:.3f} (p={v['p']:.2g})" for k, v in gain.items()})
        ratio = {}
        r11 = C.OUT / cell / "runs"
        if r11.exists():
            for mode in TUN_T_c + TUN_L_c + ["t_moead", "l_moead", "d_ts", "d_vns"]:
                rs = []
                for f in r11.glob(f"*_{mode}_rep*.json"):
                    j = json.load(open(f))
                    rs.append(j["elapsed_s"] / j["limit_s"])
                if rs:
                    ratio[mode] = dict(median=float(st.median(rs)), p95=float(np.percentile(rs, 95)), n=len(rs))
        res["time_ratio"] = ratio
        tp = {}
        for m in C.METHODS:
            f = C.OUT / "tuning" / cell / f"tuned_{m}.json"
            if f.exists():
                j = json.load(open(f))
                tp[m] = dict(best=j["best"], train_hv=j["best_mean_train_hv"])
        res["tuned_params"] = tp
        out[cell] = res
    suffix = "" if hb else "_igd_plus"
    (ROOT / "results" / "analysis" / f"e11_strong_baselines{suffix}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
