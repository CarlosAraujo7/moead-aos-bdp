"""E14: mecanismo do efeito do orcamento de custo apertado (GDP) sobre a selecao adaptativa (run_e14.py).

Hipotese: com orcamento apertado o reparo descarta muitos filhos dos operadores cegos a custo, as taxas de sucesso dos operadores diferem mais e ha mais a
aprender. Por grupo (GDP n=500 k02/k03, CDP n=500; GDP n=150 k02/k03, CDP n=150) e sob selecao UNIFORME:
  (1) taxa de filhos inviaveis antes do reparo, por operador;
  (2) taxa de insercao no arquivo (filho nao dominado e nao duplicado), por operador, e sua dispersao entre operadores (desvio-padrao das 8 taxas e razao
      max/media);
  (3) fracao de selecoes de UCB-HV e GTS-HV por operador;
  (4) ganho esperado de insercao da alocacao do seletor = sum_k share_k * taxa_k(uniforme) - media_k taxa_k(uniforme), e sua correlacao (Spearman, entre
      instancias) com a diferenca de HV_norm (seletor - uniforme).
Tambem confere que as fronteiras reproduzem as dos runs ja existentes (capacitated_norm).

    python analyze_e14.py
Saida: terminal + results/analysis/e14_mechanism.json
"""
from __future__ import annotations

import csv
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import reference_sets as RS  # noqa: E402
from operators import OPERATOR_NAMES  # noqa: E402

OUT = ROOT / "results" / "e14_mechanism"
MODES = ["random", "hypervolume", "dyts_hv"]
LAB = {"random": "Uniform", "hypervolume": "UCB-HV", "dyts_hv": "GTS-HV"}


def main():
    groups = json.load(open(OUT / "groups.json"))
    res = {}
    for g, names in groups.items():
        prob = "GDP" if g.startswith("GDP") else "CDP"
        ref = RS.load(prob)
        per = {}     # nome -> modo -> dict(stats arrays, counts, hv)
        n_chk, n_same = 0, 0
        for nm in names:
            per[nm] = {}
            for m in MODES:
                runs = []
                for k in range(3):
                    p = OUT / prob / "runs" / f"{nm}_{m}_rep{k:02d}.json"
                    if not p.exists():
                        break
                    runs.append(json.load(open(p)))
                if len(runs) < 3:
                    per[nm] = None
                    break
                st = {key: np.array([r["operator_stats"][key] for r in runs], float) for key in ("applied", "infeasible", "improved", "inserted")}
                cnt = np.array([r["operator_counts"] for r in runs], float)
                hv = np.mean([RS.indicators(np.array(r["pareto_front"], float).reshape(-1, 2), ref[nm])["hv_norm"] for r in runs])
                per[nm][m] = dict(st=st, cnt=cnt, hv=float(hv))
                for k, r in enumerate(runs):   # conferencia contra o run original
                    q = ROOT / "results" / "capacitated_norm" / prob / "runs" / f"{nm}_{m}_rep{k:02d}.json"
                    if q.exists():
                        n_chk += 1
                        n_same += int(json.load(open(q))["pareto_front"] == r["pareto_front"])
            if per[nm] is None:
                del per[nm]
        if not per:
            continue
        N = len(per)
        # (1)-(2) taxas por operador sob uniforme (media de instancias das taxas por instancia)
        def rates(m, key):
            return np.array([(d[m]["st"][key].sum(0) / np.maximum(d[m]["st"]["applied"].sum(0), 1)) for d in per.values()])   # N x 8
        inf_u, ins_u, imp_u = rates("random", "infeasible"), rates("random", "inserted"), rates("random", "improved")
        disp = ins_u.std(1)                     # dispersao das taxas de insercao entre operadores, por instancia
        ratio = ins_u.max(1) / np.maximum(ins_u.mean(1), 1e-12)
        out = dict(N=N, repro=f"{n_same}/{n_chk}", infeasible_by_op=inf_u.mean(0).tolist(), inserted_by_op=ins_u.mean(0).tolist(),
                   improved_by_op=imp_u.mean(0).tolist(), insert_rate_mean=float(ins_u.mean()), insert_rate_sd_across_ops=float(disp.mean()),
                   insert_max_over_mean=float(ratio.mean()), infeasible_mean=float(inf_u.mean()))
        print(f"\n=== {g} (N={N} instancias; 3 replicas; repro {n_same}/{n_chk})")
        print(f"   {'operador':20s} {'inviavel':>8s} {'insere':>7s} {'melhora':>8s}" + "".join(f" {LAB[m]+' sel':>11s}" for m in MODES[1:]))
        shares = {}
        for m in MODES:
            shares[m] = np.array([d[m]["cnt"].sum(0) / d[m]["cnt"].sum() for d in per.values()])   # N x 8
        for k in range(8):
            print(f"   {OPERATOR_NAMES[k]:20s} {inf_u[:, k].mean():8.3f} {ins_u[:, k].mean():7.4f} {imp_u[:, k].mean():8.4f}"
                  + "".join(f" {shares[m][:, k].mean():11.3f}" for m in MODES[1:]))
        print(f"   media das 8 taxas de insercao {ins_u.mean():.4f}; dispersao (dp entre operadores) {disp.mean():.4f}; max/media {ratio.mean():.2f}; inviavel medio {inf_u.mean():.3f}")
        # (4) ganho esperado de insercao e correlacao com delta HV
        for m in MODES[1:]:
            gain = np.array([(shares[m][i] * ins_u[i]).sum() - ins_u[i].mean() for i in range(N)])
            rel = gain / np.maximum(ins_u.mean(1), 1e-12)
            dhv = np.array([per[nm][m]["hv"] - per[nm]["random"]["hv"] for nm in per])
            rho, p = spearmanr(rel, dhv) if N > 3 else (np.nan, np.nan)
            conc = shares[m].max(1)
            out[m] = dict(delta_hv=float(dhv.mean()), expected_insert_gain_rel=float(rel.mean()), spearman_rho=float(rho), spearman_p=float(p),
                          top2_share=float(np.sort(shares[m], axis=1)[:, -2:].sum(1).mean()), shares=shares[m].mean(0).tolist())
            print(f"   {LAB[m]:7s}: dHV vs uniforme {dhv.mean():+.4f}; ganho esperado de insercao (relativo) {rel.mean():+.3f}; Spearman(ganho, dHV) rho={rho:+.2f} p={p:.3f}; "
                  f"mass nos 2 mais usados {out[m]['top2_share']:.2f}")
        res[g] = out
    # ---- distancia da convergencia no orcamento padrao (n=500): ganho de HV do uniforme ao passar de 1/4 do orcamento ao orcamento padrao (runs do E10),
    #      por grupo, e correlacao entre instancias com a vantagem dos seletores HV (dados ja existentes; sem rodar nada novo)
    def hv(prob, nm, mode, base, ref):
        v = []
        for k in range(3):
            f = base / f"{nm}_{mode}_rep{k:02d}.json"
            if not f.exists():
                return None
            v.append(RS.indicators(np.array(json.load(open(f))["pareto_front"], float).reshape(-1, 2), ref[nm])["hv_norm"])
        return float(np.mean(v))
    rows = []
    for prob in ("CDP", "GDP"):
        ref = RS.load(prob)
        norm = ROOT / "results" / "capacitated_norm" / prob / "runs"
        e10 = ROOT / "results" / "e10_budget_regime" / prob / "runs"
        for r in csv.DictReader(open(ROOT / "instances" / "catalog_capacitated.csv")):
            if r["problem"] != prob or int(r["n"]) != 500:
                continue
            nm = r["name"]
            grp = "CDP" if prob == "CDP" else ("GDP-k02" if "_k02" in nm else "GDP-k03")
            vals = [hv(prob, nm, m, norm, ref) for m in ("random", "dyts_hv", "hypervolume")] + [hv(prob, nm, m, e10, ref) for m in ("q25_random", "q50_random")]
            if None not in vals:
                rows.append((grp, vals))
    conv = {}
    print("\n=== distancia da convergencia (n=500; N instancias; media de 3 replicas)")
    for grp in ("CDP", "GDP-k02", "GDP-k03", "ALL"):
        R = [v for g, v in rows if grp == "ALL" or g == grp]
        if not R:
            continue
        u1, g1, c1, u25, u50 = (np.array([v[i] for v in R]) for i in range(5))
        slope, dg, du = u1 - u25, g1 - u1, c1 - u1
        rg, pg = spearmanr(slope, dg)
        ru, pu = spearmanr(slope, du)
        conv[grp] = dict(N=len(R), uniform_hv=float(u1.mean()), slope_q25=float(slope.mean()), slope_q50=float((u1 - u50).mean()), d_gts=float(dg.mean()), d_ucb=float(du.mean()),
                         rho_gts=float(rg), p_gts=float(pg), rho_ucb=float(ru), p_ucb=float(pu))
        print(f"   {grp:8s} N={len(R):3d} HV uniforme {u1.mean():.3f}; ganho 1/4 -> padrao {slope.mean():.3f} (1/2 -> padrao {(u1 - u50).mean():.3f}); vantagem GTS-HV {dg.mean():+.3f}, UCB-HV {du.mean():+.3f}; "
              f"Spearman(ganho, vantagem GTS-HV) {rg:+.2f} (p={pg:.3f}), UCB-HV {ru:+.2f} (p={pu:.3f})")
    res["convergence"] = conv
    (ROOT / "results" / "analysis" / "e14_mechanism.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
