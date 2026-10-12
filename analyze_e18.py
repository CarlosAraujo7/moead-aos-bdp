"""E18: tamanho do conjunto como dimensao explorada (run_e18.py). Por experimento (free, bound2, bound15) e celula:

  * referencia PROPRIA por instancia = uniao nao-dominada de todas as execucoes do experimento/celula (todos os metodos e replicas) -- no
    problema com limite de cardinalidade a referencia congelada do estudo principal nao vale (violaria o limite); no 'free' a referencia
    congelada e uma FATIA da frente (E13) e e dominada pela uniao, entao nao e acrescentada;
  * hv_norm / IGD+ / eps+ contra essa referencia (src/reference_sets.indicators); media sobre replicas por instancia;
  * ranks medios (Friedman) e comparacao pareada de cada metodo com o MOEA/D-Uniform padrao ('moead'): Wilcoxon + Holm e A12;
  * diagnosticos: MaxSum maximo / MaxSum maximo do 'moead', tamanho mediano e maximo dos membros do arquivo.

    python analyze_e18.py [--metric hv_norm|igd_plus|eps_plus] [--exps free,bound2,bound15]
Saida: terminal + results/analysis/e18_<metrica>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import reference_sets as RS  # noqa: E402
from metrics import pareto_front  # noqa: E402
from analyze_phase2 import a12, holm  # noqa: E402

OUT = ROOT / "results" / "e18_size"
LAB = {"moead": "MOEA/D-Uniform", "moead_size": "MOEA/D-Uniform + size operators", "moead_t": "MOEA/D-Uniform (tuned)",
       "moead_size_t": "MOEA/D-Uniform + size operators (tuned)", "ts": "Tabu (tuned)", "ts_add": "Tabu (tuned) + insertions",
       "vns": "VNS (tuned)", "vns_add": "VNS (tuned) + insertions", "nsga2": "NSGA-II (tuned)", "grasp": "GRASP (tuned)",
       "moead_size_ucbhv": "MOEA/D-UCB-HV + size operators", "moead_size_gtshv": "MOEA/D-GTS-HV + size operators"}
MARGIN = RS.MARGIN


def load_cell(exp: str, cell: str):
    d = OUT / exp / cell / "runs"
    runs = {}
    for f in d.glob("*.json"):
        j = json.load(open(f))
        runs[(j["instance"], j["mode"], j["rep"])] = j
    return runs


def build_refs(runs):
    pts = {}
    for (inst, _, _), j in runs.items():
        pts.setdefault(inst, []).append(np.asarray(j["pareto_front"], float).reshape(-1, 2))
    refs = {}
    for inst, fs in pts.items():
        allp = np.vstack(fs)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        refs[inst] = {"front": pareto_front(allp).tolist(), "ref_point": (lo - MARGIN * np.maximum(hi - lo, [1.0, 1e-9])).tolist()}
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus", "eps_plus"], default="hv_norm")
    ap.add_argument("--exps", default="free,bound2,bound15")
    a = ap.parse_args()
    hb = a.metric == "hv_norm"
    res = {}
    for exp in a.exps.split(","):
        for cell in ("CDP-S", "GDP-S", "CDP-L", "GDP-L"):
            runs = load_cell(exp, cell)
            if not runs:
                continue
            refs = build_refs(runs)
            cnt = {}
            for (_, m_, _) in runs:
                cnt[m_] = cnt.get(m_, 0) + 1
            full = max(cnt.values())
            modes = sorted([m for m in cnt if cnt[m] == full], key=lambda m: list(LAB).index(m) if m in LAB else 99)   # so metodos completos na celula
            if len(modes) < len(cnt):
                print(f"[{exp} {cell}] metodos incompletos ignorados: {sorted(set(cnt) - set(modes))}")
            insts = sorted({i for (i, _, _) in runs})
            val, msum, sz, szi = {}, {}, {}, {}
            for (inst, m, rep), j in runs.items():
                fr = np.asarray(j["pareto_front"], float).reshape(-1, 2)
                val.setdefault((inst, m), []).append(RS.indicators(fr, refs[inst])[a.metric])
                msum.setdefault((inst, m), []).append(float(fr[:, 0].max()))
                if j.get("archive_sizes"):
                    sz.setdefault(m, []).extend(j["archive_sizes"])
                    szi.setdefault((inst, m), []).extend(j["archive_sizes"])
            # instancias completas (todos os metodos e todas as replicas esperadas) e com referencia nao degenerada
            keep = []
            for inst in insts:
                rf = np.asarray(refs[inst]["front"])
                hv_ref = RS.hypervolume(rf, np.asarray(refs[inst]["ref_point"]))
                if hv_ref > 0 and all((inst, m) in val and len(val[(inst, m)]) >= 3 for m in modes):
                    keep.append(inst)
            if len(keep) < 3:
                print(f"\n=== {exp} {cell}: {len(keep)} instancias completas (poucas, {len(insts)} com algum run)")
                continue
            M = np.array([[np.mean(val[(i, m)]) for m in modes] for i in keep])
            if not hb:
                M = -M
            ranks = np.array([rankdata(-row) for row in M])
            import e11_common as C
            rows_ = {r["name"]: r for r in C.cells()[cell]["test"]}
            kref = {i: len(C.load_instance(C.cells()[cell]["problem"], rows_[i])._efficiency_greedy_fill()) for i in keep}
            ms_ratio = np.array([[np.mean(msum[(i, m)]) / max(np.mean(msum[(i, "moead")]), 1e-12) for m in modes] for i in keep])
            k0 = modes.index("moead")
            ps, info = [], {}
            for j, m in enumerate(modes):
                d = M[:, j] - M[:, k0]
                p = wilcoxon(d).pvalue if (m != "moead" and len(d) >= 5 and np.any(d != 0)) else 1.0
                ps.append(p)
                info[m] = dict(mean=float(M[:, j].mean()), rank=float(ranks[:, j].mean()), diff_vs_moead=float(d.mean()), a12=a12(M[:, j], M[:, k0]),
                               better=int((d > 0).sum()), p_raw=float(p), maxsum_ratio_median=float(np.median(ms_ratio[:, j])),
                               size_median=(float(np.median(sz[m])) if m in sz else None), size_max=(int(max(sz[m])) if m in sz else None),
                               size_over_kref=(float(np.median([np.median(szi[(i, m)]) / kref[i] for i in keep if (i, m) in szi])) if m in sz else None))
            idx = [j for j, m in enumerate(modes) if m != "moead"]
            for j, pa in zip(idx, holm([ps[j] for j in idx])):
                info[modes[j]]["p_holm"] = float(pa)
            try:
                chi = float(friedmanchisquare(*[M[:, j] for j in range(len(modes))]).statistic)
            except Exception:
                chi = None
            res[f"{exp}/{cell}"] = dict(N=len(keep), metric=a.metric, friedman_chi2=chi, methods=info, instances=keep)
            print(f"\n=== {exp} {cell} [{a.metric}]: N={len(keep)} instancias, {len(modes)} metodos, Friedman chi2 = {chi}")
            print(f"   {'metodo':44s} {'media':>7s} {'rank':>5s} {'dif vs MOEA/D':>13s} {'A12':>5s} {'melhor':>9s} {'p_Holm':>8s} {'MaxSum/ref':>10s} {'tamanho med(max)':>16s}")
            for m in sorted(modes, key=lambda m: info[m]["rank"]):
                i = info[m]
                s = f"{i['size_median']:.0f} ({i['size_max']})" if i["size_median"] is not None else "-"
                ph = f"{i['p_holm']:.4f}" if "p_holm" in i else "   -"
                print(f"   {LAB.get(m, m):44s} {i['mean']:7.3f} {i['rank']:5.2f} {i['diff_vs_moead']:+13.3f} {i['a12']:5.2f} {i['better']:4d}/{len(keep):<4d} {ph:>8s} "
                      f"{i['maxsum_ratio_median']:10.2f} {s:>16s}")
    suffix = "" if a.metric == "hv_norm" else f"_{a.metric}"
    (ROOT / "results" / "analysis" / f"e18{suffix}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
