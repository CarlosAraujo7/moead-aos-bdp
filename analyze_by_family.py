"""E9: ranks medios (hv_norm, media sobre replicas) por familia de instancias, regime normalizado, K=12 metodos.
Rankeia os 12 metodos em cada instancia e resume por familia (e por familia x tamanho no CDP/GDP). So descritivo (N pequeno por familia).

    python analyze_by_family.py --problem MDP|CDP|GDP [--reps 3] [--metric hv_norm]
Saida: tabela no terminal + results/analysis/by_family_<P>_<metric>.json e .md (apendice do manuscrito).
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from analyze_replicated import load_matrix  # noqa: E402
from experiment import ALL_MODES  # noqa: E402

LABEL = {"hypervolume": "UCB-HV", "thompson_hv": "TS-HV", "dyts_hv": "GTS-HV", "scalar": "UCB-Scal", "thompson_scalar": "TS-Scal",
         "dyts_scalar": "GTS-Scal", "random": "Uniform", "nsga2": "NSGA-II", "grasp": "GRASP-MO", "grasp_pr": "GRASP-PR",
         "vns_pareto": "VNS-Par", "vns_tchebycheff": "VNS-Tch"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["CDP", "GDP", "MDP"], default="MDP")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--metric", choices=["hv_norm", "igd_plus", "eps_plus"], default="hv_norm")
    a = ap.parse_args()
    maxn = {"MDP": 100000, "CDP": 500, "GDP": 500}[a.problem]
    names, cube = load_matrix(a.problem, maxn, a.metric, a.reps)
    M = cube.mean(axis=2)
    sign = -1.0 if a.metric == "hv_norm" else 1.0
    R = np.array([rankdata(sign * r) for r in M])
    cat_file = "catalog.csv" if a.problem == "MDP" else "catalog_capacitated.csv"
    meta = {r["name"]: r for r in csv.DictReader(open(ROOT / "instances" / cat_file))}
    groups = {}
    for i, n in enumerate(names):
        r = meta[n]
        key = r["family"] if a.problem == "MDP" else f"{r['family']} n={r['n']}"
        groups.setdefault(key, []).append(i)
    order = sorted(groups)
    out = {"problem": a.problem, "metric": a.metric, "reps": a.reps, "groups": {}}
    lines = ["| Group | N | " + " | ".join(LABEL[m] for m in ALL_MODES) + " | best AOS | best baseline |", "|---|---|" + "---|" * (len(ALL_MODES) + 2)]
    print(f"{a.problem}: {len(names)} instancias, {a.reps} replicas")
    for g in order:
        idx = groups[g]; avg = R[idx].mean(axis=0)
        aos = [j for j, m in enumerate(ALL_MODES) if j < 7]; base = [j for j in range(len(ALL_MODES)) if j >= 7]
        ba = min(aos, key=lambda j: avg[j]); bb = min(base, key=lambda j: avg[j])
        out["groups"][g] = dict(N=len(idx), avg_ranks=dict(zip(ALL_MODES, map(float, avg))), best_aos=ALL_MODES[ba], best_baseline=ALL_MODES[bb])
        lines.append(f"| {g} | {len(idx)} | " + " | ".join(f"{avg[j]:.2f}" for j in range(len(ALL_MODES))) + f" | {LABEL[ALL_MODES[ba]]} ({avg[ba]:.2f}) | {LABEL[ALL_MODES[bb]]} ({avg[bb]:.2f}) |")
        print(f"{g:22s} N={len(idx):3d}  melhor AOS {LABEL[ALL_MODES[ba]]:9s} {avg[ba]:5.2f} | melhor baseline {LABEL[ALL_MODES[bb]]:9s} {avg[bb]:5.2f} | Uniform {avg[ALL_MODES.index('random')]:5.2f}")
    d = ROOT / "results" / "analysis"
    (d / f"by_family_{a.problem}_{a.metric}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (d / f"by_family_{a.problem}_{a.metric}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
