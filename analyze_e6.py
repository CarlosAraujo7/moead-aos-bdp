"""E6: amostra de escalabilidade (run_e6.py): MDG-b n=2000 e MDG-c n=3000, regime normalizado, 3 replicas, 6 metodos.
Referencia por instancia = frente nao-dominada da uniao de todos os runs de E6 daquela instancia (ponto de referencia 1% abaixo do pior valor);
hv_norm = HV(A)/HV(ref). Com 5 instancias a analise e DESCRITIVA (sem testes): hv_norm medio por metodo e instancia (media das replicas),
rank por instancia, tamanho da frente e tempo mediano por run.

    python analyze_e6.py
Saida: terminal + results/analysis/e6_scalability.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import hypervolume, pareto_front  # noqa: E402

RUNS = ROOT / "results" / "e6_scalability" / "runs"
MODES = ["random", "hypervolume", "dyts_hv", "nsga2", "grasp_pr", "vns_tchebycheff"]
LAB = {"random": "Uniform", "hypervolume": "UCB-HV", "dyts_hv": "GTS-HV", "nsga2": "NSGA-II", "grasp_pr": "GRASP-PR", "vns_tchebycheff": "VNS-Tcheb."}
REPS = 3


def main():
    names = sorted({f.name.split("_random_rep")[0] for f in RUNS.glob("*_random_rep*.json")})
    ok = [n for n in names if all((RUNS / f"{n}_{m}_rep{k:02d}.json").exists() for m in MODES for k in range(REPS))]
    print(f"{len(ok)} instancias completas de {len(names)} com algum run: {ok}")
    if not ok:
        return
    H, T, S = {}, {}, {}
    for n in ok:
        runs = {(m, k): json.load(open(RUNS / f"{n}_{m}_rep{k:02d}.json")) for m in MODES for k in range(REPS)}
        fronts = {key: np.array(r["pareto_front"], float).reshape(-1, 2) for key, r in runs.items()}
        allp = np.vstack(list(fronts.values()))
        rf = pareto_front(allp)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        rp = lo - 0.01 * np.maximum(hi - lo, [1.0, 1e-9])
        hr = hypervolume(rf, rp)
        H[n] = {m: float(np.mean([hypervolume(pareto_front(fronts[(m, k)]), rp) / hr for k in range(REPS)])) for m in MODES}
        T[n] = {m: float(st.median(runs[(m, k)]["elapsed_s"] for k in range(REPS))) for m in MODES}
        S[n] = {m: float(np.mean([len(fronts[(m, k)]) for k in range(REPS)])) for m in MODES}
    print("\nhv_norm (media das 3 replicas) por instancia")
    print(f"{'instancia':24s} " + " ".join(f"{LAB[m]:>10s}" for m in MODES))
    for n in ok:
        print(f"{n:24s} " + " ".join(f"{H[n][m]:10.3f}" for m in MODES))
    R = np.array([rankdata([-H[n][m] for m in MODES]) for n in ok])
    print("\nrank medio (de 6) / hv_norm medio / tempo mediano por run (s) / |frente| medio")
    out = {}
    for j in np.argsort(R.mean(axis=0)):
        m = MODES[j]
        row = dict(rank=float(R[:, j].mean()), hv=float(np.mean([H[n][m] for n in ok])), time_median=float(st.median(T[n][m] for n in ok)),
                   front=float(np.mean([S[n][m] for n in ok])))
        out[m] = row
        print(f"  {LAB[m]:11s} rank {row['rank']:.2f}  hv_norm {row['hv']:.3f}  tempo {row['time_median']:8.0f}  |PF| {row['front']:.1f}")
    (ROOT / "results" / "analysis" / "e6_scalability.json").write_text(
        json.dumps(dict(instances=ok, hv_norm=H, time=T, front=S, summary=out), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
