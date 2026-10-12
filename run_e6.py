"""E6: amostra de escalabilidade do MDP -- MDG-b n=2000 (3 instancias) e MDG-c n=3000 (2 instancias), regime normalizado, 3 replicas,
6 metodos (Uniform, UCB-HV, GTS-HV, NSGA-II, GRASP-PR, VNS-Tchebycheff). Orcamento: budget_for(n>1000) (T=100, 600 geracoes, ...).
Os dois grupos rodam em paralelo (4 workers cada). Grava em results/e6_scalability/runs (FORA da referencia do MDP e das analises principais).
Idempotente (checkpoint por arquivo em run_full.py). Analise: analyze_e6.py.

    python run_e6.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MODES = "random,hypervolume,dyts_hv,nsga2,grasp_pr,vns_tchebycheff"
base = [sys.executable, "run_full.py", "--problem", "MDP", "--min-n", "2000", "--reps", "3", "--workers", "4",
        "--normalize", "--modes", MODES, "--results", "results/e6_scalability"]
procs = [
    subprocess.Popen(base + ["--families", "MDG-c", "--sample", "2"], cwd=ROOT),
    subprocess.Popen(base + ["--families", "MDG-b", "--sample", "3"], cwd=ROOT),
]
codes = [p.wait() for p in procs]
print("E6 FIM, codigos de saida:", codes, flush=True)
