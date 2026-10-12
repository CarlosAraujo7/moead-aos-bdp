"""Fila unica e idempotente das rodadas de blindagem/ancora (E1b, E3, E4, E7, E2), em ordem de prioridade/custo. Substitui
wait_and_run_e1b.py e wait_and_run_e7.py (que ficaram sem efeito apos o reinicio de 3 out). O checkpoint por arquivo de
run_phase2.py pula o que ja existe; pode ser relancada a qualquer momento.

    python run_queue.py [workers]
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
W = sys.argv[1] if len(sys.argv) > 1 else "8"
E1B = "fd_replace,fd_replace_c30,fd_replace_c300,fd_scalar,fd_hv,ts_replace,ucb_replace"
E3 = "c_ucbins_w3,c_ucbins_p3,c_ucbins_w3p3,c_tsins_w3p3,c_fdrep_w3p3,c_ucbsrel_w3p3,c_ucbhv_w3p3"
E2 = "b2_random,b2_dytshv,b2_ucbhv,b4_random,b4_dytshv,b4_ucbhv"
E7S = "random_both,random_sbts,dyts_hv_both"
JOBS = [  # fila 2 (6 out): E4 completo (agora com GRASP-MO/PR a tempo igual) -> E2 -> E7 n=500
    ("E4 baselines a tempo igual (CDP+GDP)", "E4", 0, 0, ""),
    ("E2 CDP", "CDP", 0, 150, E2),
    ("E7 CDP n=500", "CDP", 500, 500, "random_both,random_sbts"), ("E7 GDP n=500", "GDP", 500, 500, "random_both"),
]
for name, prob, lo, hi, vs in JOBS:
    t = time.time()
    print(f"[fila] {name} ...", flush=True)
    if prob == "E4":
        subprocess.run([sys.executable, "run_e4.py", "--workers", W, "--reps", "3"], cwd=ROOT, check=False)
    else:
        subprocess.run([sys.executable, "run_phase2.py", "--problem", prob, "--min-n", str(lo), "--max-n", str(hi), "--reps", "3",
                        "--workers", W, "--variants", vs], cwd=ROOT, check=False)
    print(f"[fila] {name} concluido em {(time.time()-t)/60:.0f} min", flush=True)
print("FILA FIM", flush=True)
