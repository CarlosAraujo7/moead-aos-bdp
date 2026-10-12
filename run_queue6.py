"""Fila 6 (10 out 2026): E15 (FRRMAB de Li et al. 2014) E16 (pool envenenado) e E17 (inicializacao) em CDP e GDP n<=150, 3 replicas. Espera a fila 5 (E14 + E8) terminar:
o E8 completo tem 8640 arquivos (CDP 60x16x3 + GDP 120x16x3) em results/e8_operator_ablation. Idempotente.

    python run_queue6.py [workers]
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
w = sys.argv[1] if len(sys.argv) > 1 else "8"
EXPECT = 8640
VARS = "frr_default,frr_w25,frr_c1,frr_c03,frr_w25_c03,px_random,px_ucbhv,px_dytshv,px_ucbscal,px_dytsscal,px_ucbins,px_fdrep,px_frr,init_a10,init_a00"


def n_e8():
    return sum(len(list((ROOT / "results" / "e8_operator_ablation" / p / "runs").glob("*.json"))) for p in ("CDP", "GDP"))


while n_e8() < EXPECT:
    print(f"aguardando E8: {n_e8()}/{EXPECT}", flush=True)
    time.sleep(120)
for prob in ("CDP", "GDP"):
    cmd = ["python", "run_phase2.py", "--problem", prob, "--max-n", "150", "--reps", "3", "--workers", w, "--variants", VARS]
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print("ERRO", cmd, r.returncode, flush=True)
        sys.exit(r.returncode)
print("FIM FILA 6", flush=True)
