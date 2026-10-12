"""Fila 7 (10 out 2026): SMS-EMOA como baseline adicional no protocolo do E11 (ajuste por busca aleatoria + successive halving e teste a T e 5T).
Espera a fila 6 (E15-E17 em results/phase2_aos) terminar: 15 variantes x (60 CDP + 120 GDP) x 3 replicas = 8100 arquivos. Idempotente.

    python run_queue7.py [workers]
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
w = sys.argv[1] if len(sys.argv) > 1 else "8"
VARS = "frr_default,frr_w25,frr_c1,frr_c03,frr_w25_c03,px_random,px_ucbhv,px_dytshv,px_ucbscal,px_dytsscal,px_ucbins,px_fdrep,px_frr,init_a10,init_a00".split(",")
EXPECT = len(VARS) * (60 + 120) * 3
PAT = re.compile(r"_(" + "|".join(VARS) + r")_rep\d\d\.json$")


def n_done():
    return sum(1 for p in ("CDP", "GDP") for f in (ROOT / "results" / "phase2_aos" / p / "runs").iterdir() if PAT.search(f.name))


while n_done() < EXPECT:
    print(f"aguardando E15-E17: {n_done()}/{EXPECT}", flush=True)
    time.sleep(180)
for cmd in (["python", "run_e11_tune.py", "--workers", w, "--methods", "sms"], ["python", "run_e11.py", "--workers", w]):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print("ERRO", cmd, r.returncode, flush=True)
        sys.exit(r.returncode)
print("FIM FILA 7", flush=True)
