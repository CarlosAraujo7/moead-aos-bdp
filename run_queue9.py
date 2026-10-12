"""Fila 9 (11 out 2026): E18 -- tamanho do conjunto como dimensao explorada (operadores que mudam o tamanho; CDP/GDP com limite de cardinalidade).
Idempotente (um arquivo por execucao). Celulas pequenas primeiro (rapidas), depois as de n=500.

    python run_queue9.py [workers]
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
w = sys.argv[1] if len(sys.argv) > 1 else "7"


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print("ERRO", cmd, r.returncode, flush=True)
        sys.exit(r.returncode)


for exp, cells in (("free", ["CDP-S", "GDP-S"]), ("bound2", ["CDP-S", "GDP-S"]), ("free", ["CDP-L", "GDP-L"]), ("bound2", ["CDP-L", "GDP-L"]),
                   ("bound15", ["CDP-S", "GDP-S"])):
    run(["python", "run_e18.py", "--exp", exp, "--cells", *cells, "--workers", w])
print("FIM FILA 9", flush=True)
