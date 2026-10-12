"""Fila 5 (9 out 2026): E14 (mecanismo do orcamento de custo apertado) e depois E8 (ablacao de operadores com replicas). Idempotente.

    python run_queue5.py [workers]
"""
import subprocess
import sys

w = sys.argv[1] if len(sys.argv) > 1 else "8"
for cmd in (["python", "run_e14.py", "--workers", w], ["python", "run_e8.py", "--workers", w]):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("ERRO em", cmd, r.returncode, flush=True)
        sys.exit(r.returncode)
print("FIM FILA 5", flush=True)
