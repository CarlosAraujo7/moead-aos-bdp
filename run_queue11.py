"""Fila 11 (13 out 2026): espera a fila 10 (E21) terminar e roda o E20 nas celulas CDP-L e GDP-L (MOEA/D semeado por VNS 30 e 50% de T), com a maquina
livre (8 workers). Idempotente. NAO importe este arquivo.

    python run_queue11.py [workers]
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def busy() -> bool:
    import psutil
    for p in psutil.process_iter(["cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if "run_queue10.py" in c or "run_e21.py" in c or "--catalog" in c or "test_E7" in c:
            return True
    return False


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "8"
    while busy():
        time.sleep(180)
    print("maquina livre; iniciando fila 11", flush=True)
    r = subprocess.run(["python", "run_e20.py", "--cells", "CDP-L", "GDP-L", "--modes", "m_hyb_vns30,m_hyb_vns50", "--workers", w, "--reps", "3"], cwd=str(ROOT))
    print("FIM FILA 11" if r.returncode == 0 else f"ERRO {r.returncode}", flush=True)
