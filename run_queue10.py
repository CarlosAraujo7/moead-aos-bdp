"""Fila 10 (12 out 2026): espera a fila 9 (E18) terminar e depois roda, com a maquina livre (8 workers, o mesmo do E11):
  E20 limpo (MOEA/D semeado por VNS 10/30/50% de T; MDP-S tambem busca local e semeadura por Tabu; MDP-S, CDP-S, GDP-S, MDP-L),
  E10 parte E (restante das MDG-b n=2000: AOS vs uniforme) e E21 (escala completa no MDP: baselines ajustados e hibrido a tempo igual).
Idempotente. NAO importe este arquivo (so executa em __main__).

    python run_queue10.py [workers]
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def waiting_for_queue9() -> bool:
    import psutil
    for p in psutil.process_iter(["cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if "run_queue9.py" in c or "run_e18.py" in c:
            return True
    return False


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print("ERRO", cmd, r.returncode, flush=True)
        sys.exit(r.returncode)


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "8"
    while waiting_for_queue9():
        time.sleep(120)
    print("fila 9 terminada; iniciando fila 10", flush=True)
    V = "m_hyb_vns10,m_hyb_vns30,m_hyb_vns50"
    run(["python", "run_e20.py", "--cells", "MDP-S", "--modes", "m_ls,m_hyb," + V, "--workers", w, "--reps", "3"])
    run(["python", "run_e20.py", "--cells", "CDP-S", "GDP-S", "--modes", V, "--workers", w, "--reps", "3"])
    run(["python", "run_e20.py", "--cells", "MDP-L", "--modes", V, "--workers", w, "--reps", "3"])
    run(["python", "run_e22.py", "--workers", w])
    run(["python", "run_e10.py", "--parts", "E", "--workers", w])
    run(["python", "run_e21.py", "--workers", w])
    print("FIM FILA 10", flush=True)
