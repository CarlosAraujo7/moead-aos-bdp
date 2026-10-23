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


def foreign_load() -> bool:
    """Ha outro job pesado na maquina (p.ex. outro projeto com `run_full.py --catalog ...`)? Os experimentos com limite de TEMPO (E20, E22) so sao
    comparaveis com o E11 se rodarem com a carga do E11 (8 workers); esperamos ate 8 h e seguimos com aviso."""
    import psutil
    for p in psutil.process_iter(["cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if "python" in c.lower() and ("--catalog" in c or "test_E7" in c):
            return True
    return False


def wait_no_foreign(max_h: float = 8.0):
    t0 = time.time()
    while foreign_load() and time.time() - t0 < max_h * 3600:
        print("job externo detectado; aguardando 10 min", flush=True)
        time.sleep(600)
    if foreign_load():
        print("AVISO: job externo ainda ativo apos a espera; seguindo (tempos podem estar afetados)", flush=True)


def my_workers(total: int = 8) -> int:
    """Mantem 8 processos ativos no total (a carga do E11) quando outro projeto ja usa workers (p.ex. `p16_tamanho_pool.py --workers N`)."""
    import re
    import psutil
    used = 0
    for p in psutil.process_iter(["cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if "p16_tamanho_pool" in c:
            m = re.search(r"--workers\s+(\d+)", c)
            used = max(used, int(m.group(1)) if m else 4)
    return max(2, total - used)


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
    # completa as celulas pequenas do E18 com UCB-HV/GTS-HV + operadores de tamanho (acrescentados depois da 1a passada; so roda o que falta)
    run(["python", "run_e18.py", "--exp", "free", "--cells", "CDP-S", "GDP-S", "--workers", w])
    run(["python", "run_e18.py", "--exp", "bound2", "--cells", "CDP-S", "GDP-S", "--workers", w])
    V = "m_hyb_vns10,m_hyb_vns30,m_hyb_vns50"
    wait_no_foreign()
    run(["python", "run_e20.py", "--cells", "MDP-S", "--modes", "m_ls,m_hyb," + V, "--workers", w, "--reps", "3"])
    run(["python", "run_e20.py", "--cells", "CDP-S", "GDP-S", "--modes", V, "--workers", w, "--reps", "3"])
    run(["python", "run_e20.py", "--cells", "MDP-L", "--modes", V, "--workers", w, "--reps", "3"])
    wait_no_foreign()
    run(["python", "run_e22.py", "--workers", w])
    run(["python", "run_e10.py", "--parts", "E", "--workers", w])
    if not (ROOT / "results" / "e18_size" / "shares_L.json").exists():     # a sonda nao e idempotente por arquivo: so roda se ainda nao existe
        run(["python", "probe_e18_shares.py", "--workers", w])
    run(["python", "run_e21.py", "--workers", str(my_workers(int(w)))])
    print("FIM FILA 10", flush=True)
