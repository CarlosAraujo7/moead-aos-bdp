"""Fila 4 (7 out): espera o run_e10 terminar e amplia E4 e E5 em n=500 para cobrir tambem as instancias folgadas (b03/k03).
    python run_queue4.py
"""
import subprocess, sys, time
from pathlib import Path
import psutil
ROOT = Path(__file__).parent
def busy():
    return any(any("run_e10.py" in x for x in (p.info["cmdline"] or [])) for p in psutil.process_iter(["cmdline"]))
quiet = 0
while quiet < 3:
    quiet = 0 if busy() else quiet + 1
    time.sleep(20)
for cmd in ([sys.executable, "run_e4.py", "--workers", "8", "--reps", "3"], [sys.executable, "run_e5.py", "--workers", "8", "--reps", "3"]):
    print("[fila4]", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)
print("FILA4 FIM", flush=True)
