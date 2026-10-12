"""Regenera TODAS as analises e figuras do manuscrito com a referencia fixa atual (results/reference_fronts/*.json).
NAO chama `reference_sets.py build`: a referencia foi congelada em 6 out 2026 (copia em results/reference_fronts/frozen_2026-10-06/).
Log: results/analysis/regen_log.txt. Idempotente; leva ~1-2 h com a maquina livre.

    python regen_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
LOG = ROOT / "results" / "analysis" / "regen_log.txt"
PY = [sys.executable]
JOBS = []
for prob, hi in (("CDP", 500), ("GDP", 500), ("MDP", 1000)):
    for met in ("hv_norm", "igd_plus", "eps_plus"):
        JOBS.append(PY + ["analyze_replicated.py", "--problem", prob, "--min-n", "0", "--max-n", str(hi), "--reps", "3", "--metric", met])
for prob in ("CDP", "GDP"):
    JOBS.append(PY + ["analyze_replicated.py", "--problem", prob, "--min-n", "0", "--max-n", "150", "--reps", "5", "--metric", "hv_norm"])
    JOBS.append(PY + ["analyze_replicated.py", "--problem", prob, "--min-n", "500", "--max-n", "500", "--reps", "3", "--metric", "hv_norm"])
for prob in ("MDP", "CDP", "GDP"):
    JOBS.append(PY + ["analyze_by_family.py", "--problem", prob, "--reps", "3"])
JOBS.append(PY + ["analyze_e4.py"])
JOBS.append(PY + ["analyze_e5.py"])
JOBS.append(PY + ["analyze_e6.py"])
JOBS.append(PY + ["analyze_e10.py"])
JOBS.append(PY + ["analyze_e10.py", "--metric", "igd_plus"])
JOBS.append(PY + ["analyze_e11.py"])
JOBS.append(PY + ["analyze_e11.py", "--metric", "igd_plus"])
JOBS.append(PY + ["analyze_e11_extremes.py"])
JOBS.append(PY + ["analyze_e13.py"])
for prob in ("CDP", "GDP"):
    JOBS.append(PY + ["analyze_e1_e2.py", "--problem", prob, "--part", "all", "--out", f"results/analysis/e1_e2_e3_{prob}.json"])
    JOBS.append(PY + ["analyze_anchor_v2.py", "--problem", prob])
JOBS.append(PY + ["analyze_aos_summary.py"])
JOBS.append(PY + ["analyze_op_shares.py"])
JOBS.append(PY + ["analyze_e8.py"])
JOBS.append(PY + ["analyze_e14.py"])
JOBS.append(PY + ["analyze_e18.py"])
JOBS.append(PY + ["analyze_e20.py", "--cells", "MDP-S,MDP-L,CDP-S,GDP-S"])
JOBS.append(PY + ["make_figures.py"])
JOBS.append(PY + ["make_manuscript_tables.py"])
JOBS.append(PY + ["make_latex.py", "--build"])
JOBS.append(PY + ["make_docx.py"])
JOBS.append(PY + ["make_docx.py", "Claude outputs/supplementary_material.md", "Claude outputs/supplementary_material.docx"])

with open(LOG, "w", encoding="utf-8") as lg:
    for cmd in JOBS:
        t = time.time()
        lg.write(f"\n$$$ {' '.join(cmd[1:])}\n"); lg.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=lg, stderr=subprocess.STDOUT, env={**__import__('os').environ, "PYTHONIOENCODING": "utf-8"})
        lg.write(f"--- exit {r.returncode} em {(time.time()-t)/60:.1f} min\n"); lg.flush()
        print(f"{' '.join(cmd[1:4])} exit {r.returncode} ({(time.time()-t)/60:.1f} min)", flush=True)
print("REGEN FIM", flush=True)
