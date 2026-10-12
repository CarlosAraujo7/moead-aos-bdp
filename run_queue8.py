"""Fila 8 (10 out 2026): REEXECUCAO das variantes afetadas pelo artefato de ponto flutuante nas recompensas de 'sucesso' (seletores binarios
Thompson/DYTS e recompensa 'insert'; corrigido em moead_aos.py com tolerancia relativa 1e-9; ver CLAUDE.md §2t). Os runs antigos sao MOVIDOS
(nunca apagados) para pastas `_artifact_20261010/` ao lado de `runs/`; os de results/capacitated_norm e results_norm sao movidos para `_stale/`
pelo proprio run_full.py (o hash de configuracao dos modos thompson_* mudou). Idempotente.

    python run_queue8.py [workers]
"""
import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
w = sys.argv[1] if len(sys.argv) > 1 else "8"

PHASE2_AFFECTED = ("ucb_insert,ts_insert,dyts_insert,ucb_insert_cost,ucb_insert_mix50,ts_insert_mix50,h_ucbins_c005,h_ucbins_c20,c_ucbins_w3,c_ucbins_p3,"
                   "c_ucbins_w3p3,c_tsins_w3p3,px_ucbins,h_tshv_w20,h_tshv_w200,h_tssc_w20,h_tssc_w200,fd_replace,fd_replace_c30,fd_replace_c300,fd_scalar,fd_hv,"
                   "ts_replace,ucb_replace,c_fdrep_w3p3,px_fdrep,frr_default,frr_w25,frr_c1,frr_c03,frr_w25_c03,px_frr").split(",")
# SO sao movidos os arquivos anteriores a correcao (mtime < corte): sem isto, reiniciar a fila moveria de volta os runs novos (acidente de 11 out)
CUT = datetime.datetime(2026, 10, 10, 20, 52).timestamp()
FAM = "SOM-a,SOM-b,GKD-c,MDG-a,MGPO,MDG-b,real_sensor,real_facility,real_portfolio,real_team"


def quarantine(runs: Path, variants):
    dst = runs.parent / "_artifact_20261010"
    pat = re.compile(r"_(" + "|".join(re.escape(v) for v in variants) + r")_rep\d\d\.json$")
    n = 0
    for f in list(runs.iterdir()):
        if f.is_file() and pat.search(f.name) and f.stat().st_mtime < CUT:
            dst.mkdir(exist_ok=True)
            shutil.move(str(f), str(dst / f.name))
            n += 1
    print(f"quarentena {runs}: {n} arquivos movidos", flush=True)


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print("ERRO", cmd, r.returncode, flush=True)
        sys.exit(r.returncode)


for p in ("CDP", "GDP"):
    quarantine(ROOT / "results" / "phase2_aos" / p / "runs", PHASE2_AFFECTED)
    quarantine(ROOT / "results" / "ablation_normalized" / p / "runs", ["thompson_hv", "thompson_scalar"])
quarantine(ROOT / "results" / "e5_budget_relaxed_norm" / "GDP" / "runs", ["thompson_hv", "thompson_scalar"])

V = ",".join(PHASE2_AFFECTED)
for p in ("CDP", "GDP"):
    run(["python", "run_phase2.py", "--problem", p, "--max-n", "150", "--reps", "3", "--workers", w, "--variants", V])
for p in ("CDP", "GDP"):
    run(["python", "run_ablation_norm.py", "--problem", p, "--max-n", "150", "--workers", w])
run(["python", "run_e5.py", "--workers", w, "--reps", "3"])
TS = "thompson_hv,thompson_scalar"
run(["python", "run_full.py", "--problem", "CDP", "--modes", TS, "--max-n", "150", "--reps", "5", "--workers", w, "--normalize"])
run(["python", "run_full.py", "--problem", "GDP", "--modes", TS, "--max-n", "150", "--reps", "5", "--workers", w, "--normalize"])
run(["python", "run_full.py", "--problem", "CDP", "--modes", TS, "--min-n", "500", "--max-n", "500", "--reps", "3", "--workers", w, "--normalize"])
run(["python", "run_full.py", "--problem", "GDP", "--modes", TS, "--min-n", "500", "--max-n", "500", "--reps", "3", "--workers", w, "--normalize"])
for reps in (1, 2, 3):
    run(["python", "run_full.py", "--problem", "MDP", "--families", FAM, "--modes", TS, "--max-n", "1000", "--reps", str(reps), "--workers", w, "--normalize"])
print("FIM FILA 8", flush=True)
