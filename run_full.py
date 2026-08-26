#!/usr/bin/env python3
"""
run_full.py -- Executor paralelo do experimento completo MOEA/D-AOS vs. baselines.

Uso:
    python run_full.py [opcoes]

Opcoes principais:
    --families  GKD-c,SOM-a,...   familias a processar (padrao: as 5 originais do artigo)
                                   (catalogo completo: 183 instancias em 11 familias --
                                    ver instances/catalog.csv; inclui GKD-c, MDG-a, MDG-b,
                                    MDG-c, MGPO, SOM-a, SOM-b e as 4 familias real_*)
    --modes     hypervolume,scalar,...  modos a executar (padrao: todos 8)
    --reps      N                 replicas por instancia/modo (padrao: 10)
    --workers   N                 processos paralelos       (padrao: 8)
    --max-n     N                 filtro de tamanho maximo  (padrao: 9999)
    --min-n     N                 filtro de tamanho minimo  (padrao: 0)
    --results   DIR               diretorio de saida        (padrao: results/)
    --no-scaling                  desativa budget scaling por n

Checkpoint: cada replica e salva imediatamente ao concluir em results/runs/.
Se interrompido, o script detecta os arquivos existentes e pula as replicas ja feitas.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
import csv
from pathlib import Path

# ------------------------------------------------------------------ #
# Configuracao de sys.path para importar src/                        #
# ------------------------------------------------------------------ #
_root = Path(__file__).parent
_src  = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from experiment import (
    ExperimentConfig, ALL_MODES, MODE_LABEL,
    load_instances, run_one,
    compute_reference_front, build_ref_point, compute_indicators,
)
from operators import OPERATOR_NAMES


# ------------------------------------------------------------------ #
# Tarefa atomica                                                      #
# ------------------------------------------------------------------ #

def _task(args):
    """Executa uma unica replica e persiste o resultado em JSON.

    Retorna um dicionario de metadados ou None se ja concluida/erro.
    """
    inst_name, inst_family, inst_n, inst_m, inst_filepath, mode, rep, cfg_dict, runs_dir = args
    runs_dir = Path(runs_dir)
    run_file = runs_dir / f"{inst_name}_{mode}_rep{rep:02d}.json"

    # Checkpoint: pula se ja existe
    if run_file.exists():
        try:
            with run_file.open() as fh:
                data = json.load(fh)
            return {
                "instance": data["instance"], "family": data["family"],
                "n": data["n"], "m": data["m"],
                "mode": data["mode"], "label": data["label"],
                "rep": data["rep"], "elapsed_s": data["elapsed_s"],
                "pareto_size": data["pareto_size"],
                "_pf_list": data["pareto_front"],
                "_skipped": True,
            }
        except Exception:
            pass  # arquivo corrompido -> reexecuta

    try:
        # Reconstroi instancia e config no processo filho
        from problem import MDPInstance
        from experiment import ExperimentConfig, run_one
        inst = MDPInstance(inst_filepath)
        cfg  = ExperimentConfig(**cfg_dict)

        pf, elapsed, hv_history, op_counts, op_qvals = run_one(inst, cfg, mode, rep)

        run_data = {
            "instance": inst_name, "family": inst_family,
            "n": inst_n, "m": inst_m,
            "mode": mode, "label": MODE_LABEL.get(mode, mode),
            "rep": rep,
            "seed": hash((inst_name, mode, rep)) % (2**31),
            "elapsed_s": round(elapsed, 4),
            "pareto_size": len(pf),
            "pareto_front": pf.tolist(),
            "hv_history": hv_history,
            "operator_counts": op_counts,
            "operator_q_values": op_qvals,
            "operator_names": OPERATOR_NAMES,
        }
        with run_file.open("w") as fh:
            json.dump(run_data, fh, indent=2)

        return {
            "instance": inst_name, "family": inst_family,
            "n": inst_n, "m": inst_m,
            "mode": mode, "label": MODE_LABEL.get(mode, mode),
            "rep": rep, "elapsed_s": elapsed,
            "pareto_size": len(pf),
            "_pf_list": pf.tolist(),
            "_skipped": False,
        }
    except Exception as exc:
        print(f"[ERRO] {inst_name}/{mode}/rep{rep}: {exc}", flush=True)
        traceback.print_exc()
        return None


# ------------------------------------------------------------------ #
# Resumo final                                                        #
# ------------------------------------------------------------------ #

def write_summary(all_records, instances_meta, modes, results_dir):
    """Calcula indicadores e salva summary.csv."""
    import numpy as np
    from metrics import hypervolume, pareto_front, set_coverage, epsilon_indicator
    from experiment import compute_indicators, compute_reference_front, build_ref_point

    # Agrupa frontes por instancia para calculo do reference front
    fronts_by_inst = {}
    for r in all_records:
        if r is None:
            continue
        key = r["instance"]
        import numpy as _np
        pf = _np.array(r["_pf_list"])
        fronts_by_inst.setdefault(key, []).append(pf)

    summary_rows = []
    for inst_meta in instances_meta:
        iname = inst_meta["name"]
        fronts = fronts_by_inst.get(iname, [])
        if not fronts:
            continue
        ref_front = compute_reference_front(fronts)
        ref_pt    = build_ref_point(fronts)

        inst_recs = [r for r in all_records if r and r["instance"] == iname]
        for mode in modes:
            mode_recs = [r for r in inst_recs if r["mode"] == mode]
            if not mode_recs:
                continue
            import numpy as _np
            hvs, hvns, covs, epss, sizes, times = [], [], [], [], [], []
            hv_ref = hypervolume(ref_front, ref_pt)
            for r in mode_recs:
                approx = _np.array(r["_pf_list"])
                ind = compute_indicators(approx, ref_front, ref_pt)
                hvs.append(ind["hv"])
                hvns.append(ind["hv_norm"])
                covs.append(ind["coverage"])
                epss.append(ind["epsilon"])
                sizes.append(r["pareto_size"])
                times.append(r["elapsed_s"])
            summary_rows.append({
                "instance": iname,
                "family":   inst_meta["family"],
                "n": inst_meta["n"], "m": inst_meta["m"],
                "mode": mode, "label": MODE_LABEL.get(mode, mode),
                "hv_mean":          round(float(_np.mean(hvs)),   6),
                "hv_std":           round(float(_np.std(hvs)),    6),
                "hv_norm_mean":     round(float(_np.mean(hvns)),  6),
                "hv_norm_std":      round(float(_np.std(hvns)),   6),
                "coverage_mean":    round(float(_np.mean(covs)),  5),
                "coverage_std":     round(float(_np.std(covs)),   5),
                "epsilon_mean":     round(float(_np.mean(epss)),  5),
                "epsilon_std":      round(float(_np.std(epss)),   5),
                "pareto_size_mean": round(float(_np.mean(sizes)), 2),
                "time_mean_s":      round(float(_np.mean(times)), 4),
            })

    if not summary_rows:
        print("[AVISO] Nenhuma linha para summary.csv.")
        return

    out_csv = results_dir / "summary.csv"
    fields = [k for k in summary_rows[0].keys()]
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary salvo: {out_csv}  ({len(summary_rows)} linhas)")


# ------------------------------------------------------------------ #
# Runner principal                                                    #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Executor paralelo MOEA/D-AOS")
    parser.add_argument("--families", default="SOM-a,SOM-b,GKD-c,MDG-a,MGPO")
    parser.add_argument("--modes",    default=",".join(ALL_MODES))
    parser.add_argument("--reps",     type=int, default=10)
    parser.add_argument("--workers",  type=int, default=8)
    parser.add_argument("--max-n",    type=int, default=9999, dest="max_n")
    parser.add_argument("--min-n",    type=int, default=0,    dest="min_n")
    parser.add_argument("--results",  default=str(_root / "results"))
    parser.add_argument("--no-scaling", action="store_true", dest="no_scaling")
    parser.add_argument("--ucb-c",    type=float, default=0.5, dest="ucb_c")
    parser.add_argument("--ucb-window", type=int, default=50, dest="ucb_window")
    parser.add_argument("--log-every", type=int, default=50, dest="log_every",
                        help="Frequencia de logging do HV (0=desativado, 50=recomendado)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    runs_dir    = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    catalog        = _root / "instances" / "catalog.csv"
    instances_root = _root / "instances"
    modes          = [m.strip() for m in args.modes.split(",") if m.strip() in ALL_MODES]

    # Config serializada para os processos filhos
    cfg_dict = dict(
        families=args.families,
        max_n=args.max_n, min_n=args.min_n,
        n_reps=args.reps, modes=tuple(modes),
        use_budget_scaling=not args.no_scaling,
        ucb_c=args.ucb_c, ucb_window=args.ucb_window,
        log_every=args.log_every,
        results_dir=results_dir,
    )

    instances = load_instances(catalog, instances_root, args.families, args.min_n, args.max_n)
    if not instances:
        print("[ERRO] Nenhuma instancia encontrada.")
        sys.exit(1)

    instances_meta = [
        {"name": i.name, "family": i.family, "n": i.n, "m": i.m,
         "filepath": str(i.filepath)}
        for i in instances
    ]

    # Monta lista de tarefas
    tasks = []
    for meta in instances_meta:
        for mode in modes:
            for rep in range(args.reps):
                tasks.append((
                    meta["name"], meta["family"], meta["n"], meta["m"],
                    meta["filepath"], mode, rep, cfg_dict, str(runs_dir),
                ))

    total      = len(tasks)
    done_count = 0
    skipped    = 0
    errors     = 0
    all_records = []

    print(f"Instancias : {len(instances)}")
    print(f"Familias   : {args.families}")
    print(f"Modos      : {modes}")
    print(f"Replicas   : {args.reps}")
    print(f"Workers    : {args.workers}")
    print(f"Tarefas    : {total}  (workers={args.workers})")
    print(f"Budget sc. : {not args.no_scaling}")
    print(f"Resultados : {results_dir}")
    print()

    t_start = time.perf_counter()

    # Ctrl+C seguro com Pool
    original_sigint = signal.getsignal(signal.SIGINT)
    interrupted = False

    def _handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[INTERROMPIDO] Aguardando tarefas em curso finalizarem...", flush=True)

    signal.signal(signal.SIGINT, _handler)

    try:
        with mp.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_task, tasks, chunksize=1):
                if interrupted:
                    pool.terminate()
                    break
                done_count += 1
                if result is None:
                    errors += 1
                else:
                    all_records.append(result)
                    if result.get("_skipped"):
                        skipped += 1

                # Progresso a cada 50 tarefas
                if done_count % 50 == 0 or done_count == total:
                    elapsed = time.perf_counter() - t_start
                    eta = (elapsed / done_count) * (total - done_count) if done_count > 0 else 0
                    pct = 100 * done_count / total
                    print(
                        f"  {done_count:5d}/{total}  ({pct:5.1f}%)  "
                        f"skip={skipped}  err={errors}  "
                        f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                        flush=True,
                    )
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    elapsed_total = time.perf_counter() - t_start
    print(
        "\nConcluido: " + str(done_count) + " tarefas em "
        + str(round(elapsed_total, 1)) + "s  "
        + "(skip=" + str(skipped) + ", err=" + str(errors) + ")"
    )

    # Grava summary apenas se ha registros suficientes
    if all_records:
        write_summary(all_records, instances_meta, modes, results_dir)
    else:
        print("[AVISO] Nenhum resultado disponivel para summary.csv.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
