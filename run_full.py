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
    CAPACITATED_COMPATIBLE_MODES,
    load_instances, load_capacitated_instances, run_one,
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
    inst_name, inst_family, inst_n, inst_m, inst_filepath, inst_problem, mode, rep, cfg_dict, runs_dir = args
    runs_dir = Path(runs_dir)
    run_file = runs_dir / f"{inst_name}_{mode}_rep{rep:02d}.json"

    # Checkpoint: pula se ja existe E a configuracao nao mudou. Runs gravados antes de
    # 1 out 2026 nao tem config_hash (legado): sao aceitos como estao. Se o hash existe e
    # difere do atual, o resultado e velho para esta configuracao -> move para _stale/ e re-executa.
    if run_file.exists():
        try:
            with run_file.open() as fh:
                data = json.load(fh)
            from experiment import ExperimentConfig, config_fingerprint, config_hash
            _cfg = ExperimentConfig(**cfg_dict)
            cur = config_hash(config_fingerprint(_cfg, mode, inst_n))
            # legado (sem hash) e sempre regime NAO normalizado: nunca aceitar como normalizado
            _legacy_bad = ("config_hash" not in data) and _cfg.normalize_objectives and mode not in (
                "nsga2", "grasp", "grasp_pr", "vns_pareto")
            if "config_hash" in data or _legacy_bad:
                if _legacy_bad or data["config_hash"] != cur:
                    stale = runs_dir / "_stale"
                    stale.mkdir(exist_ok=True)
                    run_file.replace(stale / run_file.name)
                    print(f"[STALE] {run_file.name}: config {data.get('config_hash', 'legado')} != atual {cur}; "
                          f"movido para _stale/ e re-executado", flush=True)
                    raise RuntimeError("stale")
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
        from experiment import ExperimentConfig, run_one, seed_for, config_fingerprint, config_hash
        if inst_problem == "MDP":
            from problem import MDPInstance
            inst = MDPInstance(inst_filepath)
        else:
            from capacitated_problem import CDPInstance, GDPInstance
            cls = CDPInstance if inst_problem == "CDP" else GDPInstance
            inst = cls(inst_filepath)
            # Forca .name/.family a baterem com o catalogo (fonte de verdade):
            # o construtor infere .family da pasta fisica ("CDP (Const.)" /
            # "GDP (Const.)"), nao da familia real (GKD-b/GKD-c/MDG-b/SOM-a).
            inst.name   = inst_name
            inst.family = inst_family
        cfg  = ExperimentConfig(**cfg_dict)

        pf, elapsed, hv_history, op_counts, op_qvals = run_one(inst, cfg, mode, rep)

        run_data = {
            "instance": inst_name, "family": inst_family,
            "n": inst_n, "m": inst_m,
            "mode": mode, "label": MODE_LABEL.get(mode, mode),
            "rep": rep,
            "seed": seed_for(inst_name, mode, rep),
            "elapsed_s": round(elapsed, 4),
            "pareto_size": len(pf),
            "pareto_front": pf.tolist(),
            "hv_history": hv_history,
            "operator_counts": op_counts,
            "operator_q_values": op_qvals,
            "operator_names": OPERATOR_NAMES,
            "config": config_fingerprint(cfg, mode, inst_n),
            "config_hash": config_hash(config_fingerprint(cfg, mode, inst_n)),
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
    parser.add_argument("--problem",  default="MDP", choices=["MDP", "CDP", "GDP"],
                        help="MDP (cardinalidade fixa, instances/catalog.csv) ou "
                             "CDP/GDP (tamanho variavel, instances/catalog_capacitated.csv)")
    parser.add_argument("--families", default=None,
                        help="padrao: 5 familias originais p/ MDP; "
                             "GKD-b,GKD-c,MDG-b,SOM-a p/ CDP/GDP")
    parser.add_argument("--modes",    default=None,
                        help="padrao: todos os 12 p/ MDP; os 7 compativeis "
                             "com MOEA/D-AOS p/ CDP/GDP")
    parser.add_argument("--reps",     type=int, default=10)
    parser.add_argument("--workers",  type=int, default=8,
                        help="padrao 8: benchmark limpo de 1 out 2026 (i5-12450H, maquina ociosa, n=150 normalizado): "
                             "vazao 4,3 runs/min com 1 worker; 13,2 com 4; 15,8 com 6; 16,1 com 8; 19,7 com 12. "
                             "elapsed_s so e comparavel entre execucoes com o mesmo numero de workers")
    parser.add_argument("--max-n",    type=int, default=9999, dest="max_n")
    parser.add_argument("--min-n",    type=int, default=0,    dest="min_n")
    parser.add_argument("--results",  default=None,
                        help="padrao: results/ p/ MDP; "
                             "results/capacitated/<PROBLEM>/ p/ CDP/GDP")
    parser.add_argument("--no-scaling", action="store_true", dest="no_scaling")
    parser.add_argument("--sample", type=int, default=0,
                        help="se >0, usa so N instancias espacadas uniformemente na ordem do catalogo (apos os filtros); pilotos")
    parser.add_argument("--normalize", action="store_true",
                        help="Tchebycheff com objetivos normalizados (MOEA/D-AOS e VNS-Tchebycheff). Resultados vao para "
                             "results/capacitated_norm/<PROBLEMA> (CDP/GDP) ou results_norm (MDP), separados do catalogo historico")
    parser.add_argument("--ucb-c",    type=float, default=0.5, dest="ucb_c")
    parser.add_argument("--ucb-window", type=int, default=50, dest="ucb_window")
    parser.add_argument("--log-every", type=int, default=50, dest="log_every",
                        help="Frequencia de logging do HV (0=desativado, 50=recomendado)")
    args = parser.parse_args()

    if args.families is None:
        args.families = ("SOM-a,SOM-b,GKD-c,MDG-a,MGPO" if args.problem == "MDP"
                          else "GKD-b,GKD-c,MDG-b,SOM-a")
    if args.modes is None:
        args.modes = (",".join(ALL_MODES) if args.problem == "MDP"
                      else ",".join(CAPACITATED_COMPATIBLE_MODES))
    if args.results is None:
        if args.normalize:
            args.results = (str(_root / "results_norm") if args.problem == "MDP"
                             else str(_root / "results" / "capacitated_norm" / args.problem))
        else:
            args.results = (str(_root / "results") if args.problem == "MDP"
                             else str(_root / "results" / "capacitated" / args.problem))

    results_dir = Path(args.results)
    runs_dir    = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    instances_root = _root / "instances"
    catalog = (instances_root / "catalog.csv" if args.problem == "MDP"
               else instances_root / "catalog_capacitated.csv")

    modes_requested = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes_requested:
        if m not in ALL_MODES:
            print(f"[AVISO] Modo desconhecido ignorado: {m}")
    modes = [m for m in modes_requested if m in ALL_MODES]
    if args.problem != "MDP":
        modes_before_cap = modes
        modes = [m for m in modes if m in CAPACITATED_COMPATIBLE_MODES]
        for m in modes_before_cap:
            if m not in CAPACITATED_COMPATIBLE_MODES:
                print(
                    f"[AVISO] Modo {m!r} nao suportado para --problem {args.problem} "
                    f"(instancias de tamanho variavel) -- ignorado. "
                    f"Modos compativeis: {', '.join(CAPACITATED_COMPATIBLE_MODES)}"
                )

    # Config serializada para os processos filhos
    cfg_dict = dict(
        families=args.families,
        max_n=args.max_n, min_n=args.min_n,
        n_reps=args.reps, modes=tuple(modes),
        use_budget_scaling=not args.no_scaling,
        normalize_objectives=args.normalize,
        ucb_c=args.ucb_c, ucb_window=args.ucb_window,
        log_every=args.log_every,
        results_dir=results_dir,
    )

    if args.problem == "MDP":
        instances = load_instances(catalog, instances_root, args.families, args.min_n, args.max_n)
    else:
        instances = load_capacitated_instances(
            catalog, instances_root, args.problem, args.families, args.min_n, args.max_n
        )
    if not instances:
        print("[ERRO] Nenhuma instancia encontrada.")
        sys.exit(1)
    if args.sample and args.sample < len(instances):
        import numpy as _np
        _idx = sorted(set(_np.linspace(0, len(instances) - 1, args.sample).round().astype(int).tolist()))
        instances = [instances[i] for i in _idx]

    instances_meta = [
        {"name": i.name, "family": i.family, "n": i.n, "m": getattr(i, "m", None),
         "problem": args.problem, "filepath": str(i.filepath)}
        for i in instances
    ]

    # Monta lista de tarefas
    tasks = []
    for meta in instances_meta:
        for mode in modes:
            for rep in range(args.reps):
                tasks.append((
                    meta["name"], meta["family"], meta["n"], meta["m"],
                    meta["filepath"], meta["problem"], mode, rep, cfg_dict, str(runs_dir),
                ))

    total      = len(tasks)
    done_count = 0
    skipped    = 0
    errors     = 0
    all_records = []

    print(f"Problema   : {args.problem}")
    print(f"Instancias : {len(instances)}")
    print(f"Familias   : {args.families}")
    print(f"Modos      : {modes}")
    print(f"Replicas   : {args.reps}")
    print(f"Workers    : {args.workers}")
    print(f"Tarefas    : {total}  (workers={args.workers})")
    print(f"Budget sc. : {not args.no_scaling}")
    print(f"Normalizado: {args.normalize}")
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
