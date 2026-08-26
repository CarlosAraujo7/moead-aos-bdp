"""
export_results.py — Gera summary.csv e figuras a partir dos checkpoints.

Uso:
    python export_results.py           # processa results/runs/ -> results/full_experiment/
    python export_results.py --analyze # tambem executa analysis.py ao final
"""

from __future__ import annotations
import sys, json, csv, argparse, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import hypervolume, set_coverage, epsilon_indicator, pareto_front
# BUGFIX (unificacao de indicadores e modos): este modulo tinha sua propria
# logica de ponto de referencia (f1_min*0.9, f2_min*0.9), divergente da usada
# em run_full.py (build_ref_point: nadir + margem ABSOLUTA de 1%). A versao
# multiplicativa quebra quando f1_min ou f2_min == 0 -- o que acontece de
# fato: instancias MDG-c tem MaxMin=0 legitimo (~0.2% das distancias sao
# exatamente zero por construcao), entao 0*0.9=0 colapsa a margem de
# seguranca do HV exatamente nas instancias onde ela mais importa.
# Tambem tinha sua PROPRIA copia de ALL_MODES/MODE_LABEL, divergente da de
# experiment.py -- por isso nao sabia processar os modos "dyts_hv"/
# "dyts_scalar" adicionados nesta sessao. Ambos agora importados de
# experiment.py, fonte unica de verdade.
from experiment import ALL_MODES, MODE_LABEL, build_ref_point, compute_indicators
from operators import OPERATOR_NAMES


def main(args):
    runs_dir = ROOT / "results" / "runs"
    out_dir  = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Leitura dos checkpoints
    # -----------------------------------------------------------------------
    t0 = time.time()
    files = sorted(runs_dir.glob("*.json"))
    print(f"Lendo {len(files)} arquivos de checkpoint...", flush=True)

    by_inst: dict[str, list[dict]] = {}
    for i, fp in enumerate(files):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} lidos...", flush=True)
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [AVISO] Erro ao ler {fp.name}: {e}")
            continue
        by_inst.setdefault(d["instance"], []).append(d)

    print(f"  {len(by_inst)} instancias unicas. ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # 2. Calculo de indicadores por instancia
    # -----------------------------------------------------------------------
    print("Calculando indicadores de qualidade...", flush=True)
    summary_rows: list[dict] = []
    inst_names = sorted(by_inst.keys())

    for ii, inst_name in enumerate(inst_names):
        runs = by_inst[inst_name]
        if (ii + 1) % 10 == 0:
            print(f"  {ii+1}/{len(inst_names)}: {inst_name}", flush=True)

        # Frente de referencia conjunta (todos os metodos, todas as replicas)
        all_pfs = []
        for r in runs:
            pf_raw = r.get("pareto_front", [])
            if pf_raw:
                pf = np.array(pf_raw)
                if pf.ndim == 2 and pf.shape[1] == 2 and len(pf) > 0:
                    all_pfs.append(pf)

        if not all_pfs:
            continue

        combined  = np.vstack(all_pfs)
        ref_front = pareto_front(combined)
        ref_pt    = build_ref_point(all_pfs)
        hv_ref    = hypervolume(ref_front, ref_pt)

        # Indicadores por metodo
        for mode in ALL_MODES:
            mode_runs = [r for r in runs if r["mode"] == mode]
            if not mode_runs:
                continue

            n_ops = len(OPERATOR_NAMES)
            hvs, hvns, covs, epss, psizes, times = [], [], [], [], [], []
            op_c = np.zeros(n_ops)
            op_q = np.zeros(n_ops)

            for r in mode_runs:
                pf_raw = r.get("pareto_front", [])
                if not pf_raw:
                    continue
                pf = np.array(pf_raw)
                if pf.ndim != 2 or pf.shape[1] != 2 or len(pf) == 0:
                    continue
                ind = compute_indicators(pf, ref_front, ref_pt)
                hvs.append(ind["hv"])
                hvns.append(ind["hv_norm"])
                covs.append(ind["coverage"])
                epss.append(ind["epsilon"])
                psizes.append(r.get("pareto_size", len(pf)))
                times.append(r.get("elapsed_s", 0.0))
                oc = r.get("operator_counts")
                oq = r.get("operator_q_values")
                if oc and len(oc) == n_ops:
                    op_c += np.array(oc)
                if oq and len(oq) == n_ops:
                    op_q += np.array(oq)

            if not hvs:
                continue

            nr = len(hvs)
            summary_rows.append({
                "instance":         inst_name,
                "family":           mode_runs[0].get("family", ""),
                "n":                mode_runs[0]["n"],
                "m":                mode_runs[0]["m"],
                "mode":             mode,
                "label":            MODE_LABEL[mode],
                "n_reps":           nr,
                "hv_mean":          round(float(np.mean(hvs)),   5),
                "hv_std":           round(float(np.std(hvs)),    5),
                "hv_norm_mean":     round(float(np.mean(hvns)), 5),
                "coverage_mean":    round(float(np.mean(covs)),  5),
                "coverage_std":     round(float(np.std(covs)),   5),
                "epsilon_mean":     round(float(np.mean(epss)),  5),
                "epsilon_std":      round(float(np.std(epss)),   5),
                "pareto_size_mean": round(float(np.mean(psizes)), 2),
                "time_mean_s":      round(float(np.mean(times)), 4),
                "op_counts_mean":   json.dumps((op_c / nr).round(2).tolist()),
                "op_qvals_mean":    json.dumps((op_q / nr).round(6).tolist()),
            })

    # -----------------------------------------------------------------------
    # 3. Grava summary.csv
    # -----------------------------------------------------------------------
    if not summary_rows:
        print("[AVISO] Nenhuma linha gerada.")
        return

    out_csv = out_dir / "summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    n_inst  = len(set(r["instance"] for r in summary_rows))
    n_modes = len(set(r["mode"]     for r in summary_rows))
    print(f"\nSummary exportado: {out_csv}")
    print(f"  {len(summary_rows)} linhas | {n_inst} instancias | {n_modes} metodos")
    print(f"  Tempo total: {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # 4. Exibe ranking rapido
    # -----------------------------------------------------------------------
    print("\n=== RANKING HV MEDIO ===")
    by_mode: dict[str, list[float]] = {}
    by_mode_eps: dict[str, list[float]] = {}
    for r in summary_rows:
        by_mode.setdefault(r["mode"], []).append(r["hv_mean"])
        by_mode_eps.setdefault(r["mode"], []).append(r["epsilon_mean"])

    ranked = sorted(by_mode.items(), key=lambda x: -np.mean(x[1]))
    print(f"  {'Metodo':<28} {'HV medio':>12}  {'Eps medio':>10}  {'n':>4}")
    for mode, hvs_list in ranked:
        eps_list = by_mode_eps[mode]
        print(f"  {MODE_LABEL[mode]:<28} {np.mean(hvs_list):>12.2f}  "
              f"{np.mean(eps_list):>10.4f}  {len(hvs_list):>4}")

    # -----------------------------------------------------------------------
    # 5. Opcional: executa analysis.py
    # -----------------------------------------------------------------------
    if args.analyze:
        print("\nExecutando analysis.py...")
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "src" / "analysis.py"),
             "--csv", str(out_csv)],
            check=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", action="store_true",
                   help="Executa analysis.py ao final para gerar figuras e testes.")
    main(p.parse_args())
