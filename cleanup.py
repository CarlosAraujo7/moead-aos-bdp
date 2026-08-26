#!/usr/bin/env python3
"""
cleanup.py -- Remove resultados e arquivos obsoletos do projeto.

Execute UMA VEZ antes de iniciar o experimento completo:

    python cleanup.py [--dry-run]

--dry-run   Apenas lista o que seria deletado, sem remover nada.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# Diretorios de resultados antigos a remover
OLD_RESULT_DIRS = [
    "results/analysis",
    "results/baseline",
    "results/comparison_full",
    "results/convergence",
    "results/full_experiment",
    "results/mdg-a",
    "results/scalability",
    "results/test_all",
    "results/smoke_test",
]

# Arquivos de codigo obsoletos a remover
OLD_SRC_FILES = [
    "src/generate_instances.py",
    "src/convergence_exp.py",
]

# Arquivos de checkpoint de experimentos anteriores em results/runs/
# (familias que NAO estao no catalogo atual e portanto sao obsoletas: GKD-a,
# GKD-d, GKD-b (nunca existiu -- ver instances/MDPLIB_2_0.pdf), MDG-c* etc.
# *ATENCAO: mantenha esta lista sincronizada com instances/catalog.csv --
# use `python3 -c "import csv; print(sorted({r['family'] for r in csv.DictReader(open('instances/catalog.csv'))}))"`
# para conferir. Atualizado nesta sessao: catalogo agora tem 183 instancias em
# 11 familias (adicionadas MDG-c e as 4 familias real_* -- ver README/changelog).
VALID_FAMILIES = {
    "SOM-a", "SOM-b", "MDG-a", "MDG-b", "MDG-c", "GKD-c", "MGPO",
    "real_sensor", "real_facility", "real_portfolio", "real_team",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Lista o que seria deletado sem remover nada.")
    args = parser.parse_args()
    dry = args.dry_run

    prefix = "[DRY-RUN] " if dry else ""
    removed = 0
    freed   = 0

    print(f"{'=== DRY-RUN: nada sera deletado ===' if dry else '=== LIMPEZA DO PROJETO ==='}")
    print()

    # 1. Diretorios de resultados antigos
    print("--- Diretorios de resultados antigos ---")
    for rel in OLD_RESULT_DIRS:
        p = ROOT / rel
        if p.exists():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            print(f"  {prefix}Removendo {rel}/  ({size//1024} KB)")
            if not dry:
                shutil.rmtree(p, ignore_errors=True)
            freed += size
            removed += 1
        else:
            print(f"  (ja ausente) {rel}/")

    # 2. Arquivos de codigo obsoletos
    print()
    print("--- Arquivos de codigo obsoletos ---")
    for rel in OLD_SRC_FILES:
        p = ROOT / rel
        if p.exists():
            size = p.stat().st_size
            print(f"  {prefix}Removendo {rel}  ({size} bytes)")
            if not dry:
                p.unlink()
            freed += size
            removed += 1
        else:
            print(f"  (ja ausente) {rel}")

    # 3. Checkpoints de familias que nao estao mais no catalogo
    print()
    print("--- Checkpoints de instancias fora do catalogo (results/runs/) ---")
    runs_dir = ROOT / "results" / "runs"
    stale = []
    if runs_dir.exists():
        for fp in runs_dir.glob("*.json"):
            # Nome: {inst}_{mode}_rep{N}.json
            # Familia e inferida pelo prefixo do nome (ex: GKD-b_, SOM-a_, ...)
            found = False
            for fam in VALID_FAMILIES:
                if fp.stem.startswith(fam):
                    found = True
                    break
            if not found:
                stale.append(fp)

    if stale:
        total_stale = sum(f.stat().st_size for f in stale)
        print(f"  {prefix}Removendo {len(stale)} arquivos obsoletos ({total_stale//1024} KB)")
        if not dry:
            for fp in stale:
                fp.unlink(missing_ok=True)
        freed += total_stale
        removed += len(stale)
    else:
        print("  (nenhum checkpoint obsoleto encontrado)")

    # 4. __pycache__
    print()
    print("--- __pycache__ ---")
    for cache_dir in ROOT.rglob("__pycache__"):
        if cache_dir.is_dir():
            size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            print(f"  {prefix}Removendo {cache_dir.relative_to(ROOT)}  ({size//1024} KB)")
            if not dry:
                shutil.rmtree(cache_dir, ignore_errors=True)
            freed += size

    print()
    if dry:
        print(f"DRY-RUN concluido. Seria liberado: {freed//1024} KB em {removed} itens.")
        print("Execute sem --dry-run para aplicar a limpeza.")
    else:
        print(f"Limpeza concluida. Liberado: {freed//1024} KB.")
        print("O projeto esta pronto para o experimento completo.")


if __name__ == "__main__":
    main()
