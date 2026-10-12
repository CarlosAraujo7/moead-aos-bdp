"""Reavalia, com o NOSSO avaliador (src/capacitated_problem.GDPInstance), as solucoes publicadas do GDP
(pacote Zenodo 18233882; Lozano-Osorio et al. 2022) e compara dispersao/custo/capacidade com o que o artigo reporta.

    python validate_published_solutions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from capacitated_problem import GDPInstance  # noqa: E402

PKG = ROOT / "a pedidos" / "isaaclo97-Generalized-Dispersion-Problem-2c0fa37"


def parse_solution(path: Path):
    lines = path.read_text().strip().split("\n")
    nodes = [int(x) for x in lines[0].split()]
    d = float(lines[1].split(":")[1]); c = float(lines[2].split(":")[1]); cap = float(lines[3].split(":")[1])
    return nodes, d, c, cap


def main():
    rows = []
    for f in sorted((PKG / "solution" / "solutions").glob("*.txt")):
        name = f.stem
        inst_file = ROOT / "instances" / "GDP (Const.)" / f"{name}.txt"
        if not inst_file.exists():
            continue
        nodes, d_pub, c_pub, cap_pub = parse_solution(f)
        inst = GDPInstance(inst_file)
        sol = np.array(sorted(nodes), dtype=np.intp)
        ms, mm = inst.evaluate(sol)
        cost = float(inst.cost[sol].sum()); cap = float(inst.capacity[sol].sum())
        rows.append((name, len(sol), d_pub, mm, c_pub, cost, cap_pub, cap, inst.is_feasible(sol), cost < inst.max_budget, ms))
    print(f"{len(rows)} solucoes publicadas com instancia correspondente no nosso catalogo GDP\n")
    ok_d = sum(abs(r[2] - r[3]) < 1e-6 for r in rows); ok_c = sum(abs(r[4] - r[5]) < 1e-6 for r in rows); ok_cap = sum(abs(r[6] - r[7]) < 1e-6 for r in rows)
    print(f"dispersao igual (publicada vs nosso MaxMin): {ok_d}/{len(rows)}")
    print(f"custo igual: {ok_c}/{len(rows)} | capacidade igual: {ok_cap}/{len(rows)}")
    print(f"viaveis pelo nosso is_feasible: {sum(r[8] for r in rows)}/{len(rows)} | custo < K1 estrito (criterio deles): {sum(r[9] for r in rows)}/{len(rows)}")
    bad = [r for r in rows if abs(r[2] - r[3]) >= 1e-6]
    for r in bad[:8]:
        print(f"  DIFERENTE: {r[0]} |S|={r[1]} publicada={r[2]} nosso MaxMin={r[3]}")
    print(f"tamanho medio da solucao publicada: {np.mean([r[1] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
