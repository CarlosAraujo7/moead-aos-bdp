"""Valores de referencia da literatura para o CDP, por instancia do nosso catalogo (b02 e b03).

Fontes (todas em anchors/):
  - cdp_bks_symmetry2026_table1.csv : Tabela 1 de Gomez et al. (2026), Symmetry 18(1):110 (so demanda b02; coluna `bks`).
  - cdp_mdgb_rosati2024_table5.csv  : Tabela 5 de Rosati & Schaerf (2024), ESWA 255:124484: melhor/media/pior de 40 execucoes de Lu et al.
                                      (confere com as Tabelas A.1/A.2 acima), de Mladenovic et al. (2022, COR 139:105622; VNS) e do MNSA, so
                                      MDG-b n=500 (b02 e b03). Eleva `best` e `sota_avg` nessas 20 instancias.
  - cdp_literature_lu2023.csv       : Tabelas A.1 (phi=0,2 = b02) e A.2 (phi=0,3 = b03) de Lu, Martinez-Gavara, Hao & Lai (2023),
                                      ESWA 223:119856 (SBTS): melhor valor de SO/SS1/SS2/SBTS_short/SBTS_long (`lu_best`), media de 40
                                      execucoes do SBTS (`sbts_long_avg`, `sbts_short_avg`) e o valor `f_opt` do Gurobi
                                      (Marti et al. 2021; otimo onde resolvido; ausente para MDG-b).

`reference(name)` devolve, para um nome do nosso catalogo (ex. GKD-c_01_n500_b02_m50):
  best : max(BKS do Symmetry, melhor valor de Lu et al., valor do Gurobi f_opt quando existe) -- melhor valor publicado conhecido
         por nos (otimo comprovado so onde o Gurobi resolveu; o f_opt de Lu et al. vem de Marti et al. 2021)
  sym, lu_best, sbts_avg (SBTS_long, media de 40 execucoes), sbts_short_avg, f_opt (ou None)
  sota_avg : maior media de execucoes de ~300 s entre os solvers publicados disponiveis (SBTS; no MDG-b tambem VNS e MNSA)
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).parent
_CACHE: dict | None = None


def _our_name(lu_name: str, phi: float) -> str:
    tag = "b02" if abs(phi - 0.2) < 1e-9 else "b03"
    return re.sub(r"_m(\d+)$", rf"_{tag}_m\1", lu_name)


def _load() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    sym = {}
    for r in csv.DictReader(open(ROOT / "anchors" / "cdp_bks_symmetry2026_table1.csv")):
        # tabela: GKD-b_41_n150_b02 / SOM-a_11_n50_m5 -> nome do nosso catalogo sem o sufixo _m<k>
        key = r["instance"]
        sym[key if key.startswith("SOM-a") else key] = float(r["bks"])
    out = {}
    for r in csv.DictReader(open(ROOT / "anchors" / "cdp_literature_lu2023.csv")):
        nm = _our_name(r["name"], float(r["phi"]))
        # chave do Symmetry: GKD/MDG usam <inst>_n<n>_b02; SOM-a usa <inst>_n50_m<k> (sem b02)
        stem = re.sub(r"_m\d+$", "", nm)
        s = sym.get(stem)
        if s is None and r["name"].startswith("SOM-a") and abs(float(r["phi"]) - 0.2) < 1e-9:
            s = sym.get(r["name"])
        lu = float(r["lu_best"])
        out[nm] = dict(sym=s, lu_best=lu, sbts_avg=float(r["sbts_long_avg"]), sbts_short_avg=float(r["sbts_short_avg"]),
                       f_opt=float(r["f_opt"]) if r["f_opt"] != "" else None,
                       best=max([x for x in (lu, s, float(r["f_opt"]) if r["f_opt"] != "" else None) if x is not None]))
    ro = {r["name"]: r for r in csv.DictReader(open(ROOT / "anchors" / "cdp_mdgb_rosati2024_table5.csv"))}
    # Mladenovic et al. (2022, COR 139:105622), Tabelas 10-13 (parse_cor_tables.py): GVNS/GSVNS, as 100 instancias. Conferido em 8 out: nunca
    # supera o melhor valor ja acima (SBTS e posterior); entra em `best` por completude e transparencia.
    mlad = {}
    mf = ROOT / "anchors" / "cdp_literature_mladenovic2022.csv"
    if mf.exists():
        for r in csv.DictReader(open(mf)):
            mlad[_our_name(r["name"], float(r["phi"]))] = float(r["mlad_best"])
    for nm, v in out.items():
        if nm in mlad:
            v["mlad_cor_best"] = mlad[nm]
            v["best"] = max(v["best"], mlad[nm])
    for nm, v in out.items():
        v["sota_avg"] = v["sbts_avg"]
        r = ro.get(re.sub(r"_m\d+$", "", nm))
        if r is not None:
            v["best"] = max(v["best"], float(r["mlad_best"]), float(r["mnsa_best"]))
            v["sota_avg"] = max(v["sbts_avg"], float(r["mlad_avg"]), float(r["mnsa_avg"]))
    _CACHE = out
    return out


def reference(name: str):
    return _load().get(name)


def all_references() -> dict:
    return dict(_load())


if __name__ == "__main__":
    d = _load()
    print(len(d), "instancias com referencia")
    import collections
    print(collections.Counter(k.split("_")[0] + "_n" + k.split("_n")[1].split("_")[0] + "_" + k.split("_")[-2] for k in d))
    print("sem Symmetry (b03 + ...):", sum(v["sym"] is None for v in d.values()))
