"""Extrai as Tabelas 10-13 de Mladenovic, Todosijevic, Urosevic & Ratli (2022), Computers & Operations Research 139:105622, para
anchors/cdp_literature_mladenovic2022.csv: valores por instancia (100 instancias, B = 20% e 30% da capacidade) de SO, SS1, SS2 (heuristicas
anteriores), GVNS e GSVNS (melhor valor de 10 execucoes; media e pior tambem).

O texto extraido do PDF (pypdf) fica em ENTRADA; a conferencia de consistencia compara as colunas SO/SS1/SS2 com as de Lu et al. (2023)
(anchors/cdp_literature_lu2023.csv): se baterem em todas as instancias, a tabela foi lida corretamente.

    python parse_cor_tables.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pypdf

ROOT = Path(__file__).parent
PDF = ROOT / "a pedidos" / "novos pedidos" / "1-s2.0-S0305054821003361-main.pdf"
OUT = ROOT / "anchors" / "cdp_literature_mladenovic2022.csv"
FAMS = ("GKD-a", "GKD-b", "GKD-c", "SOM-a", "MDG-b")
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def main():
    r = pypdf.PdfReader(str(PDF))
    text = "\n".join((p.extract_text() or "") for p in r.pages)
    lines = text.split("\n")
    # trechos das Tabelas 10-13 (comparacao completa), com o B de cada uma
    heads = [i for i, ln in enumerate(lines) if re.match(r"^Comparison of SO, SS1, SS2, GVNS and GSVNS methods on instances with", ln.strip())]
    ends = [i for i, ln in enumerate(lines) if ln.strip().startswith("Wilcoxon signed-rank test")]
    assert len(heads) == 4 and ends, (heads, ends)
    rows = []
    for h, nxt in zip(heads, heads[1:] + [ends[0]]):
        beta = 0.2 if "20%" in lines[h] else 0.3
        fam, n = None, None
        for ln in lines[h + 1: nxt]:
            tk = ln.replace("−", "-").split()
            if not tk or tk[0] in ("Inst.", "name", "Avg."):
                continue
            if tk[0] in FAMS:
                fam, n, inst, vals = tk[0], int(tk[1]), int(tk[2]), tk[3:]
            elif NUM.match(tk[0]) and len(tk) >= 16 and fam is not None:
                n, inst, vals = int(tk[0]), int(tk[1]), tk[2:]
            else:
                continue
            vals = [float(x) for x in vals if NUM.match(x)]
            if len(vals) < 14:
                continue
            rows.append(dict(fam=fam, n=n, inst=inst, beta=beta, so=vals[0], ss1=vals[2], ss2=vals[4], gvns_best=vals[6], gvns_avg=vals[7],
                             gsvns_best=vals[10], gsvns_avg=vals[11], gsvns_worst=vals[12], gsvns_time=vals[13]))
    print(len(rows), "linhas de instancia lidas")
    lu = {}
    for x in csv.DictReader(open(ROOT / "anchors" / "cdp_literature_lu2023.csv")):
        m = re.match(r"^(.*)_(\d+)_n(\d+)_m\d+$", x["name"])
        lu[(m.group(1), int(m.group(2)), int(m.group(3)), round(float(x["phi"]), 1))] = x
    out, bad = [], 0
    for r in rows:
        fam = r["fam"]
        cand = [k for k in lu if k[1] == r["inst"] and k[2] == r["n"] and k[3] == r["beta"] and (k[0] == fam or (fam == "GKD-a" and k[0] == "GKD-b"))]
        if len(cand) != 1:
            print("sem correspondencia unica:", r["fam"], r["n"], r["inst"], r["beta"], cand)
            bad += 1
            continue
        x = lu[cand[0]]
        same = all(abs(float(x[c]) - r[c.replace("ss1", "ss1")]) < 1e-6 for c in ("so", "ss1", "ss2"))
        if not same:
            print("DIFERE de Lu:", cand[0], [(c, x[c], r[c]) for c in ("so", "ss1", "ss2")])
            bad += 1
        out.append(dict(name=x["name"], phi=cand[0][3], so=r["so"], ss1=r["ss1"], ss2=r["ss2"], gvns_best=r["gvns_best"], gvns_avg=r["gvns_avg"],
                        gsvns_best=r["gsvns_best"], gsvns_avg=r["gsvns_avg"], gsvns_worst=r["gsvns_worst"], gsvns_time=r["gsvns_time"],
                        mlad_best=max(r["gvns_best"], r["gsvns_best"])))
    names = {(o["name"], o["phi"]) for o in out}
    print(len(out), "instancias unicas:", len(names), "| problemas:", bad)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(sorted(out, key=lambda o: (o["phi"], o["name"])))
    print("salvo:", OUT)


if __name__ == "__main__":
    main()
