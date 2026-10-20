"""Monta o pacote do artigo (rascunho) em entregas/artigo_EJOR_<data>.zip: PDFs, fonte LaTeX completa (compila sozinha), destaques, resumo grafico e um LEIA-ME
com o que esta pendente. Testa a compilacao a partir de uma copia limpa antes de zipar.

    python make_delivery_zip.py
"""
from __future__ import annotations

import datetime
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
LATEX = ROOT / "latex"
OUT = ROOT / "entregas"
stamp = datetime.date.today().isoformat()
pack = OUT / f"artigo_EJOR_{stamp}"
if pack.exists():
    shutil.rmtree(pack)
(pack / "fonte_latex" / "sections").mkdir(parents=True)
(pack / "fonte_latex" / "figures").mkdir(parents=True)

# --- fonte LaTeX (sem artefatos de compilacao)
for f in ("main.tex", "supplement.tex", "referencias.bib", "elsarticle.cls", "elsarticle-harv.bst", "algorithm.sty", "algorithmic.sty"):
    shutil.copy(LATEX / f, pack / "fonte_latex" / f)
for f in (LATEX / "sections").glob("*.tex"):
    shutil.copy(f, pack / "fonte_latex" / "sections" / f.name)
for f in (LATEX / "figures").glob("*"):
    shutil.copy(f, pack / "fonte_latex" / "figures" / f.name)

# --- teste de compilacao a partir da copia
for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], ["bibtex", "main"], ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "supplement"],
            ["bibtex", "supplement"], ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "supplement"], ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "supplement"]):
    r = subprocess.run(cmd, cwd=str(pack / "fonte_latex"), capture_output=True, text=True, errors="replace")
    if r.returncode != 0 and cmd[0] == "pdflatex":
        print("FALHA:", " ".join(cmd), "\n".join(r.stdout.splitlines()[-20:]))
        sys.exit(1)
shutil.copy(pack / "fonte_latex" / "main.pdf", pack / "artigo_EJOR_rascunho.pdf")
shutil.copy(pack / "fonte_latex" / "supplement.pdf", pack / "material_suplementar_rascunho.pdf")
for f in list((pack / "fonte_latex").glob("*.aux")) + list((pack / "fonte_latex").glob("*.log")) + list((pack / "fonte_latex").glob("*.out")) + \
        list((pack / "fonte_latex").glob("*.blg")) + list((pack / "fonte_latex").glob("*.bbl")) + list((pack / "fonte_latex").glob("*.spl")) + \
        [pack / "fonte_latex" / "main.pdf", pack / "fonte_latex" / "supplement.pdf"]:
    f.unlink(missing_ok=True)

# --- destaques e resumo grafico
fig = ROOT / "Claude outputs" / "figures"
for src, dst in ((ROOT / "Claude outputs" / "EJOR_highlights.docx", "EJOR_highlights.docx"), (fig / "graphical_abstract.pdf", "graphical_abstract.pdf"),
                 (fig / "graphical_abstract.png", "graphical_abstract.png")):
    shutil.copy(src, pack / dst)

(pack / "LEIA-ME.txt").write_text(f"""Artigo (rascunho) - {stamp}

Titulo: Objective scaling, operator pools and adaptive selection in MOEA/D for bi-objective dispersion problems
Alvo: European Journal of Operational Research (template elsarticle fornecido pelos autores)

CONTEUDO
  artigo_EJOR_rascunho.pdf            artigo completo ate agora (pendencias em VERMELHO no texto e na caixa da pagina 3)
  material_suplementar_rascunho.pdf   suplemento (S1 a S3)
  fonte_latex/                        fonte LaTeX completa (main.tex, supplement.tex, sections/, referencias.bib, figures/, classe e estilos do template)
  EJOR_highlights.docx                destaques (5 itens, no maximo 85 caracteres cada)
  graphical_abstract.pdf/.png         resumo grafico

COMO COMPILAR (testado a partir desta pasta)
  cd fonte_latex
  pdflatex main ; bibtex main ; pdflatex main ; pdflatex main        (idem para supplement)
  (o pacote cuted do template foi comentado porque nao e usado; no Overleaf pode ser reativado)

O QUE ESTA PENDENTE (tudo aparece em vermelho no PDF)
  1. Secao 6.16 (escala completa no MDP: 20 MDG-b com n = 2.000 e 8 MDG-c com n = 3.000; baselines ajustados e MOEA/D semeado, a tempo igual): execucao iniciada
     (59 de 168 arquivos), NAO analisada. Os resultados nao entram em nenhuma tabela ainda.
  2. MOEA/D semeado pelo VNS (Secao 6.15) ainda nao foi rodado no CDP e no GDP com n = 500 (so n <= 150 e MDP).
  3. Depois de rodar o que falta: regen_all.py (regenera tabelas e figuras) e regression_check.py.
  4. Repetir a busca bibliografica no Scopus/Web of Science antes de submeter (a busca feita foi por OpenAlex/Crossref).
  5. DOI do repositorio (Zenodo) na declaracao de dados e no Apendice B.
  6. Os autores precisam conferir: declaracao de uso de IA generativa, papeis CRediT (copiados do template do artigo de corte a laser), conflito de interesses,
     financiamento e agradecimentos.
  7. Revisao de ingles; cortes ate o tamanho desejado (decidir se as Secoes 6.4 e 6.6 e o Apendice C vao para o suplemento online).
  8. Formulario de conflito de interesses do EJOR e carta ao editor.

Observacao: os numeros do texto vem das tabelas geradas a partir dos arquivos de resultados do repositorio (https://github.com/CarlosAraujo7/moead-aos-bdp, ainda sem push).
""", encoding="utf-8")

# --- zip
zpath = OUT / f"artigo_EJOR_rascunho_{stamp}.zip"
if zpath.exists():
    zpath.unlink()
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(pack.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(OUT))
print(zpath, f"{zpath.stat().st_size / 1e6:.1f} MB")
for n in zipfile.ZipFile(zpath).namelist():
    if "/sections/" not in n and "/figures/" not in n:
        print("  ", n)
