# Pendências — atualizado em 12 out 2026

**Situação em uma frase:** o manuscrito está migrado para LaTeX no template do EJOR (`latex/main.pdf`, gerado por `python make_latex.py --build`), o git tem 6 commits locais (sem push) e três frentes novas de melhoria estão rodando (tamanho do conjunto, MDP híbrido, escala completa); o que resta depende de **você** (decisões e artigos) ou das **fases longas de experimentos** (~15 h de máquina, ver §4).

---

## 1. Decisões e autorizações que preciso de você

1. **Push para https://github.com/CarlosAraujo7/moead-aos-bdp.git** — os commits estão prontos localmente (`git log --oneline`). Posso fazer o push? Antes: as **notas internas em português** (`CLAUDE.md`, `ANDAMENTO.md`, `PLANO_PUBLICACAO.md`, `PENDENCIAS_*`, `REFERENCIAS_A_CONSEGUIR.md`) estão num commit separado (`414cf4b`); se não quiser publicá-las, eu as retiro do histórico antes do push.
2. **CRediT**: copiei os papéis do template (artigo de corte a laser) para os cinco autores; confirme ou ajuste para este trabalho (`latex/main_template.tex`, seção "CRediT").
3. **Declaração de IA generativa** (obrigatória no EJOR): o texto diz que o Claude foi usado para escrever e depurar código, rodar e organizar experimentos e redigir texto/tabelas, e que os autores revisaram e assumem a responsabilidade. Confirme ou edite.
4. **Título** (hoje "Objective scaling, operator pools and adaptive selection in MOEA/D for bi-objective dispersion problems"): com os resultados novos (tamanho do conjunto, híbrido) talvez valha rever com orientador/coautores.
5. **Plano B de periódico/evento**: ainda não escolhido. Lembrete: o EJOR não considera versões estendidas de artigos de conferência; enviar estes resultados ao CEC antes impediria o EJOR.

## 2. Artigos que preciso que você consiga (títulos exatos em `REFERENCIAS_NOVAS_A_CONSEGUIR.md`)

O Claude in Chrome não estava conectado, então **não consultei Scopus/WoS** (exigem sua sessão); usei OpenAlex/Crossref (sem texto completo). Faltam:
- *GRASP and tabu search for the generalized dispersion problem* (Martínez-Gavara, Corberán & Martí, ESWA 173:114703, 2021) — original do GDP;
- *Multi-objective Optimization* (cap. 14 de *Discrete Diversity and Dispersion Maximization*, Colmenar, Herrán & Martín-Santamaría, Springer 2023);
- (opcional) *Diversity and Equity Models* / *Diversity Optimization Models* (Sandoya et al.).
Há também ~15 trabalhos novos de 2023–2026 listados no arquivo (CDP, MDP com capacidade e orçamento, AOS em MOEA/D); os de prioridade alta são Wang et al. (2024, IEEE TETCI) e Radanović et al. (2025, ESWA). **Repetir as 4 consultas do arquivo no Scopus/WoS perto da submissão** (o manuscrito e a carta têm `[TODO]` para isso).

## 3. O que é seu ao final

- Repositório com DOI: criar o depósito no Zenodo (código + `results/`, conforme `DATA_PACKAGE_MANIFEST.md`, ~84 MB, ~100 mil arquivos) e colocar o DOI no `\todo` da seção "Data and Code Availability" e na carta.
- Revisão de inglês e leitura crítica por orientador/coautores; **cortes para ~30 páginas** (você disse que faz); conferir visualmente o PDF (tabelas largas estão reduzidas com `\resizebox`).
- Declaração de conflito de interesses (arquivo Word do sistema do EJOR) e carta ao editor (`Claude outputs/submission_materials_draft.md`).

## 4. Experimentos em andamento (a máquina precisa ficar ligada)

| Fila | O que | Estado em 12 out ~03h |
|---|---|---|
| `run_queue9.py` (aba do terminal) | **E18**: operadores de tamanho e variante com limite de cardinalidade (ρ=2; ρ=1,5 só n≤150), células n≤150 e n=500, 3 réplicas, tempo igual | n≤150 livre e ρ=2 concluídos; n=500 em execução (~10 h) |
| `run_queue10.py` (processo destacado, espera a fila 9) | **E20** limpo (MOEA/D semeado por VNS 10/30/50%; MDP-S/L, CDP-S, GDP-S), **E10 parte E** (restante das MDG-b n=2000), **E21** (escala completa: MDG-b n=2000 e amostra de MDG-c n=3000; baselines ajustados e híbrido a tempo igual) | aguardando |

Se a máquina reiniciar: `python run_queue9.py 8` e depois `python run_queue10.py 8` retomam do ponto (idempotentes). Não importe `run_queue*.py`: executam ao importar.

## 5. Resultados novos já conhecidos (ainda não no texto)

- **Tamanho do conjunto**: no regime atual o arquivo do MOEA/D já tem 2–3× o tamanho da solução gulosa mínima no CDP (deriva do reparo). Com os operadores de tamanho (`biobj_grow`, `bottleneck_shrink`) o MOEA/D domina Tabu/VNS (mesmo com inserções) no problema livre (hv_norm 0,853 vs 0,287 no CDP n≤150) e, **no problema com limite de cardinalidade**, a vantagem some no CDP (VNS com inserções empata: ranks 3,3 vs 3,8) e permanece no GDP (+0,20).
- **MDP híbrido**: semear o MOEA/D com VNS (50% de T) sobe 0,831 → 0,939 no MDP (VNS ajustado: 0,977), mas **piora** CDP (−0,05) e GDP (−0,075): não há híbrido universal (rodadas sob carga; refeitas limpas na fila 10).
- Detalhes: `ANDAMENTO.md` e `CLAUDE.md` §2u.
