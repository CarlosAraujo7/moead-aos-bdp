# Pendências — atualizado em 10 out 2026

**Situação em uma frase:** o manuscrito (`Claude outputs/manuscript_draft_v2.docx` e `.md`) e o suplemento (`supplementary_material.docx`) estão estruturados para o EJOR e com texto genérico nas partes de autoria; em 10 out achei e corrigi um artefato de ponto flutuante que afetava as recompensas de "sucesso" (seletores Thompson/DYTS e recompensa `insert`) — a reexecução das variantes afetadas está rodando (ver `ANDAMENTO.md`) e, quando terminar, preciso rodar `regen_all.py` e conferir todos os números de prosa. O que resta depende de **você** (artigos que faltam, DOI, declarações, decisões) ou de **sua autorização** (limpeza e commit).

---

## 1. Artigos que ainda faltam

Os 11 artigos pedidos em 8 out chegaram e foram lidos (Colmenar et al. 2018 na versão publicada, Mladenović et al. 2022 COR, Geng et al. 2023, Fialho et al. 2010, Li et al. 2014, Duarte et al. 2015, Knowles et al. 2006, Parreño et al. 2021, irace, Sun & Li 2020, Gomez et al.). Os metadados de todas as referências foram conferidos no Crossref. Ainda não tenho:

| # | Título | Por que importa | Prioridade |
|---|---|---|---|
| 1 | **GRASP and tabu search for the generalized dispersion problem** — Martínez-Gavara, Corberán & Martí (2021), *Expert Systems with Applications* 173:114703 | É o artigo **original do GDP**; o texto descreve o GDP a partir do artigo de Lozano-Osorio et al. (2022) e do pacote Zenodo. | Média |
| 2 | **Multi-objective Optimization in Diversity Problems** (cap. 14 de *Discrete Diversity and Dispersion Maximization*, Springer, 2023) — Colmenar, Herrán & Martín-Santamaría | Pode revelar trabalhos multiobjetivo de diversidade que não achei (§2). Exige acesso institucional. | Média |
| 3 | **Diversity and Equity Models** — Sandoya et al. (2018), *Handbook of Heuristics* | Só citado indiretamente; opcional. | Baixa |

## 2. Decisões que são suas

1. **CEC 2027 como plano B:** o guia do EJOR **não considera versões estendidas de artigos de conferência**; enviar estes resultados ao CEC bloquearia o EJOR. Preprint não conta como publicação prévia.
2. **Declaração de uso de IA generativa (obrigatória):** rascunho no manuscrito (Claude usado para código, experimentos e redação; autores revisaram e assumem a responsabilidade). **Confirmar ou editar.**
3. **Seções genéricas que você disse que vai editar:** CRediT, conflito de interesses, financiamento, agradecimentos (já estão preenchidas com texto padrão, marcadas `[generic text]` no manuscrito), carta ao editor (`submission_materials_draft.md`).
4. **Formulação do CDP/GDP:** o MaxSum cresce com o tamanho do conjunto (§3, §6.13). Uma variante com limite de cardinalidade seria **outro problema**: deixar como trabalho futuro (padrão atual) ou fazer agora?
5. **Suplemento:** movi para `supplementary_material.docx` os detalhes de §6.4 (S1), os ranks por família (S2) e o mecanismo do orçamento apertado (S3). Se preferir tudo no corpo, é só reverter (o gerador aceita).
6. **Título:** hoje "Objective scaling, operator pools and adaptive selection in MOEA/D for bi-objective dispersion problems"; validar com orientador/coautores.

## 3. Checklist de submissão ao EJOR

- [x] Resumo ≤ 250 palavras (conferir de novo após a atualização final), sem citações.
- [x] 1–5 palavras-chave, a primeira da lista oficial.
- [x] Highlights (`EJOR_highlights.docx`, 5 itens ≤ 85 caracteres).
- [x] Resumo gráfico: `Claude outputs/figures/graphical_abstract.pdf|png` (4 painéis; conferir a legibilidade no tamanho final).
- [x] Fonte Word: `manuscript_draft_v2.docx` (gerado por `make_docx.py`; **revisar visualmente no Word**: tabelas largas, equações em texto simples, numeração).
- [ ] **DOI do repositório** (Zenodo) e frase de disponibilidade de dados; `[TODO]` em §5, Apêndice B e na declaração de dados. Manifesto pronto: `DATA_PACKAGE_MANIFEST.md`.
- [ ] Repetir a busca bibliográfica no Scopus/Web of Science perto da submissão (a minha foi só pela web, 5 out) — `[TODO]` no §2 e na carta.
- [ ] Declaração de conflitos de interesse (arquivo Word gerado pelo sistema) e CRediT preenchidos pelos autores.
- [ ] Revisão de inglês e leitura crítica por orientador/coautor.

## 4. Pendências técnicas (dependem de sua autorização)

- **Commit:** nenhum commit foi feito (instrução de 1 out). Há dezenas de arquivos novos/modificados; indispensável antes de submeter.
- **Limpeza:** movi (sem apagar) os arquivos de depuração para `_obsolete/` e os JSONs obsoletos para `results/analysis/_stale/`; `.gitignore` atualizado (`a pedidos/`, `_obsolete/`, `_to_delete/`). Nada foi apagado.
- **Pacote de dados com DOI:** ver `DATA_PACKAGE_MANIFEST.md` (cerca de 84 MB, ~100 mil arquivos).
- **Congelamento:** a referência está congelada desde 6 out (`results/reference_fronts/frozen_2026-10-06/`); não reconstruir.
- **Código opt-in sem uso no artigo** (`polish_frac` em `moead_aos.py`): posso remover para o pacote público.
- **Catálogo histórico bruto** (`results/capacitated`): contém o artefato de ponto flutuante nas duas variantes Thompson; não foi refeito (só entra no §6.1 e no §6.6 como "melhor dos 12 métodos"); está declarado no texto.

## 5. Estado dos pontos fracos prováveis

Ver `REVIEWER_RISKS.md` (tabela completa: objeção, resposta no manuscrito, risco residual).
