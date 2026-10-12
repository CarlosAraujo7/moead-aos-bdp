# Data package manifest (draft, 10 Oct 2026)

Plan for the public deposit (Zenodo or equivalent) that EJOR's "Option B" data policy encourages. The DOI has to be created by the authors; nothing here has been uploaded. Sizes are those of 10 Oct 2026 (the files are small JSON/CSV; the run files and analysis outputs total about 84 MB (about 100,000 files)).

## What goes in

| Path | Content | Files | Role in the paper |
|---|---|---|---|
| `src/`, `*.py` (root), `requirements.txt`, `README_REPLICATION.md` | code: algorithms, runners, analysis, tables, figures | ~110 | all |
| `instances/catalog.csv`, `instances/catalog_capacitated.csv`, `instances/instance_format_*.txt` | instance catalogues (names, families, sizes, paths) | 5 | all |
| `results/capacitated/{CDP,GDP}` | historical (raw-objective) catalogue, 1 replicate | 4,798 | §6.1 |
| `results/capacitated_norm/{CDP,GDP}`, `results_norm/` (MDP) | main experiment: 12 methods, normalised regime, 3–5 replicates | 15,122 + 5,149 | Tables 1–4, 6, Figs. 1–3, 5 |
| `results/phase2_aos/{CDP,GDP}` | adaptive-selection variants, anchors, lean pools, FRRMAB, poisoned pool, initialisation | 36,480 | §6.4–6.6, S1 |
| `results/e4_equal_time`, `e5_budget_relaxed_norm`, `e6_scalability`, `e10_budget_regime` | equal-time baselines, relaxed GDP budget, large instances, budget regimes | 4,320; 5,040; 91; 2,277 | §6.7, 6.9–6.11 |
| `results/e8_operator_ablation` | single-operator and leave-one-out ablations (3 replicates, CDP and GDP) | 8,640 | §6.5, Table 6 |
| `results/e11_strong_baselines`, `e13_size_growth` | tuning, tuned baselines, tabu/VNS of Colmenar et al., size experiment | 14,092; 48 | §6.12–6.13 |
| `results/e18_size`, `e20_hybrid`, `e21_xl` | size-aware operators and cardinality-bounded variant (E18), MOEA/D seeded by trajectory search (E20), full-scale MDP runs (E21) | about 25,000 | §6.14–6.16 |
| `results/e14_mechanism` | per-operator counters (applications, infeasible children, insertions) | 540 | §6.11 |
| `results/reference_fronts/frozen_2026-10-06/{CDP,GDP,MDP}.json` | **frozen** reference fronts used by every indicator | 3 | all (do not rebuild) |
| `results/analysis/*.json`, `results/analysis/regen_log.txt` | outputs of the analysis scripts (tables and figures are generated from these) | ~70 | all |
| `anchors/*.csv`, `anchors/*.json` | published best values transcribed from the literature (Lu et al. 2023; Gomez et al. 2026; Mladenović et al. 2022b; Rosati and Schaerf 2024) and the anchor points of the pilot | 8 | §6.6 |
| `Claude outputs/manuscript_template.md`, `supplement_template.md`, `figures/` | sources of the manuscript, supplement and figures | — | — |

Every run file stores instance, method, replicate, seed, elapsed time, Pareto front and (for the newer ones) the configuration and its hash; `regression_check.py` re-executes 66 of them from the stored seeds and checks bit-for-bit equality.

## What stays out

- `a pedidos/` (about 3 GB): third-party PDFs and the Zenodo 18233882 archive of Lozano-Osorio et al. — cite, do not redistribute (the GDP instances are CC-BY 4.0 and are identical, by MD5, to those in `instances/GDP (Const.)`; the repository should link the record instead of copying it).
- `instances/` (5.2 GB): MDPLIB 2.0 and derived instance files; keep the catalogues and tell users where to fetch each family (MDPLIB 2.0 page of Martí et al.; GDP: Zenodo 18233882; CDP: the Peiró et al. testbed). The CDP/GDP files (184 MB) could be included if the licences allow.
- `results/runs/` (obsolete catalogue from before the operator fix), `results/prelim_sense_check`, `results/integration_test`, `results/smoke_new`, `results/ablation_*` of 1 Oct 2026 (single replicate; superseded by E8 and by `phase2_aos`), `results/analysis/_stale/`, `_obsolete/`, `_to_delete/`.

## Before the deposit

1. `python regression_check.py` (70/70) and `python regen_all.py` (log in `results/analysis/regen_log.txt`).
2. Decide the licence (suggestion: MIT for code, CC-BY 4.0 for data and results).
3. Pin the package versions in `requirements.txt` (`pip freeze` of the environment used) and record the Python version (3.12).
4. Add the DOI to the manuscript (`[TODO: add the repository DOI]` in §5, Appendix B and the data availability statement) and to the cover letter.
5. Do not commit or push anything that the authors have not reviewed (the working tree has no commit of the new work; this was requested).
