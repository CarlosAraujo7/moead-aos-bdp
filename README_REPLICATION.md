# Replication guide

Code, run files and analysis scripts for *"Objective scaling, operator pools and adaptive selection in MOEA/D for bi-objective dispersion problems"* (working title). Draft status, 10 Oct 2026: nothing here is yet archived with a DOI.

## 1. What is in the package

| Path | Content |
|---|---|
| `src/` | Algorithms (`moead_aos.py`, `mab.py` [UCB1, Thompson, Gaussian sampler, faithful DYTS, FRRMAB], `operators.py`, `nsga2.py`, `smsemoa.py`, `grasp_mo.py`, `vns.py`, `bo_trajectory.py` [tabu search and VNS with the front as incumbent]), problems (`problem.py` = MDP, `capacitated_problem.py` = CDP/GDP), MaxMin anchoring (`anchor.py`, `sbts_anchor.py`), indicators and reference fronts (`metrics.py`, `reference_sets.py`), experiment configuration and budgets (`experiment.py`). |
| `instances/` | Instance catalogues (`catalog.csv` MDP, `catalog_capacitated.csv` CDP/GDP). The instance files themselves come from public sources (MDPLIB 2.0; the CDP benchmark of Peiró et al. 2021; the GDP benchmark of Lozano-Osorio et al. 2022, Zenodo 18233882) and are not redistributed here. |
| `results/capacitated_norm/{CDP,GDP}/runs`, `results_norm/runs` | Main experiment: 12 methods × all instances × 3 (n = 500, MDP) or 5 (CDP/GDP with n ≤ 150) replicates, normalised objectives. One JSON per (instance, method, replicate): seed, configuration hash, elapsed time, Pareto front, operator selection counts. |
| `results/capacitated/` | Earlier raw-objective catalogue (one replicate), used only for the raw-versus-normalised comparison. |
| `results/phase2_aos/` | Operator-selection variants (rewards, cost-aware, exploration floor, hyperparameter sweep, longer budgets, faithful DYTS, contextual selectors) and the MaxMin-anchored variants. |
| `results/e4_equal_time/`, `results/e5_budget_relaxed_norm/`, `results/e6_scalability/`, `results/e10_budget_regime/` | Equal-time baselines, GDP budget relaxation, scalability sample, budget-regime test. |
| `results/e11_strong_baselines/` | Tuned baselines (tuning logs and chosen configurations in `tuning/`, test runs in `<cell>/runs/`); kept out of the reference fronts. |
| `results/e8_operator_ablation/`, `results/e14_mechanism/` | Single-operator and leave-one-out ablations with replicates; per-operator counters of the tight-budget mechanism test. |
| `results/ablation_*` | Earlier (single-replicate) normalisation, single-operator and leave-one-out ablations; superseded by E8 for the operator ablations. |
| `results/reference_fronts/frozen_2026-10-06/` | **The reference fronts used in the paper** (see §3). |
| `results/analysis/` | JSON outputs of every analysis script. |
| `anchors/` | Values from the literature used to judge the MaxMin end of the fronts (CDP: Gomez et al. 2026 Table 1, Lu et al. 2023 Tables A.1–A.2, Rosati and Schaerf 2024 Table 5; GDP: `results.xlsx` of the Zenodo package). |
| `Claude outputs/` | Manuscript template, generated tables, figures. |

## 2. Environment

Python 3.12; `numpy`, `scipy`, `matplotlib`, `openpyxl`, `psutil` (versions in `requirements.txt`). No GPU. Experiments ran on a laptop (Intel i5-12450H, 8 cores/12 threads) with 4–8 worker processes; wall-clock times in the run files depend on the load at the time and are only comparable within one experiment.

## 3. Reproducing the results

```
python regression_check.py            # re-executes 70 stored runs from their seeds and compares the fronts bit for bit (about 2 min)
python regen_all.py                   # regenerates every statistic, table and figure from the stored runs (about 20 min)
```

`regen_all.py` **does not rebuild the reference fronts**: they are frozen (`results/reference_fronts/{CDP,GDP,MDP}.json`, copy in `frozen_2026-10-06/`). Rebuilding them (`python src/reference_sets.py build`) after adding runs changes every HV_norm value by construction; the paper's numbers use the frozen version.

Re-running experiments (all idempotent: a run whose file exists, with a matching configuration hash, is skipped):

| Experiment (paper section) | Command |
|---|---|
| Main comparison, normalised (6.2–6.3, 6.8) | `python run_full.py --problem CDP --reps 3 --workers 8 --normalize` (also `GDP`, `MDP`; sizes via `--min-n/--max-n`) |
| Selector variants, hyperparameters, budgets, contextual, anchoring (6.4, 6.6) | `python run_phase2.py --problem CDP --max-n 150 --reps 3 --variants <names>` (names: `run_phase2.py`) |
| Equal wall-clock baselines (6.7) | `python run_e4.py` |
| GDP budget relaxation (6.9) | `python run_e5.py` |
| Scalability sample (6.10) | `python run_e6.py` |
| Budget regime for adaptive selection (6.10) | `python run_e10.py` |
| Tuned baselines and Colmenar-style tabu search / VNS (6.12) | `python run_e11_tune.py --cells CDP-S,GDP-S,MDP-S,CDP-L,GDP-L,MDP-L` then `python run_e11.py` (both resumable; the tuning of each method and cell is in `results/e11_strong_baselines/tuning/`; `python test_bo_trajectory.py` checks the tabu/VNS implementation) |
| Size-growth probe (6.13) | `python run_e13.py` (after the tuning of the tabu search) and `python analyze_e13.py` |
| Operator ablation with replicates (6.5) | `python run_e8.py` then `python analyze_e8.py` |
| Tight-budget mechanism (6.11, Supplement S3) | `python run_e14.py` then `python analyze_e14.py` |
| FRRMAB, poisoned pool, GRASP initialisation (6.4, 6.5, Supplement S1) | `python run_phase2.py --problem CDP --max-n 150 --reps 3 --variants frr_default,...,px_random,...,init_a10,init_a00` (list in `run_queue6.py`), then `python analyze_e1_e2.py --problem CDP --part e15` (`e16`, `e17`) and `python analyze_aos_summary.py` |
| Size-aware operators and cardinality-bounded variant (6.14) | `python run_e18.py --exp free\|bound2\|bound15 [--cells CDP-S GDP-S CDP-L GDP-L]` (queue: `run_queue9.py`), then `python analyze_e18.py` |
| MOEA/D seeded by a trajectory search, MDP and constrained problems (6.15) | `python run_e20.py --cells MDP-S MDP-L CDP-S GDP-S --modes m_hyb_vns10,m_hyb_vns30,m_hyb_vns50` then `python analyze_e20.py` |
| Full scale on the MDP: 20 MDG-b n = 2,000 and a sample of MDG-c n = 3,000 (6.10, 6.16) | `python run_e10.py --parts E` then `python run_e21.py` and `python analyze_e21.py` (queue: `run_queue10.py`) |
| SPEA2 and Iterated Greedy with the irace settings of Colmenar et al. (6.12) | `python run_e22.py` then `python analyze_e20.py --dir e22_colmenar_rest --cells CDP-S,GDP-S,MDP-S,CDP-L,GDP-L,MDP-L` |
| SMS-EMOA in the tuned-baseline comparison (6.12) | `python run_e11_tune.py --methods sms` then `python run_e11.py` (queue: `run_queue7.py`) |

Analyses: `analyze_e11.py` (tuned-baseline tables), `analyze_e11_extremes.py` (coverage of the ends of the fronts), `analyze_replicated.py` (Tables 1–5, indicators), `analyze_phase2.py` / `analyze_e1_e2.py` (selector variants), `analyze_e4.py`, `analyze_e5.py`, `analyze_e6.py`, `analyze_e10.py`, `analyze_anchor_v2.py` and `analyze_anchor_lit.py` (anchoring and literature values), `analyze_by_family.py` (Supplement S2), `make_figures.py`, `make_manuscript_tables.py` (assembles the manuscript and the supplement from the templates and the JSON outputs; tables are numbered automatically by the order of their captions), `make_docx.py` (Word versions).

## 4. Conventions worth knowing

* Seeds: CRC32 of (instance, method, replicate) (`experiment.seed_for`); a run file stores its seed and configuration hash.
* The adaptive selector coded as `dyts_*` is a Gaussian sliding-window Thompson sampler (called GTS in the paper), **not** the DYTS of Sun and Li; the faithful DYTS is `operator_selection="dyts_beta"`.
* `normalize_objectives=True` divides each objective by its running range inside the Tchebycheff function; it is the regime of every result in the paper unless stated otherwise.
* Cost-aware selector variants depend on the clock and are not reproducible bit for bit by construction.
* Reference fronts are the non-dominated union of all valid runs of an instance; solutions of the relaxed-budget experiment violate the original GDP constraint and are kept out of the reference.
