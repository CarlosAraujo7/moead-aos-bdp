# Supplementary material

*Supplement to "Objective scaling, operator pools and adaptive selection in MOEA/D for bi-objective dispersion problems". Assembled by `make_manuscript_tables.py` from `supplement_template.md`; every table is generated from `results/analysis/*.json`. References are those of the main article.*

## S1. Attempts to repair adaptive selection (details of Section 6.4)

All variants run on the 60 CDP and 120 GDP instances with n ≤ 150 for three replicates, with the seeds of the main experiment and normalised objectives; the reference of each comparison is uniform selection on the same instances (for the budget family, uniform selection with the same enlarged budget). Table S1 gives the result of every variant; the paragraphs below describe the designs and the points that do not fit in a table. Success criteria use the tolerance of Section 4.3.

**Table S1. Every variant against uniform selection on the same instances (n ≤ 150, three replicates): difference in HV_norm, Vargha–Delaney Â12 and Holm-adjusted Wilcoxon p within the family; an asterisk marks p < 0.05 (all such cases are losses)**

| Family | Variant | CDP (60 instances) | GDP (120 instances) |
|---|---|---|---|
| Rewards, cost, floor | UCB1, insert reward | −0.0022, 0.51, 1 | −0.0072, 0.49, 0.502 |
|  | Thompson, insert reward | +0.0031, 0.50, 1 | −0.0144, 0.49, 0.262 |
|  | GTS, insert reward | −0.0052, 0.50, 1 | −0.0166, 0.49, 0.502 |
|  | UCB1, insert reward, cost-aware | −0.0057, 0.48, 1 | −0.0113, 0.48, 0.502 |
|  | UCB1, relative scalar reward, cost-aware | −0.0244, 0.40, 0.0118* | – |
|  | UCB1, insert reward, 50% exploration floor | +0.0034, 0.51, 1 | −0.0098, 0.49, 0.502 |
|  | Thompson, insert reward, 50% exploration floor | −0.0129, 0.47, 1 | −0.0144, 0.49, 0.456 |
|  | UCB1, relative scalar reward, 50% exploration floor | −0.0017, 0.48, 1 | – |
| DYTS and replace | DYTS (Sun and Li), C = 100 | −0.0110, 0.45, 0.642 | −0.0177, 0.48, 0.0177* |
|  | DYTS, C = 30 | −0.0113, 0.46, 0.926 | −0.0174, 0.48, 0.0908 |
|  | DYTS, C = 300 | −0.0032, 0.50, 1 | −0.0146, 0.46, 0.0139* |
|  | DYTS posterior, binary scalar reward | −0.0307, 0.38, 0.00295* | −0.0685, 0.32, 1.7e−10* |
|  | DYTS posterior, binary HV reward | −0.0061, 0.48, 1 | −0.0106, 0.48, 0.17 |
|  | Thompson, replace reward | −0.0027, 0.48, 1 | −0.0238, 0.46, 0.0139* |
|  | UCB1, replace reward | −0.0024, 0.49, 1 | −0.0108, 0.48, 0.472 |
| Contextual | UCB1 insert, 3 weight contexts | −0.0044, 0.50, 1 | −0.0087, 0.49, 0.87 |
|  | UCB1 insert, 3 phases | −0.0013, 0.50, 1 | −0.0090, 0.48, 0.87 |
|  | UCB1 insert, 9 contexts | −0.0051, 0.52, 1 | −0.0125, 0.49, 0.549 |
|  | Thompson insert, 9 contexts | −0.0009, 0.52, 1 | −0.0137, 0.48, 0.319 |
|  | DYTS replace, 9 contexts | −0.0017, 0.50, 1 | −0.0145, 0.49, 0.549 |
|  | UCB1 relative scalar, 9 contexts | −0.0120, 0.45, 0.906 | −0.0148, 0.46, 0.224 |
|  | UCB1 HV, 9 contexts | −0.0068, 0.48, 1 | −0.0118, 0.50, 0.87 |
| Hyper-parameters | UCB1-HV, c = 0.05 | −0.0135, 0.48, 1 | −0.0104, 0.50, 1 |
|  | UCB1-HV, c = 2 | +0.0058, 0.52, 1 | −0.0061, 0.50, 1 |
|  | UCB1-HV, window 20 | +0.0055, 0.51, 1 | −0.0151, 0.49, 0.51 |
|  | Thompson-HV, window 20 | −0.0057, 0.49, 1 | −0.0135, 0.49, 0.657 |
|  | Thompson-Scal, window 20 | −0.0081, 0.46, 1 | −0.0208, 0.44, 0.0083* |
|  | GTS-HV, window 20 | +0.0015, 0.51, 1 | −0.0058, 0.49, 1 |
|  | GTS-Scal, window 20 | −0.0004, 0.52, 1 | −0.0044, 0.49, 1 |
|  | UCB1-HV, window 200 | +0.0019, 0.50, 1 | +0.0016, 0.52, 1 |
|  | Thompson-HV, window 200 | +0.0035, 0.51, 1 | −0.0166, 0.48, 0.317 |
|  | Thompson-Scal, window 200 | −0.0378, 0.33, 0.00011* | −0.0692, 0.32, 9.62e−11* |
|  | GTS-HV, window 200 | −0.0167, 0.47, 0.529 | −0.0049, 0.51, 1 |
|  | GTS-Scal, window 200 | −0.0017, 0.49, 1 | −0.0066, 0.49, 1 |
|  | UCB1 relative scalar, c = 0.05 | −0.0365, 0.34, 3.62e−05* | −0.0835, 0.29, 7.55e−13* |
|  | UCB1 relative scalar, c = 0.5 | −0.0392, 0.36, 0.00109* | −0.0829, 0.30, 5.36e−11* |
|  | UCB1 relative scalar, c = 2 | −0.0259, 0.39, 0.0029* | −0.0856, 0.30, 1.26e−12* |
|  | UCB1 insert, c = 0.05 | −0.0013, 0.49, 1 | −0.0021, 0.49, 1 |
|  | UCB1 insert, c = 2 | +0.0041, 0.52, 1 | −0.0120, 0.48, 0.317 |
| FRRMAB | FRRMAB (W = 0.5N, C = 5, D = 1) | +0.0063, 0.52, 0.95 | −0.0081, 0.48, 0.402 |
|  | FRRMAB, W = 2.5N | −0.0102, 0.49, 0.95 | −0.0071, 0.50, 0.707 |
|  | FRRMAB, C = 1 | −0.0112, 0.45, 0.551 | −0.0167, 0.46, 0.0483* |
|  | FRRMAB, C = 0.3 | −0.0045, 0.49, 0.988 | −0.0115, 0.49, 0.427 |
|  | FRRMAB, W = 2.5N, C = 0.3 | −0.0086, 0.47, 0.551 | −0.0093, 0.49, 0.707 |
| More generations | GTS-HV, ×2 generations | −0.0080, 0.46, 0.189 | +0.0001, 0.50, 0.77 |
|  | UCB1-HV, ×2 generations | −0.0026, 0.46, 0.522 | +0.0033, 0.52, 0.552 |
|  | GTS-HV, ×4 generations | −0.0020, 0.48, 1 | +0.0033, 0.51, 0.353 |
|  | UCB1-HV, ×4 generations | +0.0044, 0.52, 1 | −0.0020, 0.50, 0.561 |

*Other rewards, cost-aware scaling and exploration floor.* Four designs, with UCB1, Thompson sampling and the Gaussian sampler (GTS) where applicable: (a) the *insert* reward, 1 if the child enters the non-dominated archive and 0 otherwise, which does not saturate as the hypervolume gain does; (b) the *relative scalar* reward, the Tchebycheff improvement divided by the parent's value, in [0, 1] (CDP and GDP for UCB1; the two variants with it were run on the CDP only); (c) each combined with *cost-aware scaling*, the reward divided by the operator's relative running time; (d) an *exploration floor* mixing the learned policy 50/50 with uniform selection. None beats uniform selection. On the CDP the only significant difference is a loss, for UCB1 with the relative scalar reward and cost-aware scaling (−0.024, Â12 0.40, p_Holm = 0.012). With the insert reward the selectors spread the selections moderately (CDP, mean share per operator 0.10–0.18 under UCB1, Thompson sampling and GTS; BiObjBestSwap and MaxMinSwap the most chosen), as with the hypervolume reward, because the reward is also zero for most children once the front is crowded. Cost-aware scaling did not make the search faster: with it, the median run time was 2.70 s against 2.65 s without it on the CDP and 7.45 s against 7.71 s on the GDP (the medians are dominated by the instances with n = 50), because of the overhead of the selector and of the archive check. Two independent runs of uniform selection on the CDP differ by +0.006 in HV_norm (Â12 0.52, p = 0.23), which calibrates the noise floor.

*Faithful dynamic Thompson sampling.* The DYTS of Sun and Li (2020) as published: Beta posteriors initialised at (1, 1), decay threshold C = 100 (we also ran 30 and 300), a binary reward equal to 1 when the child strictly improves the Tchebycheff value of at least one neighbouring subproblem (the *replace* reward, with the tolerance of Section 4.3), and no warm-up; for comparison, the same reward with UCB1 and with our Thompson sampler, and the DYTS posterior with the binary scalar and HV rewards. On the CDP no variant beats uniform selection and only the binary scalar reward is significantly worse (−0.031, Â12 0.38, p_Holm = 0.003). On the GDP four variants are significantly worse: the published DYTS (C = 100: −0.018, p = 0.018; C = 300: −0.015, p = 0.014), Thompson sampling with the same reward (−0.024, p = 0.014) and the binary scalar reward (−0.068, p < 0.0001). With the published reward the selector stays moderately spread over the operators (CDP: 0.064–0.257 per operator, with BiObjBestSwap 0.257 and MaxMinSwap 0.192), so that its equality with uniform selection is again explained by near-uniform behaviour rather than by a badly tuned bandit.

*Contextual selection.* A separate bandit per context, so that the best operator may differ between regions of the weight simplex and phases of the search: contexts defined by the third of the weight-vector range the subproblem belongs to (three contexts), by the third of the run (three) or by both (nine), with UCB1 and the insert reward; and the nine-context version with Thompson sampling and the insert reward, with the published DYTS and the replace reward, with UCB1 and the relative scalar reward, and with UCB1 and the HV reward (seven variants; no information is shared between contexts). None differs significantly from uniform selection on either problem (CDP −0.012 to −0.001; GDP −0.015 to −0.009; p_Holm ≥ 0.9 on the CDP and ≥ 0.22 on the GDP). The sign of the effect of the context is not consistent across selectors.

*Hyper-parameters.* The exploration constant (c ∈ {0.05, 0.5, 2}) and the sliding-window length (20, 50, 200) of the HV-reward UCB1, the window of Thompson sampling and of the Gaussian sampler under both historical rewards, and c for UCB1 with the relative scalar and the insert rewards (17 variants). None beats uniform selection (largest differences +0.006 on the CDP and +0.002 on the GDP; Â12 ≤ 0.52). Four variants are significantly worse on the CDP and five on the GDP, always those with a scalar or relative-scalar reward (UCB1 with the relative scalar reward at the three values of c; Thompson sampling with the scalar reward and a window of 200, and on the GDP also of 20).

*FRRMAB.* The selector of MOEA/D-FRRMAB (Li et al., 2014): a sliding window of the fitness improvement rates (the sum over the replaced neighbours of the relative improvement of the Tchebycheff value), rank-based credit with decay D, and a UCB rule with scaling factor C. With the published parameters (W = 0.5N, C = 5, D = 1) it is effectively round-robin — each operator takes 0.124–0.127 of the selections on the CDP and about 0.125 on the GDP — because the credits are in [0, 1] and C = 5 makes the exploration term dominant; with C = 0.3 the shares stay within 0.117–0.141 (CDP) because the improvement rates are zero for most children late in the run. No variant beats uniform selection (CDP −0.011 to +0.006; GDP −0.017 to −0.007, one of them, C = 1, borderline significant: −0.017, p = 0.048).

*Longer searches.* We doubled and quadrupled the number of generations for uniform selection, the HV-reward UCB1 and the Gaussian HV sampler. More generations help uniform selection (CDP: +0.022 at ×2, Â12 0.61, p_Holm = 0.006; +0.036 at ×4, Â12 0.62, p_Holm < 0.0001; GDP: +0.015 and +0.045), but at the same budget no selector differs significantly from it (CDP −0.008 to +0.004, GDP −0.002 to +0.003).

## S2. Average ranks by instance family

Average ranks (HV_norm, mean of three replicates, 12 methods; lower is better) computed within each family (and size, for CDP/GDP). Descriptive only; generated by `analyze_by_family.py`.

**Table S2. Average ranks, MDP**

| Group | N | UCB-HV | TS-HV | GTS-HV | UCB-Scal | TS-Scal | GTS-Scal | Uniform | NSGA-II | GRASP-MO | GRASP-PR | VNS-Par | VNS-Tch | best AOS | best baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GKD-c | 10 | 2.40 | 3.60 | 3.80 | 6.90 | 4.60 | 4.00 | 2.70 | 8.00 | 10.00 | 9.30 | 10.90 | 11.80 | UCB-HV (2.40) | NSGA-II (8.00) |
| MDG-a | 10 | 5.40 | 4.70 | 5.30 | 6.20 | 7.00 | 4.80 | 5.60 | 11.90 | 8.00 | 8.60 | 6.70 | 3.80 | TS-HV (4.70) | VNS-Tch (3.80) |
| MDG-b | 40 | 3.84 | 4.86 | 3.38 | 6.90 | 6.10 | 4.39 | 4.42 | 11.05 | 8.40 | 8.28 | 8.88 | 7.51 | GTS-HV (3.38) | VNS-Tch (7.51) |
| MGPO | 5 | 2.80 | 4.60 | 2.00 | 3.60 | 3.60 | 6.20 | 5.20 | 11.20 | 8.60 | 8.40 | 11.60 | 10.20 | GTS-HV (2.00) | GRASP-PR (8.40) |
| SOM-a | 50 | 4.83 | 5.76 | 5.11 | 5.23 | 5.19 | 4.92 | 4.58 | 10.82 | 8.16 | 8.55 | 8.81 | 6.04 | Uniform (4.58) | VNS-Tch (6.04) |
| SOM-b | 12 | 4.25 | 3.92 | 4.42 | 3.62 | 5.12 | 4.96 | 4.58 | 10.75 | 10.17 | 10.08 | 8.83 | 7.29 | UCB-Scal (3.62) | VNS-Tch (7.29) |
| real_facility | 4 | 4.25 | 3.50 | 3.25 | 5.00 | 2.75 | 5.25 | 4.00 | 10.00 | 8.25 | 8.75 | 11.50 | 11.50 | TS-Scal (2.75) | GRASP-MO (8.25) |
| real_portfolio | 4 | 5.12 | 8.12 | 6.38 | 6.25 | 4.75 | 6.50 | 5.88 | 7.50 | 6.88 | 6.12 | 8.12 | 6.38 | TS-Scal (4.75) | GRASP-PR (6.12) |
| real_sensor | 4 | 3.75 | 4.25 | 2.25 | 4.75 | 3.50 | 4.25 | 5.25 | 10.00 | 8.50 | 8.50 | 12.00 | 11.00 | GTS-HV (2.25) | GRASP-MO (8.50) |
| real_team | 4 | 4.25 | 4.25 | 2.25 | 5.75 | 4.00 | 4.25 | 3.25 | 9.50 | 9.00 | 8.50 | 12.00 | 11.00 | GTS-HV (2.25) | GRASP-PR (8.50) |

**Table S3. Average ranks, CDP**

| Group | N | UCB-HV | TS-HV | GTS-HV | UCB-Scal | TS-Scal | GTS-Scal | Uniform | NSGA-II | GRASP-MO | GRASP-PR | VNS-Par | VNS-Tch | best AOS | best baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GKD-b n=150 | 20 | 3.40 | 3.20 | 3.75 | 6.05 | 5.65 | 2.80 | 3.60 | 10.10 | 8.35 | 8.20 | 11.25 | 11.65 | GTS-Scal (2.80) | GRASP-PR (8.20) |
| GKD-b n=50 | 20 | 3.75 | 3.80 | 3.30 | 5.80 | 5.70 | 3.55 | 3.15 | 10.80 | 8.55 | 7.40 | 10.80 | 11.40 | Uniform (3.15) | GRASP-PR (7.40) |
| GKD-c n=500 | 20 | 3.85 | 4.35 | 3.85 | 3.90 | 5.40 | 2.95 | 3.70 | 10.25 | 8.65 | 8.35 | 10.80 | 11.95 | GTS-Scal (2.95) | GRASP-PR (8.35) |
| MDG-b n=500 | 20 | 3.75 | 3.70 | 3.35 | 5.60 | 4.35 | 3.80 | 3.45 | 10.35 | 8.40 | 8.60 | 10.70 | 11.95 | GTS-HV (3.35) | GRASP-MO (8.40) |
| SOM-a n=50 | 20 | 4.15 | 5.00 | 3.80 | 5.20 | 4.40 | 4.85 | 4.20 | 11.10 | 7.15 | 6.25 | 11.45 | 10.45 | GTS-HV (3.80) | GRASP-PR (6.25) |

**Table S4. Average ranks, GDP**

| Group | N | UCB-HV | TS-HV | GTS-HV | UCB-Scal | TS-Scal | GTS-Scal | Uniform | NSGA-II | GRASP-MO | GRASP-PR | VNS-Par | VNS-Tch | best AOS | best baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GKD-b n=150 | 40 | 3.15 | 3.65 | 3.20 | 7.00 | 5.70 | 2.95 | 2.90 | 8.43 | 9.72 | 9.40 | 11.00 | 10.90 | Uniform (2.90) | NSGA-II (8.43) |
| GKD-b n=50 | 40 | 3.42 | 3.10 | 3.58 | 6.15 | 5.60 | 3.52 | 2.98 | 8.95 | 10.06 | 9.36 | 10.55 | 10.72 | Uniform (2.98) | NSGA-II (8.95) |
| GKD-c n=500 | 40 | 3.40 | 3.95 | 2.73 | 5.42 | 5.55 | 3.48 | 3.52 | 9.45 | 9.45 | 9.05 | 11.00 | 11.00 | GTS-HV (2.73) | GRASP-PR (9.05) |
| MDG-b n=500 | 40 | 3.52 | 3.12 | 3.05 | 5.92 | 4.83 | 3.58 | 3.98 | 9.85 | 9.47 | 9.07 | 10.65 | 10.95 | GTS-HV (3.05) | GRASP-PR (9.07) |
| SOM-a n=50 | 40 | 4.76 | 4.71 | 3.73 | 5.67 | 4.60 | 4.17 | 4.26 | 10.29 | 7.64 | 7.64 | 10.15 | 10.38 | GTS-HV (3.73) | GRASP-MO (7.64) |



## S3. Mechanism of the effect of a tight cost budget (E14)

We re-ran 540 runs of the standard-budget experiment (CDP and GDP; n = 500 and n = 150; ten instances per group — GDP with the tight and the loose cost budget, and CDP —, uniform selection, UCB-HV and GTS-HV, three replicates) with counters per operator: applications, children that violate the budget or the capacity before the repair step, children that improve at least one subproblem of the neighbourhood, and children that enter the non-dominated archive. The counters do not change the search: all 540 fronts are identical to those of the main experiment, with the same seeds. Table S5 gives the main figures for n = 500; the corresponding tables for n = 150 are in `results/analysis/e14_mechanism.json`.

**Table S5. Per-operator counters under uniform selection (10 instances per group, three replicates; same seeds as the main experiment): share of children infeasible before repair, share of children that enter the non-dominated archive, and share of the selections of UCB-HV**

| Operator | GDP n = 500, tight cost: infeasible / inserted / UCB-HV share | GDP n = 500, loose cost: infeasible / inserted / UCB-HV share | CDP n = 500: infeasible / inserted / UCB-HV share |
|---|---|---|---|
| RandomSwap1×1 | 0.343 / 0.0114 / 0.073 | 0.003 / 0.0069 / 0.084 | 0.007 / 0.0074 / 0.078 |
| BestSwap1×1 | 0.389 / 0.0189 / 0.088 | 0.003 / 0.0172 / 0.109 | 0.010 / 0.0194 / 0.101 |
| RandomSwap2×2 | 0.439 / 0.0034 / 0.064 | 0.005 / 0.0019 / 0.079 | 0.010 / 0.0020 / 0.074 |
| GraspGreedy | 0.000 / 0.0000 / 0.059 | 0.000 / 0.0000 / 0.076 | 0.000 / 0.0000 / 0.072 |
| BiObjBestSwap | 0.462 / 0.0708 / 0.317 | 0.004 / 0.0745 / 0.286 | 0.007 / 0.0809 / 0.303 |
| MaxMinSwap | 0.472 / 0.1079 / 0.252 | 0.003 / 0.0939 / 0.186 | 0.006 / 0.0999 / 0.205 |
| LargePerturbation | 0.598 / 0.0109 / 0.077 | 0.006 / 0.0128 / 0.102 | 0.016 / 0.0122 / 0.093 |
| SubsetCrossover | 0.000 / 0.0044 / 0.068 | 0.000 / 0.0014 / 0.078 | 0.000 / 0.0020 / 0.075 |
| All (mean of the eight) | 0.338 / 0.0285 / – | 0.003 / 0.0261 / – | 0.007 / 0.0280 / – |

The share of children that need repair is 34% on average on the tight-budget GDP at n = 500 (45% at n = 150), against 0.3% on the loose-budget GDP and 0.7% on the CDP. On the tight GDP the six operators that ignore the cost produce between 34% (RandomSwap1×1) and 60% (LargePerturbation) infeasible children; GraspGreedy and SubsetCrossover build feasible children by construction. Per application, BiObjBestSwap and MaxMinSwap insert 0.07–0.11 children into the archive in every group, against at most 0.02 for the other six operators, and the rates are about the same in the three groups; the standard deviation of the eight insertion rates is 0.037 on the tight GDP, 0.035 on the loose GDP and 0.038 on the CDP (maximum over mean 3.6, 3.8 and 3.8). The two HV-reward selectors move 0.57 (UCB-HV) and 0.54 (GTS-HV) of the selections to these two operators on the tight GDP, against 0.47 and 0.46 on the loose GDP and 0.51 and 0.50 on the CDP.
