# Model Card: BBO Capstone Optimisation Approach


---

## Overview

**Name**: Adaptive Bayesian Optimisation Pipeline for Black-Box Function Maximisation

**Type**: Sequential model-based optimisation (SMBO) with per-function adaptive configuration


**Summary**: An iterative Bayesian optimisation pipeline that maintains eight independent surrogate models - one per black-box function  each with function-specific acquisition functions, candidate generation strategies, range validation, and model selection via Leave-One-Out cross-validation. The pipeline evolved over ten weeks from global exploration to tight exploitation, incorporating specialist techniques including HEBO, CMA-ES, Dynamic BO, and model-free weighted centroid as evidence accumulated.

---

## Intended Use

**What tasks is this approach suitable for?**

- Sequential black-box optimisation of continuous functions in [0,1]^d where the function cannot be evaluated cheaply or in parallel
- Settings with a strict query budget (one query per function per iteration)
- Functions with unknown structure, noise characteristics, and dimensionality up to approximately 8 dimensions
- Problems where surrogate model accuracy can be validated via cross-validation on accumulated observations

**What use cases should be avoided?**

- Functions requiring more than one query per iteration to make meaningful progress - this pipeline is specifically designed for extreme query budgets
- Dynamic functions without a windowing or recency-weighting mechanism - fitting a standard GP on full history from a non-stationary function will produce unreliable surrogates (demonstrated by F4)
- Very high-dimensional functions (d > 10) where the GP's computational cost and the curse of dimensionality make reliable surrogate fitting impractical without additional techniques
- Settings where the true function is known to be multimodal and the budget is insufficient for global exploration - premature exploitation of local optima is a documented failure mode (demonstrated by F5)
- Real-world deployment decisions where surrogate predictions have not been validated appropriately

---

## Strategy Details

### Week 1–3: Global Exploration

**Techniques**: Gaussian Process with RBF kernel, UCB acquisition function (beta=2-5), Latin Hypercube Sampling across full [0,1]^d space.

**Rationale**: No prior knowledge of function landscape. LHS provides space-filling coverage. UCB with high beta rewards uncertainty - explores broadly rather than committing to early promising regions.

**Outcome**: Established baseline observations for all 8 functions. Identified rough promising regions for F1, F2, F5. Recognised that UCB was returning similar candidates week-on-week for F7/F8 - sign of model failure to learn from new data.

---

### Weeks 4–6: Exploitation with EI

**Techniques**: Expected Improvement replaced UCB as primary acquisition function. Confirmed ranges identified per function. Length scale bounds tightened after observing dimensions hitting upper bounds (suggesting GP ignoring those dimensions).

**Rationale**: EI balances exploitation of known good regions with exploration of uncertain nearby regions, but is more focused than UCB for functions where promising regions had been identified. Lower xi (exploration parameter) applied to functions with confirmed peaks.

**Key decisions**:
- F1: Signed log transform introduced, raw Y values span 17 orders of magnitude, preventing meaningful GP fitting
- F3: x1/x2 identified as near-fixed, x3 as the active dimension
- F5: Committed to exploitation of x1≈0.31, x2≈0.86 region - this later proved to be a local optimum

**Outcome**: New bests established for most functions. F5 plateau at Y≈3006 incorrectly interpreted as near-optimum.

---

### Weeks 7–8: Surrogate Diversification

**Techniques**: SVR with GridSearchCV tuning added alongside GP. LOO cross-validation introduced for objective surrogate selection. RF and NN added for F8. Plausibility checks (70%-130% of best real Y) added to prevent surrogate hallucinations influencing submissions.

**Rationale**: Single GP surrogate insufficient for all function types. SVR handles tight clusters differently from GP. LOO provides empirical evidence for model selection rather than assumption. Plausibility check addresses GP tendency to predict implausibly high values in unexplored regions.

**Key decisions**:
- F4: LOO RMSE of 1.437 on full history flagged as problematic - motivated later Dynamic BO development
- F7: HEBO introduced - NeurIPS 2020 BBO Challenge winner, handles suspected log-scale hyperparameters through automatic per-dimension input scaling
- F8: CMA-ES introduced - tight local exploitation anchored to confirmed best point with sigma0 = min_range/6

**Outcome**: SVR became primary surrogate for F4 and F6. HEBO and CMA-ES consistently outperformed GP/SVR for F7 and F8 respectively.

---

### Weeks 9–10: Advanced Techniques and Course Correction

**Techniques**: Dynamic BO region window (F4), UCB exploration (F5), weighted centroid model-free candidate (F6), range validation on all candidates, asymmetric bias search (F1).

**Key decisions**:

**F4 Dynamic BO**: Time-based sliding window (last k observations) failed because recent queries included Y=-23 negatives from exploratory queries outside the positive region. Replaced with region-based filter - only observations within dist=0.09 of confirmed best point included in surrogate fitting. LOO RMSE dropped from 1.437 to 0.125. New best 0.7097 achieved.

**F5 UCB Exploration**: Competitor analysis revealed another student achieved Y=8585 in week 3 - confirming a much higher global optimum existed. UCB with beta=5 across unexplored x1=0.40-0.90, x2=0.20-0.70 region returned Y=4039 - new best, confirming the local optimum failure. GP subsequently pointed consistently toward x1≈0.95, x2≈0.74 as potentially higher.

**F6 Weighted Centroid**: Model-free candidate added - top-k results weighted by 1/|Y|, weighted average taken as candidate.  Added alongside GP and SVR rather than replacing them.

**F1 Directional Search Adjustment**: Week 9 returned a new best at x2=0.7257, marginally improving on the previous best at x2=0.7266. Since x1 also shifted between the two results, the improvement could not be attributed to x2 alone. The search was deliberately weighted toward lower x2 values as a testable hypothesis, while acknowledging both dimensions may have contributed to the improvement.

**Outcome**: 6 new bests out of 8 functions in week 9. F5 course correction confirmed. F4 Dynamic BO validated.

---

## Performance

### Best Results Achieved (Week 10)

| Function | Best Y | Week achieved | Method |
|----------|--------|--------------|--------|
| F1 | 7.89e-15 | Week 10 | Proximity search, asymmetric x2 bias |
| F2 | 0.7392 | Week 4 | GP EI near confirmed peak |
| F3 | −0.00600 | Week 10 | SVR uniform, fixed x1/x2, x3=0.472 |
| F4 | 0.7097 | Week 10 | SVR with Dynamic BO region window |
| F5 | 4039.81 | Week 10 | UCB exploration of new region |
| F6 | −0.1615 | Week 10 | SVR tightened ranges |
| F7 | 2.9071 | Week 7 | HEBO with tightened x2/x5/x6 |
| F8 | 9.8799 | Week 10 | CMA-ES sigma0=0.02 |

### Metrics Used

**Primary metric**: Best real Y returned by the course platform for each function is the baseline.

**Surrogate validation metric**: LOO RMSE - measures how well each surrogate model predicts held-out observations. Used to select the primary model each week. Values ranged from 0.076 (F3 GP, well-fitted) to 475 (F5 GP, landscape too complex for reliable interpolation).

**Candidate quality metrics**: Distance from confirmed best point (proximity check), range validation (all dimensions within confirmed bounds), plausibility check (prediction within 70%-130% of best real Y).

### Surrogate Performance by Function

| Function | Primary surrogate | LOO RMSE (week 10) | Notes |
|----------|-----------------|-------------------|-------|
| F1 | Proximity search | N/A | GP unreliable at sub-1e-15 scale |
| F2 | Proximity search | 0.187 (GP) | Noisy function - proximity more defensible |
| F3 | GP | 0.076 | x2 length scale hitting bound - GP ignoring x2 |
| F4 | SVR | 0.125 | Region window - 8 positive observations |
| F5 | GP | 320 | High RMSE - complex landscape with two distinct peaks |
| F6 | SVR | 0.213 | GP hitting x1 length scale bound |
| F7 | HEBO | 0.411 (GP) | HEBO overrides standard GP - better per-dimension scaling |
| F8 | GP | 0.141 | CMA-ES wins selection - GP/RF/NN excluded by range check |

---

## Assumptions and Limitations

### Key Assumptions

**Stationarity**: The GP assumes the function landscape does not change between queries. This assumption is explicitly violated by F4 (dynamic function). For all other functions stationarity is assumed but not guaranteed.

**Smoothness**: RBF and Matérn kernels assume smooth continuous relationships between inputs and outputs. This is appropriate for F3 and F6 but questionable for F8, where x8 consistently hits a length scale of 1000, suggesting either an encoded categorical variable with discrete jumps or a genuinely insensitive dimension that the smoothness assumption cannot represent correctly.

**Unimodality**: The F5 brief described the function as "typically unimodal." This assumption caused premature exploitation of a local optimum from weeks 4-9. The function demonstrated at least two distinct high-value regions, making the unimodality assumption incorrect and costing approximately 6 weeks of queries.

**Confirmed range validity**: Range validation assumes that dimensions outside the empirically confirmed ranges will return poor results. This is supported by evidence but not guaranteed, it is possible that unexplored regions outside the confirmed ranges contain better results that were never discovered.

### Constraints

**Query budget**: One query per function per week creates an extreme exploitation-exploration trade-off. Every exploratory query foregoes an exploitation opportunity with no recovery mechanism within that week. This constraint directly caused the F5 local optimum failure.

**Computational scaling**: GP fitting scales as O(n³) with observation count. With 49 observations for F8 and n_restarts_optimizer=10, fitting time is non-trivial. At 100+ observations a standard GP would become impractical on a standard laptop.

**Dimensionality**: With 49 observations across 8 dimensions (F8), the dataset is significantly under-sampled relative to dimensionality. Standard practice suggests 10-30 observations per dimension for reliable GP generalisation - F8 has approximately 6 per dimension.

### Failure Modes

**Premature exploitation** (demonstrated F5): High early returns anchored the search to a local optimum. UCB exploration with high beta is the recommended mitigation but was applied too late.

**Surrogate hallucination** (mitigated by plausibility check): GP occasionally predicts implausibly high values in unexplored regions. The 70%-130% plausibility bound catches the most outlyers.

**Boundary saturation** (observed F7, F8): SVR and GP occasionally suggest candidates with multiple dimensions simultaneously at boundary values - a sign of extrapolation rather than genuine landscape modelling. Range validation catches this when it pushes outside confirmed bounds.

**Dynamic landscape mismatch** (demonstrated F4): Fitting a standard GP on full history from a non-stationary function produces high LOO RMSE and unreliable suggestions. The region-based window is a partial but not complete solution.

---

## Ethical Considerations

### Transparency and Reproducibility

Every submission decision is logged with its full justification - LOO RMSE values, range validation results, plausibility bounds, candidate distances, and the acquisition function scores that generated each candidate. A researcher reviewing the weekly code outputs could trace exactly why each submission was chosen.

The LOO cross-validation provides an objective, reproducible model selection criterion. Any researcher running the same code on the same accumulated observations would select the same primary surrogate model. The one exception is manual overrides - documented in weekly write-up notes but not encoded in the selection logic - which create a reproducibility gap that was acknowledged and partially addressed in later weeks.

The signed log transform applied to F1 is fully documented with the formula, epsilon value, and rationale. Any researcher could reproduce the exact transformation and verify its effect on the GP fitting.

### Limitations of Transparency

The black-box nature of the competition means the true function definitions are unknown. While the optimisation strategy is fully transparent, the relationship between submitted coordinates and returned outputs cannot be explained - only observed. This limits the ability to provide mechanistic explanations for why certain regions perform better.

The manual override decisions (notably F7 week 10, where HEBO was selected over the LOO primary model SVR) reflect practitioner judgement that is documented but not coded. 

### Real-World Adaptation

The techniques developed in this project - GP surrogates with LOO validation, CMA-ES for tight exploitation, UCB for exploration, Dynamic BO windowing for non-stationary functions are directly applicable to real-world optimisation problems including hyperparameter tuning, materials discovery, drug design, and industrial process optimisation. The explicit documentation of assumptions, failure modes, and the conditions under which each technique is appropriate supports responsible adaptation to new domains.

The F5 premature exploitation failure is a documented cautionary example with direct real-world relevance - in any optimisation setting where early returns appear high, deliberate global exploration should be scheduled before committing to exploitation, regardless of how promising the early results appear.

---

## References

- Mitchell et al. (2019) - Model Cards for Model Reporting
- Jones et al. (1998) - Expected Improvement acquisition function
- Rasmussen & Williams (2006) - Gaussian Processes for Machine Learning
- Srinivas et al. (2010) - UCB acquisition function
- Brochu, Cora & de Freitas (2010) - Bayesian Optimisation tutorial
- Cowen-Rivers et al. (2022) - HEBO: Heteroscedastic Evolutionary Bayesian Optimisation
- Bogunovic et al. (2016) - Dynamic Bayesian Optimisation
- Turner et al. (2021) - NeurIPS 2020 Black-Box Optimisation Challenge
- Arlot & Celisse (2010) - Leave-One-Out cross-validation survey
- Bergstra & Bengio (2012) - Random search for hyperparameter optimisation
