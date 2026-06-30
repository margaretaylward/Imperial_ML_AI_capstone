# Model Card: BBO Capstone Optimisation Approach

---

## Overview

**Name**: Adaptive Bayesian Optimisation Pipeline for Black-Box Function Maximisation

**Type**: Sequential model-based optimisation (SMBO) with per-function adaptive configuration

**Summary**: An iterative Bayesian optimisation pipeline that maintains eight independent surrogate models — one per black-box function — each with function-specific acquisition functions, candidate generation strategies, range validation, and model selection via Leave-One-Out cross-validation. The pipeline evolved over thirteen weeks from global exploration to tight exploitation, incorporating specialist techniques including HEBO, CMA-ES, Dynamic BO, and model-free weighted centroid as evidence accumulated.

---

## Model Description

**Input**: Continuous valued vectors in [0,1]^d, where d ranges from 2 to 8 depending on the function. One input vector submitted per function per week across 13 weeks.

**Output**: A scalar Y value returned by the black-box platform representing the function's output at the submitted coordinates. The goal is to maximise this value across all eight functions over the full project duration.

**Model Architecture**: An adaptive multi-surrogate pipeline combining Gaussian Process regression, Support Vector Regression, Random Forest, Neural Network, HEBO, and Optuna CMA-ES — each selected per function per week via Leave-One-Out cross-validation. Candidate generation uses Latin Hypercube Sampling, local noise perturbation, and UCB exploration. Final submission selection applies range validation and plausibility checks before committing to any surrogate suggestion.

---

## Intended Use

**What tasks is this approach suitable for?**

- Sequential black-box optimisation of continuous functions in [0,1]^d where the function cannot be evaluated cheaply or in parallel
- Settings with a strict query budget (one query per function per iteration)
- Functions with unknown structure, noise characteristics, and dimensionality up to approximately 8 dimensions
- Problems where surrogate model accuracy can be validated via cross-validation on accumulated observations

**What use cases should be avoided?**

- Functions requiring more than one query per iteration to make meaningful progress — this pipeline is specifically designed for extreme query budgets
- Dynamic functions without a windowing or recency-weighting mechanism — fitting a standard GP on full history from a non-stationary function will produce unreliable surrogates (demonstrated by F4)
- Very high-dimensional functions (d > 10) where the GP's computational cost and the curse of dimensionality make reliable surrogate fitting impractical without additional techniques
- Settings where the true function is known to be multimodal and the budget is insufficient for global exploration — premature exploitation of local optima is a documented failure mode (demonstrated by F5)
- Real-world deployment decisions where surrogate predictions have not been validated appropriately

---

## Strategy Details

### Weeks 1–3: Global Exploration

**Techniques**: Gaussian Process with RBF kernel, UCB acquisition function (beta=2-5), Latin Hypercube Sampling across full [0,1]^d space.

**Rationale**: No prior knowledge of function landscape. LHS provides space-filling coverage. UCB with high beta rewards uncertainty — explores broadly rather than committing to early promising regions.

**Outcome**: Established baseline observations for all 8 functions. Identified rough promising regions for F1, F2, F5. Recognised that UCB was returning similar candidates week-on-week for F7/F8 — sign of model failure to learn from new data.

---

### Weeks 4–6: Exploitation with EI

**Techniques**: Expected Improvement replaced UCB as primary acquisition function. Confirmed ranges identified per function. Length scale bounds tightened after observing dimensions hitting upper bounds (suggesting GP ignoring those dimensions).

**Rationale**: EI balances exploitation of known good regions with exploration of uncertain nearby regions, but is more focused than UCB for functions where promising regions had been identified. Lower xi (exploration parameter) applied to functions with confirmed peaks.

**Key decisions**:
- F1: Signed log transform introduced — raw Y values span 17 orders of magnitude, preventing meaningful GP fitting
- F3: x1/x2 identified as near-fixed, x3 as the active dimension
- F5: Committed to exploitation of x1≈0.31, x2≈0.86 region — this later proved to be a local optimum

**Outcome**: New bests established for most functions. F5 plateau at Y≈3006 incorrectly interpreted as near-optimum.

---

### Weeks 7–8: Surrogate Diversification

**Techniques**: SVR with GridSearchCV tuning added alongside GP. LOO cross-validation introduced for objective surrogate selection. RF and NN added for F8. Plausibility checks (70%-130% of best real Y) added to prevent surrogate hallucinations influencing submissions.

**Rationale**: Single GP surrogate insufficient for all function types. SVR handles tight clusters differently from GP. LOO provides empirical evidence for model selection rather than assumption. Plausibility check addresses GP tendency to predict implausibly high values in unexplored regions.

**Key decisions**:
- F4: LOO RMSE of 1.437 on full history flagged as problematic — motivated later Dynamic BO development
- F7: HEBO introduced — NeurIPS 2020 BBO Challenge winner, handles suspected log-scale hyperparameters through automatic per-dimension input scaling
- F8: CMA-ES introduced — tight local exploitation anchored to confirmed best point with sigma0 = min_range/6

**Outcome**: SVR became primary surrogate for F4 and F6. HEBO and CMA-ES consistently outperformed GP/SVR for F7 and F8 respectively.

---

### Weeks 9–10: Advanced Techniques and Course Correction

**Techniques**: Dynamic BO region window (F4), UCB exploration (F5), weighted centroid model-free candidate (F6), range validation on all candidates, asymmetric bias search (F1).

**F4 Dynamic BO**: The windowing mechanism for Function 4 underwent two iterations before converging on the region-based filter used in the final rounds. An initial time-based sliding window (k=10 most recent observations) was implemented following Bogunovic et al. (2016) but failed in practice because recent exploratory queries outside the positive region introduced Y=-23 negatives into the fitting window. With only 10 observations and a Y range spanning from -23 to +0.72, the surrogate was fitting an average of multiple landscape configurations rather than the current one and a LOO RMSE of 5.204 confirmed this failure. The region-based filter replaced the time window: only observations within a Euclidean distance threshold of the confirmed best point were included in surrogate fitting, regardless of query order. The threshold of 0.09 was determined by inspecting the nearest neighbour distance distribution in the ranked results — all 9 positive observations clustered naturally within this distance from the confirmed best, while all negative observations fell beyond dist=0.25. LOO RMSE dropped from 1.437 to 0.125, an 87% reduction.

**F5 UCB Exploration**: Competitor analysis revealed another student achieved Y=8585 in prior weeks — confirming a much higher global optimum existed. UCB with beta=5 across unexplored x1=0.40-0.90, x2=0.20-0.70 region returned Y=4039, confirming the local optimum failure. GP subsequently pointed consistently toward x1≈0.95, x2≈0.74 as potentially higher.

**F6 Weighted Centroid**: Model-free candidate added — top-k results weighted by 1/|Y|, weighted average taken as candidate. Added alongside GP and SVR rather than replacing them.

**F1 Directional Search Adjustment**: Week 9 returned a new best at x2=0.7257, marginally improving on the previous best at x2=0.7266. Since x1 also shifted between the two results, the improvement could not be attributed to x2 alone. The search was deliberately weighted toward lower x2 values as a testable hypothesis, while acknowledging both dimensions may have contributed to the improvement.

**Outcome**: 6 new bests out of 8 functions in week 9. F5 course correction confirmed. F4 Dynamic BO validated.

---

### Weeks 11–13: Convergence and Final Exploitation

**Techniques**: Pure exploitation across all functions. CMA-ES, HEBO, and proximity search maintained per function. Manual overrides documented where model evidence conflicted with empirical data.

**Key decisions**: F5 followed the GP gradient toward the boundary corner [1,1,1,1], delivering three consecutive new bests (4849, 8464, 8662), validating the week 10 UCB pivot. F8 benefited from a deliberate GP gamble in week 11 (LOO RMSE 0.089, best recorded) returning 9.9613, with CMA-ES anchoring to the new peak for the remaining weeks. F6 confirmed genuine observation noise on the final query — identical coordinates returning -0.3056 in week 11 and -0.1425 in week 13.

**Outcome**: 5 new bests across weeks 12 and 13. The pipeline's LOO cross-validation, range validation and plausibility checks proved reliable enough to trust in the final weeks without further structural changes.

---

## Performance

### Best Results Achieved (Week 13 — Final)

| Function | Best Y | Week achieved | Method |
|----------|--------|--------------|--------|
| F1 | 2.828936e-14 | Week 13 | Proximity search, confirmed lower x2 trajectory |
| F2 | 0.739239 | Week 4 | GP EI near confirmed peak — noisy function, best held |
| F3 | -0.006004 | Week 10 | SVR uniform, fixed x1/x2, x3=0.472 confirmed peak |
| F4 | 0.723045 | Week 12 | SVR local with Dynamic BO region window |
| F5 | 8662.405001 | Week 12 | GP local following gradient to boundary corner [1,1,1,1] |
| F6 | -0.142570 | Week 13 | Weighted centroid — noise confirmed by duplicate coordinate |
| F7 | 2.924410 | Week 11 | HEBO override — input warping outperformed GP/SVR |
| F8 | 9.961345 | Week 11 | GP gamble week 11, CMA-ES exploitation weeks 12-13 |

### Metrics Used

**Primary metric**: Best real Y returned by the course platform for each function.

**Surrogate validation metric**: LOO RMSE — measures how well each surrogate model predicts held-out observations. Used to select the primary model each week. Values ranged from 0.083 (F3 GP, well-fitted) to 395 (F5 GP, landscape too complex for reliable interpolation).

**Candidate quality metrics**: Distance from confirmed best point (proximity check), range validation (all dimensions within confirmed bounds), plausibility check (prediction within 70%-130% of best real Y).

### Surrogate Performance by Function

| Function | Primary surrogate | LOO RMSE (final) | Notes |
|----------|-----------------|-----------------|-------|
| F1 | Proximity search | N/A | GP unreliable at sub-1e-14 scale — signed log transform applied |
| F2 | Proximity search | 0.177 (GP) | Noisy function — WhiteKernel σ²≈0.037, proximity more defensible than EI |
| F3 | GP | 0.083 | x1/x2 length scales hitting bound — GP correctly ignoring fixed dimensions |
| F4 | SVR | 0.112 | Region window — 10 positive observations, LOO improved from 1.437 to 0.112 |
| F5 | GP | 395 | High RMSE reflects complex two-region landscape — plausibility override active |
| F6 | SVR | 0.210 | GP hitting x1 length scale bound — observation noise confirmed on final query |
| F7 | HEBO | 0.306 (GP) | GP/SVR failed range check on x4 every week — HEBO only eligible candidate |
| F8 | GP | 0.089 | Best LOO recorded — CMA-ES selected over GP for final submissions |

---

## Assumptions and Limitations

### Key Assumptions

**Stationarity**: The GP assumes the function landscape does not change between queries. This assumption is explicitly violated by F4 (dynamic function). For all other functions stationarity is assumed but not guaranteed.

**Smoothness**: RBF and Matérn kernels assume smooth continuous relationships between inputs and outputs. This is appropriate for F3 and F6 but questionable for F8, where x8 consistently hits a length scale of 1000, suggesting either an encoded categorical variable with discrete jumps or a genuinely insensitive dimension that the smoothness assumption cannot represent correctly.

**Unimodality**: The F5 brief described the function as "typically unimodal." This assumption caused premature exploitation of a local optimum from weeks 4-9. The function demonstrated at least two distinct high-value regions.

**Confirmed range validity**: Range validation assumes that dimensions outside the empirically confirmed ranges will return poor results. This is supported by evidence but not guaranteed — it is possible that unexplored regions outside the confirmed ranges contain better results that were never discovered.

### Constraints

**Query budget**: One query per function per week creates an extreme exploitation-exploration trade-off. Every exploratory query foregoes an exploitation opportunity with no recovery mechanism within that week. This constraint directly caused the F5 local optimum failure.

**Computational scaling**: GP fitting scales as O(n³) with observation count. With 52 observations for F8 and n_restarts_optimizer=10, fitting time is non-trivial. At 100+ observations a standard GP would become impractical on a standard laptop.

**Dimensionality**: With 52 observations across 8 dimensions (F8), the dataset is significantly under-sampled relative to dimensionality. Standard practice suggests 10-30 observations per dimension for reliable GP generalisation — F8 has approximately 6 per dimension.

### Proximity Search Calibration at Length Scale Boundaries

When a Gaussian Process length scale hits its upper bound, the GP has learned to treat that dimension as flat — moving along it makes no meaningful difference to the predicted output. For Function 1, where Y values span approximately 17 orders of magnitude, a signed log transform was first applied to compress the range to a GP-tractable scale. Even after transformation, the GP's length scale on x2 remained large because the peak is extremely narrow — the function drops off so sharply around the true source that the GP's smooth kernel cannot represent it accurately at sub-femtometre precision.

In this regime proximity search was adopted as a structurally justified fallback rather than an arbitrary override. When LOO RMSE exceeds a meaningful fraction of the Y range of interest, surrogate predictions are unreliable as absolute guides. For Function 1, LOO RMSE of approximately 2 units in transformed space corresponds to multiple orders of magnitude in raw Y space, making EI scores meaningless for distinguishing candidates within the confirmed neighbourhood. Proximity to the empirically confirmed best point maximises the probability of querying near the true optimum without requiring reliable absolute predictions.

For Function 8, x8 consistently hit a length scale of 1000 throughout the project, consistent with the brief's suggestion of an encoded categorical variable where the smoothness assumption breaks down at category boundaries. In this case the range validation check served the same structural role as proximity search did for F1 — preventing the surrogate from acting on a dimension it could not reliably model, and deferring instead to CMA-ES whose covariance-learning approach does not assume smoothness across all dimensions equally.

### Failure Modes

**Premature exploitation** (demonstrated F5): High early returns anchored the search to a local optimum. UCB exploration with high beta is the recommended mitigation but was applied too late.

**Surrogate hallucination** (mitigated by plausibility check): GP occasionally predicts implausibly high values in unexplored regions. The 70%-130% plausibility bound catches the most significant outliers.

**Boundary saturation** (observed F7, F8): SVR and GP occasionally suggest candidates with multiple dimensions simultaneously at boundary values — a sign of extrapolation rather than genuine landscape modelling. Range validation catches this when it pushes outside confirmed bounds.

**Dynamic landscape mismatch** (demonstrated F4): Fitting a standard GP on full history from a non-stationary function produces high LOO RMSE and unreliable suggestions. The region-based window is a partial but not complete solution.

---

## Trade-offs

**Exploration vs exploitation**: Every exploratory query foregoes an exploitation opportunity with no recovery within that week. This trade-off was managed differently per function and per phase — UCB for early exploration, EI for mid-phase balance, and proximity or CMA-ES for final exploitation. The most costly trade-off failure was F5, where six weeks of exploitation in a local optimum restricted earlier discovery of the true peak at the boundary corner.

**Model complexity vs data availability**: More sophisticated surrogates require more data to generalise reliably. With 52 observations across 8 dimensions for F8, the Neural Network consistently underperformed (LOO RMSE 2.17) while the simpler GP achieved 0.089. Complexity was therefore calibrated to the available data rather than applied uniformly across all functions.

**Surrogate trust vs empirical evidence**: LOO cross-validation selects the primary surrogate objectively, but manual overrides were applied where the selected model's suggestion conflicted with accumulated empirical evidence — notably F7 where HEBO was chosen over the LOO primary model SVR on multiple occasions. This trade-off between algorithmic objectivity and practitioner judgement is documented throughout but creates a reproducibility gap.

---

## Ethical Considerations

### Transparency and Reproducibility

Every submission decision is logged with its full justification — LOO RMSE values, range validation results, plausibility bounds, candidate distances, and the acquisition function scores that generated each candidate. A researcher reviewing the weekly code outputs could trace exactly why each submission was chosen.

The LOO cross-validation provides an objective, reproducible model selection criterion. Any researcher running the same code on the same accumulated observations would select the same primary surrogate model. The one exception is manual overrides — documented in weekly write-up notes but not encoded in the selection logic — which create a reproducibility gap that was acknowledged and partially addressed in later weeks.

The signed log transform applied to F1 is fully documented with the formula, epsilon value, and rationale. Any researcher could reproduce the exact transformation and verify its effect on the GP fitting.

### Limitations of Transparency

The black-box nature of the competition means the true function definitions are unknown. While the optimisation strategy is fully transparent, the relationship between submitted coordinates and returned outputs cannot be explained — only observed. This limits the ability to provide mechanistic explanations for why certain regions perform better.

The manual override decisions (notably F7 where HEBO was selected over the LOO primary model SVR) reflect practitioner judgement that is documented but not algorithmically encoded.

### Real-World Adaptation

The techniques developed in this project — GP surrogates with LOO validation, CMA-ES for tight exploitation, UCB for exploration, Dynamic BO windowing for non-stationary functions — are directly applicable to real-world optimisation problems including hyperparameter tuning, materials discovery, drug design, and industrial process optimisation. The explicit documentation of assumptions, failure modes, and the conditions under which each technique is appropriate supports responsible adaptation to new domains.

The F5 premature exploitation failure is a documented cautionary example with direct real-world relevance — in any optimisation setting where early returns appear high, deliberate global exploration should be scheduled before committing to exploitation, regardless of how promising the early results appear.

---

## References

- Mitchell et al. (2019) — Model Cards for Model Reporting
- Jones et al. (1998) — Expected Improvement acquisition function
- Rasmussen & Williams (2006) — Gaussian Processes for Machine Learning
- Srinivas et al. (2010) — UCB acquisition function
- Brochu, Cora & de Freitas (2010) — Bayesian Optimisation tutorial
- Cowen-Rivers et al. (2022) — HEBO: Heteroscedastic Evolutionary Bayesian Optimisation
- Bogunovic et al. (2016) — Dynamic Bayesian Optimisation
- Turner et al. (2021) — NeurIPS 2020 Black-Box Optimisation Challenge
- Arlot & Celisse (2010) — Leave-One-Out cross-validation survey
- Bergstra & Bengio (2012) — Random search for hyperparameter optimisation
