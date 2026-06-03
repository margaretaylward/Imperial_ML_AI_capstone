# Datasheet: BBO Capstone Project Dataset



## Motivation

**Why was this dataset created?**

This dataset was created as part of a Black-Box Optimisation (BBO) capstone project. The objective is to optimise eight unknown functions by iteratively querying each function with input vectors and recording the scalar outputs. The dataset accumulates over a 13-week period, with one query per function per week, building a record of inputs, outputs, and the surrogate modelling decisions that informed each query.

**What task does it support?**

The dataset supports the development, evaluation, and comparison of Bayesian optimisation strategies including Gaussian Process (GP) surrogates, Support Vector Regression (SVR), Random Forest (RF), Neural Networks (NN), HEBO, and Optuna-based samplers (CMA-ES, TPE). It documents the full optimisation trajectory for each function, enabling analysis of exploration-exploitation trade-offs, surrogate model accuracy, and acquisition function behaviour across diverse function types.

**Who created it and for what purpose?**

Created by a student as part of a structured academic assignment. The dataset serves both as the primary experimental record and as evidence of iterative strategy development for assessment purposes.

---

## Composition

**What does the dataset contain?**

The dataset contains query records for 8 black-box functions across up to 13 weeks of submissions. Each record consists of:

- **Input vector X**: continuous values in [0, 1] per dimension
- **Output scalar Y**: the function's return value for that input
- **Week number**: the submission round (1–13)
- **Function identifier**: F1 through F8

| Function | Dimensions | Type | Goal | Notes |
|----------|-----------|------|------|-------|
| F1 | 2 | Radiation detection | Maximise | Point source, inverse square law behaviour |
| F2 | 2 | Noisy log-likelihood | Maximise | Explicit observation noise |
| F3 | 3 | Drug discovery | Minimise side effects | Framed as maximisation of negative |
| F4 | 4 | Warehouse optimisation | Maximise | Dynamic - landscape shifts each week |
| F5 | 4 | Chemical process yield | Maximise | Unimodal, two distinct peak regions discovered |
| F6 | 5 | Cake recipe | Maximise toward zero | All outputs negative |
| F7 | 6 | ML hyperparameter tuning | Maximise | Suspected log-scale dimensions |
| F8 | 8 | ML hyperparameter tuning | Maximise | Suspected encoded categorical variables |

**What is the size of the dataset?**

As of week 10, the dataset contains approximately 340 records in total across all 8 functions (approximately 28–49 records per function depending on early exploration queries). The dataset grows by 8 records per week (one per function).

**What is the format?**

Data is stored as structured observations within Jupyter notebooks (`.ipynb`) and Python scripts (`.py`). Each function's data is loaded as NumPy arrays:

```python
X  # shape (n, dim) - input vectors
Y  # shape (n,)     - scalar outputs
```

No external file format (CSV, JSON) is used - data is embedded directly in the notebook environment provided by the course.

**Are there any gaps or missing values?**

There are no missing values within the recorded observations. However the dataset is intentionally sparse relative to the dimensionality of each function - this sparsity is a core characteristic of the BBO setting and not a data quality issue. F4 (dynamic function) has structural inconsistency across observations because the underlying function changes between weeks, making historical observations partially misleading for current landscape modelling.

---

## Collection Process

**How were queries generated?**

Queries were generated through a weekly iterative process:

1. Fit surrogate models (GP, SVR, RF, NN, HEBO) on all historical observations
2. Generate candidate points using multiple strategies (local noise, Latin Hypercube Sampling, Optuna CMA-ES/TPE, UCB exploration)
3. Score candidates using acquisition functions (Expected Improvement, Upper Confidence Bound, Thompson Sampling, proximity search)
4. Apply range validation and plausibility checks to filter implausible suggestions
5. Select the final submission based on LOO cross-validation primary model
6. Submit to the course platform and record the returned output

**What strategy was used?**

Strategy evolved across the 12 weeks:

- **Weeks 1–3**: Global exploration using Latin Hypercube Sampling across the full [0,1]^d space
- **Weeks 4–7**: Exploitation of confirmed good regions with tightening search bounds per function
- **Weeks 8–10**: Advanced surrogates introduced (HEBO for F7, CMA-ES for F5/F8, Dynamic BO for F4), UCB exploration for F5 after competitor analysis revealed a missed global optimum
- **Weeks 11–13**: Pure exploitation of confirmed peaks with tight local search

**Over what time frame?**

Weeks 1–13 of the academic term, with one submission round per week. Each weekly iteration involved approximately 2–4 hours of analysis, code development, and strategy refinement.

**Was there any sampling bias?**

Yes - F5 exhibits a documented anchoring bias. High early returns (Y≈2700–3006) in a local optimum caused premature exploitation from weeks 4–9. A competitor achieved Y≈8585 by finding the true global optimum in week 3, demonstrating that early high returns can mislead exploration strategy. UCB exploration in week 10 identified a new region returning Y=4039, partially correcting for this bias.

---

## Preprocessing and Uses

**Have any transformations been applied?**

Yes, function-specific transformations were applied:

- **F1**: Signed log transform applied to Y values before GP fitting. Raw Y spans 17 orders of magnitude (7.82e-23 to 7.89e-15); log transform compresses this to a manageable scale while preserving sign. Formula: `sign(y) * log1p(|y| / epsilon)` where `epsilon = 1e-25`
- **F3**: Y values negated conceptually - the function returns negative side-effect scores; maximising Y means minimising adverse reactions
- **F4**: Region-based window filtering applied before surrogate fitting - only observations within Euclidean distance 0.09 of the current best point are used for model fitting, addressing the dynamic landscape problem
- **All functions**: StandardScaler or MinMaxScaler applied to X before GP/SVR fitting; inverse transform applied before recording submissions

**What are the intended uses?**

- Evaluating and comparing Bayesian optimisation strategies across diverse function types
- Studying exploration-exploitation trade-offs in limited-budget settings (one query per function per week)
- Demonstrating surrogate model selection via LOO cross-validation
- Academic assessment of iterative optimisation strategy development

**What are inappropriate uses?**

- The dataset should not be used to make claims about the true optima of the underlying functions - the true function definitions are unknown (black-box) and optimal values have not been confirmed
- F4 observations should not be used to train a single global surrogate - the dynamic landscape means older observations reflect different function configurations and are partially misleading
- The dataset is not suitable for benchmarking general-purpose ML algorithms due to its small size and function-specific characteristics
- Results should not be generalised beyond the specific function instances used in this competition

---

## Distribution and Maintenance

**Where is the dataset available?**

The dataset is available within the course Jupyter notebook environment provided by the institution. Selected observations, analysis scripts, and documentation are maintained in a private GitHub repository created for this project.

**What are the terms of use?**

The dataset was generated through a course platform provided by the institution. The underlying function definitions are proprietary to the course and cannot be shared or reproduced outside the academic context. Student-generated query records and analysis code are the student's own work and subject to standard academic integrity policies.

**Who maintains the dataset?**

The student maintains the query records and analysis code in the GitHub repository. The course platform (which evaluates submitted queries and returns function outputs) is maintained by the course instructors. No automated pipeline exists - updates are manual and weekly.

**Will the dataset be updated?**

Yes, the dataset grows by 8 records per week until the end of week 13. After the competition concludes the dataset will be complete and no further updates are planned.

---

## Additional Notes

**Surrogate model accuracy by function (LOO RMSE, week 10)**

| Function | Best surrogate | LOO RMSE | Notes |
|----------|---------------|----------|-------|
| F1 | Proximity search | N/A | GP unreliable at sub-1e-15 precision |
| F2 | Proximity search | 0.187 (GP) | Noisy function - model predictions unreliable |
| F3 | GP | 0.076 | x2 consistently ignored by GP (large length scale) |
| F4 | SVR | 0.125 | Region window improves LOO from 1.437 |
| F5 | GP | 320 | High RMSE reflects complex landscape |
| F6 | SVR | 0.213 | GP hitting length scale bound on x1 |
| F7 | HEBO | 0.411 (GP) | HEBO input warping outperforms standard GP |
| F8 | GP | 0.141 | x8 length scale hitting 1000 - likely categorical |

**Key references**

- Gebru et al. (2018) - Datasheets for Datasets framework
- Jones et al. (1998) - Expected Improvement acquisition function
- Rasmussen & Williams (2006) - Gaussian Processes for Machine Learning
- Srinivas et al. (2010) - UCB acquisition function
- Brochu, Cora & de Freitas (2010) - Bayesian Optimisation tutorial
- Cowen-Rivers et al. (2022) - HEBO
- Bogunovic et al. (2016) - Dynamic Bayesian Optimisation
- Turner et al. (2021) - NeurIPS 2020 BBO Challenge
