# Black-Box Optimisation Capstone Project

## Project Overview

The purpose of this project is to model a real-world machine learning challenge — optimising unknown functions using limited information. The objective is to demonstrate how optimisation is approached in research, using exploration, evidence and iterative strategy refinement over 13 weeks.

The project follows a Bayesian optimisation framework in which eight unknown black-box functions must be maximised. The equations and internal logic of these functions are not known. Only the output of each query is returned, making this a true black-box setting.

I work in front office technology systems which generate high volumes of data. Learning and applying machine learning techniques and building surrogate models directly supports my professional goal of aligning with strategies required as industries adopt AI at pace.

---

## Functions

Each function takes an array of continuous values between 0 and 1 and returns a scalar output. One query per function is submitted each week.

| Function | Input Dimensions | Output | Description |
|----------|-----------------|--------|-------------|
| F1 | 2D | 1D | Radiation field detection — locate a point source |
| F2 | 2D | 1D | Noisy log-likelihood — maximise under noise |
| F3 | 3D | 1D | Drug discovery — minimise side effects (framed as maximisation) |
| F4 | 4D | 1D | Warehouse optimisation — dynamic function, landscape shifts weekly |
| F5 | 4D | 1D | Chemical process yield — unimodal, single peak |
| F6 | 5D | 1D | Cake recipe — all outputs negative, maximise toward zero |
| F7 | 6D | 1D | ML hyperparameter tuning — suspected log-scale dimensions |
| F8 | 8D | 1D | ML hyperparameter tuning — suspected encoded categorical variables |

Submissions are formatted as hyphen-separated values correct to 6 decimal places, for example:
```
0.416484-0.379587-0.523628
```

---

## Technical Approach

### Early Phase (Weeks 1–3): Global Exploration
Initial queries used Latin Hypercube Sampling across the full [0,1]^d space to establish a baseline understanding of each function's landscape. A Gaussian Process with RBF kernel and UCB acquisition function was used to explore broadly and avoid premature commitment to local optima.

### Mid Phase (Weeks 4–7): Exploitation of Confirmed Regions
As promising regions were identified, the strategy shifted toward exploitation. Expected Improvement replaced UCB as the primary acquisition function. Confirmed ranges were tightened per function based on accumulated evidence. Leave-One-Out (LOO) cross-validation was introduced to select the most reliable surrogate model each week.

### Advanced Phase (Weeks 8–10): Specialist Techniques
Several advanced techniques were introduced as limitations of the standard GP became apparent:

- **HEBO** (NeurIPS 2020 BBO Challenge winner) introduced for F7 — handles suspected log-scale hyperparameters through per-dimension input warping
- **Optuna CMA-ES** introduced for F3, F5, F8 — tight local exploitation anchored to confirmed best point
- **Dynamic BO region window** introduced for F4 — filters to observations within confirmed positive region rather than fitting on full dynamic history
- **UCB exploration** applied to F5 after competitor analysis revealed a missed global optimum — confirmed a new peak region returning Y=4039 vs previous best of 3006
- **Weighted centroid** model-free candidate added for F6 — inspired by competitor analysis showing model-free approaches can outperform GP in certain regimes
- **Signed log transform** applied to F1 Y values — compresses 17 orders of magnitude to a scale the GP can model meaningfully

### Key Lessons
- Premature exploitation on F5 caused 6 weeks of queries in a local optimum — high early returns anchored the search incorrectly
- Dynamic functions (F4) require specialised windowing — fitting on full history includes stale landscape configurations
- LOO cross-validation provides objective, reproducible surrogate selection — critical for principled decision-making
- Range validation and plausibility checks prevent surrogate hallucinations from influencing submissions

---

## Results Summary (Week 10)

| Function | Best Y | Strategy that achieved it |
|----------|--------|--------------------------|
| F1 | 7.89e-15 | Proximity search, asymmetric x2 bias |
| F2 | 0.7392 | Proximity search near confirmed peak |
| F3 | −0.00600 | SVR uniform, fixed x1/x2, lower x3 |
| F4 | 0.7097 | SVR with Dynamic BO region window |
| F5 | 4039.81 | UCB exploration of new region |
| F6 | −0.1615 | SVR with tightened confirmed ranges |
| F7 | 2.9071 | HEBO with tightened x2/x5/x6 bounds |
| F8 | 9.8799 | CMA-ES anchored to confirmed best |

---

## Repository Structure

```
/
├── README.md               — This file
├── DATASHEET.md            — Dataset documentation (Gebru et al. 2018 framework)
├── MODEL_CARD.md           — Model documentation (Mitchell et al. 2019 framework)
├── notebooks/              — Weekly Jupyter notebooks with analysis and submissions
├── scripts/                — Python scripts for each function by week
└── data/                   — Input/output records by function
```

---

## Documentation

- [Dataset Datasheet](DATASHEET.md) — Motivation, composition, collection process, preprocessing, distribution and maintenance
- [Model Card](MODEL_CARD.md) — Overview, intended use, strategy details, performance, assumptions and limitations, ethical considerations

---

## References

- Jones et al. (1998) — Expected Improvement acquisition function
- Rasmussen & Williams (2006) — Gaussian Processes for Machine Learning
- Srinivas et al. (2010) — UCB acquisition function
- Brochu, Cora & de Freitas (2010) — Bayesian Optimisation tutorial
- Hutter et al. (2011) — Random Forest surrogates
- Snoek et al. (2012) — Practical Bayesian Optimisation
- McKay et al. (1979) — Latin Hypercube Sampling
- Arlot & Celisse (2010) — Leave-One-Out cross-validation
- Cowen-Rivers et al. (2022) — HEBO
- Bogunovic et al. (2016) — Dynamic Bayesian Optimisation
- Turner et al. (2021) — NeurIPS 2020 BBO Challenge
- Gebru et al. (2018) — Datasheets for Datasets
- Mitchell et al. (2019) — Model Cards for Model Reporting
