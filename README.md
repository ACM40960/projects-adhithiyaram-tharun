[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-bKyY6qM)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23964140&assignment_repo_type=AssignmentRepo)

# Predicting Formula 1 Race Outcomes: A Machine Learning and Monte Carlo Simulation Approach

**Module:** ACM40960 – Project in Maths Modelling
**Authors:** Adhithiyaram Ramakrishnan (25204180), Tharun Subramanya Sendil (25208175)
**Institution:** University College Dublin

---

## Overview

This repository contains the full implementation for the group project
submitted for ACM40960. The objective is to build a
reproducible, end-to-end pipeline for predicting Formula 1 race outcomes,
combining:

1. a leakage-safe machine learning classifier trained on pre-race
   information only (grid position, driver and constructor form, circuit
   characteristics),
2. a lap-by-lap in-race simulation informed by tyre degradation modelling,
   and
3. a Monte Carlo simulation layer that propagates model and outcome
   uncertainty through to season-level championship probabilities.

The design of this pipeline is informed by the accompanying literature
review, in particular the leakage-safe feature construction of Alahmadi et
al. (2026), the state-space tyre degradation model of Cappello and Hoegh
(2026), and the uncertainty-propagation methodology of Demsyn-Jones (2019).
Full discussion is provided in the accompanying literature review
submitted for this module.

## Seasons and Regulatory Context

Formula 1 has undergone substantial regulatory and competitive change
across the period covered by this project, which directly shapes how the
data is split and how features are constructed. A full account of the
technical, power unit, constructor, and driver changes across 2022–2026,
and their specific modelling implications, is provided in
[`docs/market_analysis.md`](docs/market_analysis.md). In summary:

**2022–2025 (training and validation).** A single, stable regulatory
generation: ground-effect aerodynamics (reintroduced 2022) and the 2014
power unit formula, both unchanged through 2025. Seasons 2022–2024 are
used for training; 2025 is held out as a validation set within the same
regulatory generation, so performance can be assessed without the
confound of a rules change. Several constructors changed name within this
window while remaining the same operational entity (e.g. Alfa Romeo →
Sauber → Kick Sauber), and some drivers changed teams (e.g. Lewis
Hamilton, Mercedes → Ferrari for 2025); both are handled explicitly in the
pipeline rather than left implicit.

**2026 (prediction target).** The largest single-season regulatory change
in over a decade: a new power unit formula (50/50 combustion/electric
split, MGU-H removed), a new chassis, and active aerodynamics replacing
DRS, alongside power unit supplier changes for several teams and the
arrival of an eleventh constructor, Cadillac. Because the car and power
unit regulations reset simultaneously, constructor-level performance from
2022–2025 cannot be assumed to transfer directly into 2026. This is
encoded in `src/config.py` via `REGULATION_ERA` (tags each season's
regulatory generation) and `CONSTRUCTOR_NAME_MAP` (normalises constructor
identity across rebrands, e.g. Kick Sauber → Audi), and is treated as an
explicit source of prediction uncertainty rather than a gap to be
smoothed over.

## Current Progress

This submission covers **data acquisition and exploratory analysis**,
**feature engineering**, a **machine learning classifier**, and a
**tyre degradation model**:

- Automated retrieval of race data for the 2022–2026 seasons via the
  `fastf1` API, with local caching to ensure reproducibility and to
  respect API rate limits.
- A structured data pipeline that tags each record with its regulation
  era and normalises constructor identity across rebrands; season role
  (training/validation/prediction) is determined by `src/config.py`.
- Exploratory analysis validating data integrity (missingness, coverage,
  grid–finish relationship) and visualising constructor performance
  across the regulatory boundary.
- A leakage-safe feature table (one row per driver per race) built
  strictly from information available before each race: rolling and
  exponentially-weighted driver and constructor form, season-progress
  indicators, and explicit handling of cold-start (new constructor) and
  non-standard grid (pit-lane start) cases.
- A pre-race classifier predicting top-10 finish and podium (top-3)
  outcomes, comparing Logistic Regression, Random Forest, Extra Trees,
  and XGBoost under a temporal train/validation split, evaluated with
  row-level classification metrics and a custom per-race precision@k
  ranking metric.
- A lap-by-lap tyre degradation model, informed by Cappello and Hoegh
  (2026): a linear baseline model (ordinary least squares) used to
  validate the feature construction, followed by a Bayesian state-space
  version fit via MCMC, extended with skewed, heavy-tailed observation
  noise per the source paper's best-performing model. Fit and validated
  across one driver's full 2022–2025 race history.

Subsequent stages (single-race simulation, Monte Carlo season
simulation, final validation) are outlined under **Project Plan** below
and will be developed in subsequent submissions.

## Repository Structure

```
f1_race_predictor/
├── src/
│   ├── config.py          # Centralised configuration (paths, seasons, constants)
│   ├── data_fetch.py      # Data acquisition and caching (fastf1 -> tidy CSV)
│   ├── features.py        # Leakage-safe feature engineering
│   ├── model.py           # Classifier comparison, evaluation, feature importance
│   └── tyre_model.py      # Tyre degradation modelling (OLS baseline + Bayesian state-space)
├── scripts/
│   ├── run_eda.py            # Exploratory data analysis and diagnostic plots
│   ├── build_features.py     # Builds the leakage-safe feature table
│   ├── train_classifier.py   # Trains and evaluates the pre-race classifiers
│   ├── fetch_laps.py         # Fetches lap-by-lap timing data for tyre modelling
│   └── fit_tyre_model.py     # Fits and evaluates the tyre degradation models
├── docs/
│   └── market_analysis.md  # Regulatory and competitive landscape, 2022-2026
├── data/
│   ├── raw/                # Retrieved race data (not version-controlled)
│   └── processed/          # Derived tables and figures (not version-controlled)
├── notebooks/               # Exploratory notebooks
├── requirements.txt
└── README.md
```

Raw and processed data are excluded from version control (see
`.gitignore`) and are regenerated by running the pipeline described below,
in line with standard practice for reproducible data science projects.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Setup and Usage

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Retrieve and cache race data (2022-2026 seasons)
python -m src.data_fetch

# 4. Run exploratory data analysis
python -m scripts.run_eda

# 5. Build the leakage-safe feature table
python -m scripts.build_features

# 6. Train and evaluate the pre-race classifiers
python -m scripts.train_classifier

# 7. Fetch lap-by-lap timing data (large, slow; required for tyre modelling)
python -m scripts.fetch_laps

# 8. Fit and evaluate the tyre degradation models
python -m scripts.fit_tyre_model
```

Running `run_eda` produces a console summary of data structure,
completeness, and coverage by season and regulation era, and saves
diagnostic figures (grid vs. finish position by era, points distribution,
and constructor count by season) to `data/processed/figures/`.

Running `build_features` produces `data/processed/features.csv`, a
leakage-safe feature table with one row per driver per race.

Running `train_classifier` trains Logistic Regression, Random Forest,
Extra Trees, and XGBoost on the top-10 and podium targets, and prints
row-level metrics, per-race precision@k, and Random Forest feature
importance for each target.

Running `fetch_laps` retrieves lap-by-lap timing data (compound, tyre
age, pit stops, track status) for 2022–2026, saved per-race to allow
safe resuming if interrupted by the data source's hourly rate limit.

Running `fit_tyre_model` fits the OLS baseline tyre degradation model
and the Bayesian state-space model (MCMC via PyMC), and prints fitted
coefficients, posterior summaries, and convergence diagnostics.

## Project Plan

| Stage | Description | Status |
|---|---|---|
| 1 | Data pipeline and exploratory analysis | Complete |
| 2 | Feature engineering (leakage-safe, pre-race only) | Complete |
| 3 | Machine learning classifier (top-10 and podium prediction) | Complete |
| 4 | Tyre degradation and single-race simulation | Complete |
| 5 | Monte Carlo season simulation and calibration analysis | Planned |
| 6 | Validation against 2026 season and final report | Planned |

## Data Source and Acknowledgements

Historical and current race data is retrieved via the open-source
[`fastf1`](https://github.com/theOehrly/Fast-F1) Python package, which
provides access to official F1 timing data. `fastf1` is unofficial software
and is not affiliated with the Formula 1 group of companies.

## References

Full citations and critical discussion are provided in the accompanying
literature review submitted for this module. Key sources informing this
pipeline:

- Alahmadi, L. et al. (2026). *Predicting Formula One race outcomes using
  supervised machine learning.* IEEE CSNT 2026.
- Cappello, C. and Hoegh, A. (2026). *A state-space approach to modeling
  tire degradation in Formula 1 racing.* Journal of Sports Analytics.
- Demsyn-Jones, R. (2019). *Misadventures in Monte Carlo.* Journal of
  Sports Analytics.
