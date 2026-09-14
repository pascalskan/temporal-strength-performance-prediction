# Temporal Evaluation Methodology in Sports Performance Prediction

[![Python CI](https://github.com/pascalskan/temporal-strength-performance-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/pascalskan/temporal-strength-performance-prediction/actions/workflows/ci.yml)

A research engineering project investigating how evaluation methodology and temporal leakage affect machine learning benchmarking in sports performance prediction.

This repository implements a reproducible, leakage-safe experimental framework for evaluating machine learning models against temporal baselines (Persistence, Rolling Mean, Drift) using an expanding-window walk-forward validation strategy.

---

## Overview

Predicting athletic performance is a common applied machine learning problem, but evaluation methodology often introduces misleading conclusions. 

Historically, machine learning models in sports analytics are evaluated using randomized cross-validation or static train-test splits. These approaches allow for **temporal leakage**, where models implicitly learn future population trends or observe an athlete's future state before predicting their past.

When evaluated retrospectively, machine learning models frequently appear to achieve extraordinary accuracy. However, when subjected to strict, leakage-safe temporal validation, apparent performance drops significantly, and simple baselines often prove highly competitive.

### Core Research Question

> How does strict, leakage-safe temporal validation alter conventional conclusions regarding machine learning superiority in sports performance prediction?

---

## Key Contributions

This repository implements:

- **Leakage-safe temporal validation:** Expanding-window walk-forward forecasting.
- **Forecasting-safe feature engineering:** Features computed strictly from historical windows.
- **Athlete-level temporal baselines:** Persistence, Rolling Mean, and Drift models.
- **Statistical significance testing:** Bootstrap confidence intervals for metric comparison.
- **Reproducible experiment orchestration:** Deterministic pipelines with automated CI validation.

Key engineering characteristics:

- modular `src/` architecture
- typed contracts / structured interfaces
- deterministic validation fixtures
- runtime validation safeguards
- reproducibility-oriented experiment execution
- pinned dependency management
- automated CI testing pipeline
- structured experiment metadata capture

---

## Methodology

### Modelling Approaches

Implemented models:

#### Temporal Forecasting Baselines
- **Persistence:** Predicts the athlete's most recent performance.
- **Rolling Mean:** Predicts the average of the athlete's recent performances.
- **Drift:** Extrapolates the athlete's historical trajectory.

#### Statistical Models
- Linear Regression
- Ridge Regression

#### Machine Learning Models
- Random Forest Regressor
- Gradient Boosting Regressor

---

### Feature Engineering

The predictive framework incorporates forecasting-safe features:

- **Demographics:** age, sex, bodyweight
- **Historical performance:** previous competition totals, rolling means, rolling standard deviation
- **Experience:** competition count, competition history depth
- **Consistency:** performance variability, coefficient of variation

*All features are strictly engineered using only data available prior to the prediction timestamp.*

---

### Evaluation Framework

The core evaluation strategy is **Expanding-Window Walk-Forward Validation**.

Models are trained on an initial historical window (e.g., years 1-5). They then forecast the *immediate next event* for each athlete in year 6. The training window expands to include year 6, and models predict year 7. This process continues iteratively.

Purpose:

- ensures genuine forward-prediction assessment
- prevents structural leakage (future-to-past information flow)
- aligns evaluation with real-world forecasting constraints
- provides robust athlete-level confidence intervals via bootstrapping

This represents the project’s core scientific contribution.

---

## Repository Architecture

```text
temporal-strength-performance-prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── test_fixture.csv
│
├── docs/
│
├── scripts/
│
├── src/
│   ├── analysis/
│   ├── config/
│   ├── contracts/
│   ├── core/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── io/
│   ├── models/
│   ├── pipelines/
│   ├── reproducibility/
│   └── utils/
│
├── tests/
│
├── CITATION.cff
├── LICENSE
├── README.md
├── create_validation_fixture.py
├── main.py
├── pytest.ini
└── requirements.txt
```

Architecture design principles:

- separation of concerns
- explicit package boundaries
- reusable pipeline orchestration
- testable modular components
- reproducibility-first experiment execution

---

## Installation

Clone repository:

```bash
git clone https://github.com/pascalskan/temporal-strength-performance-prediction.git
cd temporal-strength-performance-prediction
```

Create virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start

Run the full experimental pipeline:

```bash
python main.py
```

This executes the complete experimental workflow, including:

- data preprocessing
- temporal feature engineering
- expanding-window walk-forward evaluation
- baseline comparison
- statistical significance testing
- performance analysis
- result generation

---

## Testing

Run test suite:

```bash
pytest -q
```

Current repository validation:

- 25 automated tests passing

Compile validation:

```bash
python -m compileall src main.py
```

Import smoke validation:

```bash
python -c "import src.analysis; import src.config; import src.contracts; import src.core; import src.data; import src.evaluation; import src.features; import src.io; import src.models; import src.pipelines; import src.reproducibility; import src.utils"
```

Testing coverage includes:

- package import integrity
- contract validation
- data cleaning behaviour
- metric correctness
- splitting behaviour
- reproducibility safeguards
- training contract integrity

---

## Continuous Integration

GitHub Actions CI automatically validates:

- dependency installation
- Python compilation
- automated tests
- core package imports

Triggered on:

- push to `main`
- pull requests targeting `main`

This helps preserve engineering integrity, reproducibility, and deployment confidence.

---

## Dataset

This project uses competition data derived from the **OpenPowerlifting** dataset.

Repository contents intentionally exclude:

- raw full datasets
- generated experiment outputs
- cached artifacts
- local virtual environments
- logs
- environment secrets
- machine-specific temporary files

This keeps the repository lightweight, reproducible, and version-control appropriate.

A deterministic validation fixture is included for automated testing and reproducibility verification.

---

## Reproducibility

Reproducibility considerations include:

- pinned dependency versions
- deterministic validation fixtures
- controlled preprocessing
- leakage-aware temporal splitting
- standardised pytest execution
- CI-backed validation
- structured runtime metadata
- environment-aware experiment recording

This repository is designed as a reproducible research software artifact rather than a one-off academic code submission.

Note:

Full experimental reproducibility depends on access to the same source dataset and compatible runtime environment.

---

## Key Findings

Figures are from the Raw cohort (601,835 records, 101,695 forward predictions)
on the dataset snapshot recorded in `data/dataset_manifest.json`.

**Evaluation protocol, not model choice, dominates measured performance.**
Identical models and features, untuned in both arms, protocol the only
difference:

| Model | Retrospective R² | Walk-forward R² | Retrospective MAE | Walk-forward MAE |
|---|---|---|---|---|
| Linear Regression | 0.955 | 0.342 | 22.51 | 105.89 |
| Random Forest | 0.970 | 0.555 | 17.82 | 106.85 |
| Gradient Boosting | 0.965 | 0.570 | 19.45 | 103.37 |

Error inflates five- to six-fold. The effect reproduces in the Equipped cohort
(R² 0.890–0.926 falling to 0.459–0.474).

Note what does *not* cause this. Both arms use chronological splits; the only
difference is whether the target is the same competition or the next one. The
collapse is therefore attributable to target coupling alone, rather than to
randomised splitting.

**Information beats architecture by roughly ten to one.** Feature ablation under
the same walk-forward protocol, best learned model at each step:

| Feature set | Features | Best MAE | Gain from added information | Spread across algorithms |
|---|---|---|---|---|
| Demographics only | 3 | 139.05 | — | 9.73 |
| + previous total | 4 | 108.27 | 30.78 | 5.35 |
| + recent form | 7 | 104.04 | 35.01 | 3.09 |
| Full engineered set | 17 | 103.37 | 35.69 | 3.49 |

A single feature accounts for 86% of the total information gain; the remaining
thirteen contribute 4.91 kg between them. The spread across algorithms narrows
as information improves, so model choice matters least exactly where the data is
richest.

**No learned model beat the simple temporal baselines.** With the full
engineered feature set, the best learned model remains 1.51 kg behind the better
of Persistence (MAE 101.85) and Rolling Mean (102.15).

**Which metric is reported changes the ranking.** Baselines win on MAE while
Gradient Boosting wins on RMSE and R². Persistence has by far the lowest median
error (22.5 kg against 65.9 for Gradient Boosting) and much heavier tails, which
RMSE penalises quadratically and MAE does not. "Do simple baselines match
machine learning?" has no answer independent of the loss function.

**Comparisons against traditional equations must be made on shared data.** Epley
and Brzycki require recorded attempts, available for 59.1% of Raw predictions,
and that subpopulation is systematically easier. On the full population Brzycki
appears to beat Gradient Boosting by 11.7 kg; restricted to observations every
model scored, the gap is 2.1 kg and Persistence overtakes Brzycki. See
`matched_subset/` in any walk-forward output directory.

**Feature attribution is less concentrated than retrospective analysis
suggested.** Under walk-forward the rolling mean accounts for 46.9% of Gradient
Boosting's attribution and 37.1% of Random Forest's — not the 80–90% measured
retrospectively — and six to ten features are needed to reach 80%.

---

## Limitations

Current limitations include:

- dependence on competition-level public data
- absence of physiological / training load variables
- short-horizon forward prediction
- limited subgroup balance

Potential future extensions:

- recurrent neural networks
- transformer-based temporal modelling
- uncertainty-aware prediction
- richer athlete context modelling
- probabilistic forecasting
- sequence-aware progression modelling

---

## Research Context

This repository originated from an undergraduate dissertation research project investigating predictive modelling in competitive strength performance.

Its current form has since been engineered into a reproducible research software artifact suitable for:

- publication-grade methodological evaluation
- research engineering demonstration
- machine learning portfolio presentation
- methodological reproducibility case study

---

## Citation

If this repository contributes to academic or applied work, please cite using the metadata provided in:

```text
CITATION.cff
```

---

## License

This project is licensed under the MIT License.

See:

```text
LICENSE
```

---

## Author

**Pascal Skannavis**  
Machine Learning / Software Engineering

---

## Repository Purpose

This repository serves as:

- research engineering portfolio work
- reproducible machine learning experimentation
- predictive modelling case study
- temporal evaluation methodology demonstration
- research software engineering showcase