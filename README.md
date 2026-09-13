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

Core findings from the research:

- **Retrospective leakage artificially inflates ML performance:** When evaluated using randomized splits, ML models show extraordinary accuracy due to implicit structural leakage.
- **Temporal baselines are highly competitive:** Under strict walk-forward evaluation, simple baselines like Rolling Mean and Persistence perform remarkably well, challenging the presumed necessity of complex ML for athlete forecasting.
- **Evaluation methodology determines the outcome:** The comparative ranking of models shifts dramatically when transitioning from retrospective to temporally consistent evaluation frameworks.

This demonstrates that evaluation design is fundamentally more impactful than model selection in applied predictive modelling.

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