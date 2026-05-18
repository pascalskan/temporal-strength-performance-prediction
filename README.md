# Temporal Strength Performance Prediction

[![Python CI](https://github.com/pascalskan/temporal-strength-performance-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/pascalskan/temporal-strength-performance-prediction/actions/workflows/ci.yml)

A research engineering project investigating whether machine learning models provide more reliable forward prediction of competitive strength performance than traditional deterministic estimation methods.

This repository implements a reproducible experimental framework for comparing traditional strength estimation formulas, statistical baselines, and machine learning models under both retrospective and temporally consistent forward evaluation conditions.

---

## Overview

Predicting athletic performance is a common applied machine learning problem, but evaluation methodology often introduces misleading conclusions.

Traditional strength estimation methods such as **Epley** and **Brzycki** are widely used for estimating performance from known lifting data, while machine learning approaches are frequently reported as outperforming classical methods.

However, retrospective evaluation frameworks often allow structural leakage, where models effectively reconstruct known outcomes rather than genuinely predict unseen performance.

This project investigates that problem directly.

Core question:

> Do machine learning models genuinely outperform traditional strength estimation methods when evaluated under realistic forward prediction conditions?

---

## Key Contributions

This repository implements:

- leakage-aware predictive modelling
- temporally consistent forward evaluation
- deterministic traditional baseline implementation
- statistical baseline comparison
- machine learning regression pipelines
- engineered longitudinal athlete feature generation
- reproducible experiment orchestration
- automated testing and CI validation
- structured result analysis

Key engineering characteristics:

- modular `src/` architecture
- typed contracts / structured interfaces
- deterministic validation fixtures
- runtime validation safeguards
- reproducibility-oriented experiment execution
- automated CI testing pipeline

---

## Methodology

### Modelling Approaches

Implemented models:

#### Traditional deterministic methods
- Epley formula
- Brzycki formula

#### Statistical baseline
- Linear Regression

#### Machine learning models
- Linear Regression (engineered features)
- Random Forest Regression
- Gradient Boosting Regression

---

### Feature Engineering

The predictive framework incorporates:

**Demographic features**
- age
- sex
- bodyweight

**Historical performance features**
- previous competition totals
- rolling means
- rolling standard deviation
- progression trends
- momentum indicators

**Experience features**
- competition count
- competition history depth
- athlete progression metrics

**Consistency features**
- performance variability
- coefficient of variation

---

### Evaluation Frameworks

Two evaluation paradigms are implemented.

#### Retrospective Evaluation

Standard modelling approach where training and testing occur within the same temporal context.

Purpose:
- benchmark comparison
- conventional ML evaluation

Limitation:
- vulnerable to reconstruction bias / leakage

---

#### Forward Prediction Evaluation

Temporal split framework where models predict future athlete performance using only historical information.

Purpose:
- realistic predictive assessment
- leakage-aware evaluation
- genuine generalisation testing

This represents the project's core scientific contribution.

---

## Repository Architecture

```text
temporal-strength-performance-prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
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
├── validation/
│
├── main.py
├── run_all.py
├── requirements.txt
└── README.md
```

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

Run full experimental pipeline:

```bash
python run_all.py
```

This executes:

- preprocessing
- feature engineering
- retrospective evaluation
- forward prediction evaluation
- statistical comparison
- analysis generation
- result export

---

## Testing

Run test suite:

```bash
pytest
```

Compile validation:

```bash
python -m compileall src main.py
```

Import smoke validation:

```bash
python -c "import src.analysis; import src.config; import src.contracts; import src.core; import src.data; import src.evaluation; import src.features; import src.io; import src.models; import src.pipelines; import src.reproducibility; import src.utils"
```

---

## Continuous Integration

GitHub Actions CI automatically validates:

- dependency installation
- Python compilation
- automated tests
- core module imports

Triggered on:

- push to `main`
- pull requests

This helps preserve engineering integrity and reproducibility.

---

## Dataset

This project uses competition data derived from the **OpenPowerlifting** dataset.

Tracked repository contents intentionally exclude:

- raw full datasets
- generated experiment outputs
- cached artifacts
- local virtual environments

This keeps the repository lightweight, reproducible, and version-control appropriate.

---

## Reproducibility

Reproducibility considerations include:

- deterministic validation fixtures
- controlled preprocessing
- leakage-aware temporal splitting
- consistent evaluation metrics
- CI-backed validation

Note:

Full experimental reproducibility depends on access to the same source dataset and environment dependencies.

---

## Key Findings

Core findings from the research:

- traditional deterministic methods perform extremely well under retrospective evaluation
- this performance is largely attributable to structural reconstruction rather than genuine prediction
- machine learning models show stronger forward predictive robustness
- feature engineering contributes more to predictive performance than model complexity
- temporal evaluation methodology fundamentally changes model rankings

This demonstrates that evaluation design is a critical component of predictive modelling.

---

## Limitations

Current limitations include:

- dependence on competition-level public data
- absence of physiological / training load variables
- short-horizon forward prediction
- limited subgroup balance
- classical ML focus (no deep sequence modelling)

Future extensions could include:

- recurrent neural networks
- transformer-based temporal modelling
- uncertainty-aware prediction
- richer athlete context features

---

## Research Context

This repository originated from an undergraduate dissertation research project investigating predictive modelling in competitive strength performance.

Its current form has been engineered as a reproducible research software artifact.

---

## Citation

If this repository contributes to academic or applied work, citation guidance will be provided via `CITATION.cff`.

---

## License

License to be added.

Recommended: MIT License.

---

## Author

**Pascal Skannavis**  
Machine Learning / Software Engineering

---

## Repository Purpose

This repository serves as:

- research engineering portfolio work
- reproducible ML experimentation artifact
- predictive modelling case study
- temporal evaluation methodology demonstration