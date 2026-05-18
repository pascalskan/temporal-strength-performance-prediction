# Powerlifting Performance Prediction Using Machine Learning

## Overview

This project investigates the use of machine learning techniques to predict competitive powerlifting performance using historical athlete competition data obtained from the OpenPowerlifting dataset.

The research compares traditional strength estimation methods against multiple machine learning approaches, including:

* Linear Regression
* Random Forest Regression
* Gradient Boosting Regression

The project additionally evaluates:

* retrospective prediction performance
* forward prediction capability
* feature engineering effectiveness
* model stability
* subgroup behaviour
* performance across strength levels

The work was completed as part of the CSC3094 Major Project and Dissertation module.

---

# Project Objectives

The primary objectives of the project were:

1. Evaluate whether machine learning models can accurately predict powerlifting totals.
2. Compare engineered ML models against traditional strength estimation formulas.
3. Investigate the impact of historical performance features on prediction accuracy.
4. Assess model generalisation using forward prediction.
5. Evaluate model robustness, statistical significance, and stability.

---

# Dataset

The project uses the OpenPowerlifting dataset containing competitive powerlifting results.

Dataset features include:

* athlete demographics
* bodyweight
* competition dates
* equipment categories
* individual lift attempts
* total competition results

Two equipment categories were analysed:

* Raw
* Wraps

---

# Project Structure

```text
project/
│
├── data/
│   └── openpowerlifting.csv
│
├── results/
│   ├── raw/
│   └── wraps/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   ├── pipelines/
│   └── utils/
│
├── main.py
├── run_all.py
├── README.md
├── requirements.txt
```

---

# Methodology

## Data Preprocessing

The preprocessing stage included:

* missing value removal
* competition filtering
* equipment filtering
* athlete history construction
* chronological sorting
* leakage prevention

A time-aware train/test split was used throughout the project to prevent future information leakage.

---

## Feature Engineering

Several engineered features were developed to improve predictive performance, including:

### Historical Features

* Previous competition total
* Rolling mean
* Rolling standard deviation
* Personal best progression

### Progression Features

* Improvement rate
* Momentum

### Experience Features

* Competition count
* Career length
* Experience density

### Consistency Features

* Coefficient of variation

### Athlete Features

* Age
* Age squared
* Bodyweight
* Sex

---

# Models

## Traditional Methods

* Epley Formula
* Brzycki Formula

## Baseline Model

* Linear Regression using basic demographic features

## Machine Learning Models

* Linear Regression (engineered features)
* Random Forest Regression
* Gradient Boosting Regression

Hyperparameter tuning was performed using GridSearchCV.

---

# Evaluation

Models were evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score

Additional evaluation included:

* cross-validation
* confidence intervals
* paired t-tests
* effect size analysis
* subgroup analysis
* strength-level analysis
* forward prediction evaluation
* feature importance analysis
* residual analysis

---

# Forward Prediction

The project implemented a forward prediction pipeline designed to predict an athlete’s next competition performance using only historical data available prior to the prediction date.

This provided a more realistic evaluation of real-world predictive capability.

---

# Reproducibility

A global random seed was used throughout the project to improve reproducibility.

```python
np.random.seed(42)
```

The project also uses:

* time-aware splitting
* deterministic preprocessing
* fixed hyperparameter search spaces

---

# Continuous Integration

This project uses GitHub Actions for continuous integration. The CI pipeline automatically validates the engineering integrity of the codebase on every push and pull request to the `main` branch.

The workflow performs the following checks:
1.  **Dependency Installation**: Installs all project dependencies from `requirements.txt`.
2.  **Code Compilation**: Verifies that all Python source files compile successfully.
3.  **Automated Testing**: Runs the full `pytest` suite to check for regressions in contracts, metrics, and data handling logic.
4.  **Import Validation**: Confirms that all core library modules are importable.

This automated process ensures that the software remains robust and supports the scientific reproducibility of the research by guarding against accidental breakages.

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Full Experiment Pipeline

```bash
python run_all.py
```

This executes:

1. retrospective modelling
2. forward prediction pipeline
3. statistical evaluation
4. analysis plot generation

---

# Key Outputs

Generated outputs include:

* model evaluation tables
* feature importance plots
* residual plots
* subgroup analysis
* forward prediction metrics
* stability analysis
* comparison visualisations

Outputs are stored inside:

```text
results/
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* SciPy
* tqdm

---

# Dissertation Context

This repository accompanies the dissertation:

> "Powerlifting Performance Prediction Using Machine Learning"

submitted for:

* CSC3094 Major Project and Dissertation
* Newcastle University

---

# Ethical Considerations

The project used publicly available competition data and did not involve direct human participation, intervention, or collection of sensitive personal information beyond publicly accessible records.

---

# Author

**Pascal Skannavis**

Newcastle University
CSC3094 Major Project and Dissertation
