# Publication Positioning Document

This document defines the scientific and methodological identity of the paper derived from the `temporal-strength-performance-prediction` repository. Its purpose is to align the research narrative with the evidence provided by the codebase, enforce strict claim boundaries, and document unresolved scientific gaps.

## 1. Paper Identity

**Methodological Case Study in Applied Forecasting Evaluation.**

This paper uses competitive athlete performance forecasting to demonstrate how the choice of evaluation protocol materially influences measured predictive performance and comparative model conclusions. It highlights that temporally consistent validation can yield substantially different inferences from retrospective evaluation.

## 2. Problem Statement

In applied machine learning literature, models intended for forward forecasting are frequently evaluated using retrospective techniques, such as randomized cross-validation. While retrospective evaluation is statistically valid for static pattern recognition or imputation tasks, it is structurally inappropriate for establishing forward forecasting capability. Applying retrospective evaluation to temporal data can introduce an optimistic bias by implicitly granting models access to future population trends or individual subsequent states.

## 3. Why This Matters Scientifically

The scientific literature in sports analytics and applied forecasting risks accumulating claims of predictive superiority that may not generalize to actual deployment scenarios. By quantifying the performance disparity between retrospective and temporal evaluation protocols on the identical dataset, this work demonstrates that the evaluation framework itself is a **major determinant of measured performance** in assessing model capability.

## 4. Novelty

The primary contribution is a rigorous, quantitative demonstration of the impact of evaluation protocol selection. We provide an open-source, reproducible framework that explicitly measures the performance delta when identical models transition from a retrospective environment to a strictly constrained, expanding-window temporal environment, allowing for an analysis of how comparative model rankings may shift in response.

## 5. Benchmark Philosophy

The benchmark models (Persistence, Rolling Mean, Ridge, Random Forest, Gradient Boosting) are representative archetypes rather than an exhaustive search for state-of-the-art performance. They serve to illustrate how evaluation methodology impacts different classes of algorithms (heuristics vs. linear models vs. complex ensembles).

## 6. Evaluation Philosophy

Our philosophy is that evaluation methodology must match the deployment constraint. Retrospective evaluation is not invalid, but it answers the wrong question for forecasting. The `WalkForwardEvaluator` enforces temporal discipline. By comparing its outputs to standard retrospective outputs, we measure the **protocol-induced discrepancy** introduced by misapplied evaluation protocols.

## 7. Publication Audience

*   **Primary:** Applied machine learning researchers and data scientists focused on time-series evaluation methodology.
*   **Secondary:** Academic reviewers and practitioners in sports analytics and performance science.

## 8. What This Paper Is NOT

*   **It is NOT a claim to have "solved" temporal leakage.** We mitigate a specific structural form of it; subtle leakage vectors may always remain.
*   **It is NOT a definitive benchmark.** We are not claiming these models represent the absolute upper bound of performance.
*   **It is NOT a dismissal of retrospective evaluation.** Retrospective evaluation is a valid tool for certain tasks; it is merely inappropriate for justifying forward-prediction claims.
*   **It is NOT a claim that simple baselines universally outperform machine learning.** The findings are bounded by the specific dataset, feature set, and evaluation protocol implemented.

## 9. Required Scientific Strengthening (Unresolved Gaps)

While the core infrastructure is robust, the following gaps should be addressed to maximize publication defensibility:

*   **Defensible Hyperparameter Optimization:** Currently, ML models use default or arbitrarily chosen hyperparameters. Without rigorous, nested, temporally-safe tuning, claims about relative model performance are vulnerable to the attack that the ML models are simply under-optimized.
*   **Feature Ablation Studies:** The specific contribution of engineered features versus raw history is not systematically isolated.
*   **Multi-Horizon Forecasting:** The current framework predicts only the immediate next event ($t+1$). Expanding to multi-step prediction could provide deeper insights into the comparative advantages of different model classes.
*   **Broader Representative Benchmark Sensitivity Analysis:** The current model set is representative but limited. A broader sensitivity analysis including other classes of temporal models would help assess how these findings apply across different algorithmic approaches.
*   **Data Forensic Audit & Leakage Provenance:** OpenPowerlifting is a living dataset. Retroactive corrections to historical records are a known form of unpreventable leakage. A strict audit of data versioning and provenance is required to bound this risk.
*   **External Validity Tests:** Validating the findings against a completely independent dataset (e.g., a different sport or a holdout cohort not drawn from the same underlying distribution) **would strengthen generalisability claims.**