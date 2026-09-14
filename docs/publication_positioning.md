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

## 9. Scientific Strengthening

### 9.1 Addressed

*   **Feature Ablation.** Nested feature sets — demographics, plus last result, plus recent form, full engineered set — are each evaluated through a complete walk-forward run, so the gain from adding information is measured against the gain from changing algorithm under the same protocol. Previously the features-versus-complexity claim rested on a retrospective measurement, which is the protocol this work argues is invalid for such claims.
*   **Data Provenance.** Every run fingerprints its input (SHA-256, row count, date range, equipment composition) and compares it against a committed manifest, recording the digest in the run metadata. A retroactive correction to a single historical record leaves row and column counts unchanged, so a content digest is the only thing that detects it. This does not prevent retroactive revision — nothing can — but it bounds the risk by making the snapshot behind any result identifiable.
*   **Temporally Safe Tuning.** Hyperparameter search used K-fold on chronologically ordered data, selecting parameters using later records to predict earlier ones. This was the project's own central leakage mechanism operating inside its tuning step. Replaced with forward-chaining splits.
*   **Equipment Cohort Viability.** The equipped cohort was previously unusable (~11 forward predictions) because Single-ply was filtered out before the cohort split ran. It now yields ~103,000 forward predictions, so equipped is a second properly powered evaluation domain rather than an exploratory footnote.
*   **Traditional Equations Under Walk-Forward.** Epley and Brzycki are now evaluated under the primary protocol, not only the retrospective and forward ones. Because they require recorded attempts — present for 62.7% of raw records and 30.4% of equipped — a matched-subset comparison restricts every model to the observations all models scored.
*   **Subgroup Robustness and Feature Attribution.** Both now run under walk-forward, with sample sizes reported alongside every subgroup and attribution spread retained across folds.

### 9.2 Remaining

*   **Hyperparameter Optimisation Under Walk-Forward.** Tuning is now temporally safe where it occurs, but the walk-forward and forward protocols still use fixed hyperparameters. The headline protocol contrast is unaffected — both of its arms are untuned — but the claim that complex models add little over simple baselines remains open to the objection that the complex models were never optimised. Nested tuning inside each fold is the defensible fix and is expensive.
*   **Multi-Horizon Forecasting.** The framework predicts only the next event ($t+1$). Since the central finding is that short-term continuity carries most of the signal, testing whether that advantage decays at longer horizons is the most direct challenge to it.
*   **Broader Model Sensitivity.** The model set is representative rather than exhaustive. Sequence models in particular are absent, and their exclusion bounds any claim about what temporal modelling can achieve here.
*   **External Validity.** All results come from one dataset. Replication against an independent cohort or a different sport would separate findings about forecasting evaluation from findings about OpenPowerlifting.

### 9.3 Discrepancies Between the Dissertation and This Codebase

Recorded so that the paper describes what the code does rather than repeating the dissertation:

*   The dissertation names `run_all.py` as the central execution script (§4.9). No such file exists; the entry point is `main.py`.
*   The dissertation describes retrospective evaluation as using a **random** train/test split (Appendices B.2.4, E.8.5.1). It does not. Both arms use chronological splits, and what differs is the target: the same competition versus the next one. This makes the finding stronger and more surgical than the dissertation claims — the collapse from R² ≈ 0.96 to ≈ 0.34 occurs with the split strategy held fixed, isolating target coupling as the mechanism rather than confounding it with randomised splitting.
*   The dissertation groups athletes by name, sex and **bodyweight** (§4.2). Bodyweight is not stable across a career; the identifier is now name and sex, with the measured justification recorded in `src/data/identity.py`.
*   The dissertation's Raw-versus-Wraps comparison (§5.6) and Appendix G never state the Wraps sample size. Wraps contributes 2,911 source records and 496 after cleaning. Appendix G's conclusion that Wraps feature importance is "more balanced and realistic" is drawn from that; it should either carry the sample size or be withdrawn.
*   Appendix D states `n_estimators = 100` for both ensembles. The walk-forward configuration uses 200.