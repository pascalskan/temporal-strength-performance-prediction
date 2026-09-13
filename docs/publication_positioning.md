# Publication Positioning Document

This document defines the scientific and methodological identity of the paper derived from the `temporal-strength-performance-prediction` repository. Its purpose is to align the research narrative with the evidence provided by the codebase and prevent claim boundary violations.

## 1. Paper Identity

**Methodological Critique and Blueprint.**

This paper uses sports performance forecasting as a case study to quantitatively demonstrate the degree to which evaluation methodology, not algorithmic complexity, dictates model performance. It is primarily a paper about *how to evaluate models in time-series contexts*, presented through a compelling and accessible real-world example.

## 2. Problem Statement

In applied machine learning, particularly in domains like sports analytics, models are often evaluated using standard retrospective techniques (e.g., randomized k-fold cross-validation). This common practice introduces **temporal leakage**, where information from the future is implicitly made available to the model during training. This leads to misleadingly optimistic performance metrics and a distorted view of model superiority, creating a reproducibility crisis where reported gains fail to translate to real-world forward-prediction scenarios.

## 3. Why This Matters Scientifically

Science requires that we test hypotheses under conditions that match the real world. For forecasting, this means models must only use information available at the time of prediction. By violating this principle, the current standard practice in much of the applied literature is not measuring a model's predictive capability, but rather its ability to reconstruct a known history.

This work matters because it:
a) Provides a **quantitative, reproducible measure** of this methodological error.
b) Presents a **robust, open-source framework** that corrects this error.
c) Re-establishes a more **realistic and defensible performance baseline** for a complex forecasting problem, forcing future work to demonstrate genuine, temporally-sound improvement.

## 4. Novelty

The novelty is **not** the idea that temporal data requires temporal validation. This is known.

The novelty is the **rigorous, end-to-end, and reproducible implementation** that moves this principle from a well-known "gotcha" to a solved problem in this domain. We provide the first open-source framework that allows for a direct, quantitative comparison between leaky and non-leaky evaluation pipelines, complete with statistical significance testing of the resulting performance gap.

## 5. Benchmark Philosophy

The models included in this paper (Persistence, Rolling Mean, Ridge, Random Forest, Gradient Boosting) are not intended to be an exhaustive list of all possible algorithms. Rather, they are chosen as **representative archetypes**:

*   **Temporal Baselines (Persistence, Rolling Mean):** Represent simple, no-cost heuristics.
*   **Statistical Models (Ridge):** Represent a classic, interpretable regression approach.
*   **Machine Learning Ensembles (Random Forest, Gradient Boosting):** Represent complex, non-linear, high-capacity models that are often considered "state-of-the-art" in applied settings.

The purpose of the benchmark is not to crown a "winner," but to use these archetypes to demonstrate how their relative performance rankings **shift dramatically** depending on the evaluation protocol.

## 6. Evaluation Philosophy

**"The evaluation protocol is the most important parameter to tune."**

Our philosophy is that a robust evaluation framework is more critical than the choice of model. The core of this paper is the `WalkForwardEvaluator`, which enforces temporal discipline through an expanding-window design. All metrics derived from this evaluator are considered "true" performance, while metrics from the retrospective pipeline are considered "inflated" or "leaky." We supplement this with athlete-level bootstrapping to generate confidence intervals, acknowledging that point-metric comparisons are insufficient for making credible claims.

## 7. Publication Audience

The primary audience is the **applied machine learning research community**. These are researchers and practitioners who build and evaluate predictive models on time-series data.

The secondary audience includes **sports scientists and data analysts** who can use this work to set more realistic expectations for their own forecasting initiatives.

The paper should be written to be accessible to a technical ML audience while using the sports context as a clear and motivating example.

## 8. What This Paper Is NOT

*   **It is NOT a "which model is best for powerlifting" paper.** It is a paper demonstrating how to properly *ask* that question.
*   **It is NOT a deep dive into the physiology or strategy of powerlifting.** The domain is a model system, not the subject of inquiry itself.
*   **It is NOT a claim that machine learning is "bad" or "useless."** It is a claim that its true performance is often overestimated and must be validated against simple, robust baselines under strict temporal conditions.
*   **It is NOT a perfect, "leakage-proof" solution.** It is a robust and practical framework designed to prevent the most common and egregious forms of temporal leakage in event-based forecasting.