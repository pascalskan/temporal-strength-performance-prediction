# Temporal Evaluation Methodology in Sports Performance Prediction (Strength Forecasting Dissertation and Research Software)

## Basics
- Type: Study (final-year B.Sc. Computer Science dissertation), later extended into a personal research-software project.
  - Source: `docs/source_material/Dissertation Pascal Skannavis.pdf` title page; `README.md` "Research Context".
- My role/title: Author and sole developer. The dissertation cover lists Pascal Skannavis as author with Alma Cantu as supervisor. The git history shows one author (`pascalskan`).
  - Many commits carry a `Co-Authored-By: Claude Opus 5` trailer, so AI assistance was used in the post-dissertation engineering phase. The extent of that assistance is not documented beyond the trailers. See "Gaps".
- Organisation/client (if any): University (name NOT FOUND in the files read). Student ID and supervisor appear on the poster and dissertation cover.
- Dates or duration (best estimate, approximate):
  - Proposal, poster and dissertation: file dates show the proposal PDF modified 6 May 2026 / 18 May 2026 (its own timestamps), the poster 17 May 2026, and the dissertation 6 May 2026. The dissertation cover says "May 2026".
  - Software repository: first commit 2026-05-18. A second work phase ran 2026-09-13 to 2026-09-14 (31 commits on the branch `feature/scientific-evaluation-phase`).
  - Overall: roughly May 2026 to September 2026. The proposal start date is NOT FOUND.
- Status: Ongoing / paused (not confirmed). The main branch is at 2026-05-18. The feature branch holds the September work. It is 31 commits ahead of `main`, so it has not been merged into it.

## One-line summary
A dissertation and reproducible Python research project showing that how you evaluate a model (retrospective versus strict forward-in-time testing) changes conclusions about machine learning accuracy when predicting powerlifters' competition totals.

## The problem or goal
- Dissertation goal (from the abstract): evaluate whether machine learning models predict strength performance more accurately and generalisably than traditional 1RM formulas (Epley, Brzycki) under realistic predictive conditions.
  - Source: `Dissertation Pascal Skannavis.pdf`, abstract.
- Repository goal (from the README): study how evaluation methodology and temporal leakage affect machine learning benchmarking in sports performance prediction. The stated research question is "How does strict, leakage-safe temporal validation alter conventional conclusions regarding machine learning superiority in sports performance prediction?"
  - Source: `README.md`.
- Intended audience: applied ML and time-series evaluation researchers, and sports-analytics practitioners.
  - Source: `docs/publication_positioning.md` §7. It describes a planned paper. No paper submission or publication is evidenced.
- Data: the public OpenPowerlifting competition dataset (1,446,977 records, 1966-02-27 to 2026-03-22).
  - Source: `data/dataset_manifest.json`, `docs/results_summary.md`.

## What I did
Only items evidenced in files or git history are listed. Solo versus assisted work is flagged.

**Dissertation phase (May 2026)**
- Wrote a 62-page, 16,034-word dissertation comparing traditional formulas, linear regression, Random Forest and Gradient Boosting. Source: PDF cover and page count.
- Wrote the project proposal and produced a research poster with a model comparison chart. Sources: `dissertation proposal.pdf`, `Pascal_Skannavis_230280128_Poster.pdf`.
- Built the original experimental pipeline (data cleaning, feature engineering, retrospective and forward evaluation). Source: dissertation Chapter 4 table of contents; initial commit `19116fe`.

**Engineering and scientific-extension phase (September 2026; AI co-authorship trailers present on these commits)**
- Designed and built an expanding-window walk-forward validation framework. Features are recomputed inside each fold from past data only, and predictions are keyed per observation. Source: commit `f89f636`, `src/evaluation/walk_forward.py`.
- Added forecasting-safe features and three temporal baselines (Persistence, Rolling Mean, Drift). Source: `src/features/temporal.py`, `src/models/baselines.py`.
- Implemented a paired, athlete-level cluster bootstrap for model comparisons, with Holm-Bonferroni multiplicity correction. Source: `src/evaluation/statistics.py`, commits `1abf096` and `01d47a8`.
- Rewrote the bootstrap so it computes from per-athlete aggregates. The commit message reports a 5000-resample comparison dropping from an estimated ~5 hours to ~11 seconds at production size, with results exact to ~1e-15. Source: commit `20b4cbb`.
- Audited and corrected the athlete-identity key (Name + Sex), with measurements of the alternatives on the Raw cohort. Added an age-reversal check to detect merged identities. Source: commits `9a87ed4`, `4cc869f`, `src/data/identity.py`.
- Found and fixed a filter bug that had silently left the "equipped" cohort at 496 records (~11 forward predictions). Fixing it gave 412,934 records and about 103,000 walk-forward predictions. Source: commit `004364f`.
- Replaced K-fold hyperparameter search with forward-chaining splits. Added a separate walk-forward stage that selects hyperparameters inside each fold's training window. Source: commits `26d4e69`, `6821489`, `src/models/tuning.py`, `src/models/tuned.py`.
- Built feature ablation, subgroup, feature-attribution and traditional-equation (Epley/Brzycki) analyses under walk-forward, with matched-subset comparison. Source: commits `87b47d7`, `f80c92d`.
- Added reproducibility infrastructure:
  - dataset SHA-256 fingerprinting and a committed manifest (`2d1bc96`);
  - determinism tests (`26d4e69`);
  - resumable, code-version-keyed checkpointing (`2e19d70`, `cf0d810`);
  - a results manifest and summary (`01d47a8`);
  - CI that runs every pipeline stage end to end on a fixture (`2d1bc96`).
- Packaged the project (`pyproject.toml`, version 0.2.0), added `CITATION.cff` and an MIT licence, and rewrote the README and positioning document so that claims match regenerated results. Source: commits `2e19d70`, `4bde618`.
- Found and corrected discrepancies between the dissertation and the code. Examples: the README had claimed random splits; the retrospective arm actually used chronological splits. Source: `docs/publication_positioning.md` §9.3.

## Outcomes and results
All figures are from files in the repo. They are for the Raw cohort on one dataset snapshot unless stated.

| Result | Source |
|---|---|
| Same models and features, evaluation protocol the only difference: R² falls from 0.955–0.970 (retrospective) to 0.342–0.570 (walk-forward). MAE rises about five- to six-fold (e.g. Gradient Boosting 19.45 to 103.37 kg). Equipped cohort: R² 0.890–0.926 falls to 0.459–0.474. | `README.md` Key Findings; `docs/results_summary.md` |
| Walk-forward predictions evaluated: 101,695 (Raw), 103,095 (Equipped). | `docs/results_summary.md` |
| No learned model beat the best simple baseline on MAE: Persistence 101.85 and Rolling Mean 102.15, against Gradient Boosting 103.37 and Ridge 105.11. | `docs/results_summary.md` |
| Metric choice changes the ranking: Gradient Boosting wins on RMSE (144.56) and R² (0.570), while baselines win on MAE. | `docs/results_summary.md` |
| Feature ablation: one feature (previous total) accounts for 86% of the total information gain (30.78 kg of 35.69 kg). Spread across algorithms narrows from 9.73 to 3.49 kg as features are added. | `README.md`; `docs/results_summary.md` ablation table |
| Traditional equations scored only a subset of predictions (59.1% of Raw forward predictions). On the full population Brzycki appears to beat Gradient Boosting by 11.7 kg. On observations every model scored the gap is 2.1 kg, and Persistence overtakes Brzycki. | `README.md`; commit `01d47a8` |
| Retrospective analysis suggested rolling mean carries 80–90% of feature importance. Under walk-forward it is 46.9% for Gradient Boosting and 37.1% for Random Forest. | `README.md` |
| Tuning inside walk-forward folds improves models without changing the conclusion (fixture run: Gradient Boosting gains 4.84 kg and Ridge 1.11 kg; best tuned learned model still trails the best baseline by ~10 kg). | commit `6821489`. **Fixture result only; no production tuned result found in `docs/results_summary.md`.** |
| Test suite: 217 test functions across 32 test files, plus an end-to-end CI run of every stage on a fixture. | `tests/` (counted by me). A pass count was **NOT run in this session**; README's "25 tests passing" is outdated. |
| Codebase size: about 10,500 lines of Python across `src`, `scripts` and `tests`. | counted from files |
| Results index: 147 result files checksummed in `docs/results_manifest.json`. | commit `01d47a8` message |
| Runtimes noted: walk-forward ~50 min (Raw) and ~2 h (Equipped); ablation up to ~8 h (Equipped, estimated). | `main.py` help text; commit `cf0d810` |
| Dissertation grade, awards, feedback, citations, downloads or users: **NOT FOUND**. | — |

Caveat on the results above: the dissertation abstract says ML models show "stronger generalisation and more consistent predictive accuracy" than traditional formulas. The later repository work reports that learned models do not beat simple temporal baselines on MAE under walk-forward. These address different comparisons (formulas versus ML, and baselines versus ML). Present them carefully and do not blend them.

## Skills and tools
- Languages and libraries: Python 3.12, pandas 2.2.2, NumPy 1.26.4, scikit-learn 1.5.1, SciPy 1.13.1, matplotlib 3.9.0, tqdm. Source: `pyproject.toml`.
- Models: Linear and Ridge regression, Random Forest, Gradient Boosting, Epley and Brzycki 1RM equations, Persistence, Rolling-Mean and Drift forecasting baselines.
- Methods:
  - walk-forward / expanding-window validation;
  - temporal-leakage prevention;
  - forward-chaining hyperparameter search;
  - athlete-level cluster bootstrap with confidence intervals;
  - Holm-Bonferroni correction;
  - calibration and residual diagnostics;
  - feature ablation and attribution;
  - subgroup analysis;
  - data-provenance auditing and forensic data-quality audits.
- Software engineering:
  - modular package architecture (`src/` with 12 subpackages);
  - typed contracts;
  - pytest;
  - GitHub Actions CI;
  - deterministic seeds and a determinism test;
  - checkpoint/resume;
  - dependency pinning;
  - CITATION.cff and semantic versioning;
  - command-line stage and cohort selection (argparse).
- Professional skills evidenced: research design, critical self-review of claims, technical writing (README, positioning document, dissertation, poster), and large-dataset handling (~307 MB CSV).
- Git and GitHub for version control and CI.

## Key deliverables
- `Dissertation Pascal Skannavis.pdf` (`docs/source_material/`): 62 pages, B.Sc. dissertation, May 2026.
- `dissertation proposal.pdf` and `Pascal_Skannavis_230280128_Poster.pdf` (`docs/source_material/`): the project proposal and the research poster.
- `main.py` plus `src/`: the full experimental framework (pipelines for retrospective, forward, walk-forward, walk-forward tuned, and ablation).
- `tests/` and `.github/workflows/ci.yml`: automated validation.
- `README.md`: project documentation and key findings.
- `docs/publication_positioning.md`: the planned paper's framing and claim boundaries, with addressed and remaining scientific gaps.
- `docs/results_summary.md` and `docs/results_manifest.json`: committed headline result tables and a checksummed results index.
- `data/dataset_manifest.json` and `data/test_fixture.csv`: dataset fingerprint and deterministic test fixture.
- `scripts/`: manifest recording, output verification and metric auditing.
- `CITATION.cff` and `LICENSE` (MIT): citation metadata and licence.
- `results/`: generated output tree and forensic-audit plots (git-ignored, local only).

## Challenges and how I handled them
1. **Silent bugs that unit tests could not see.** An equipment filter had emptied the equipped cohort (~11 forward predictions), and two pipeline stages could not run at all after a refactor. The unit suite stayed green throughout. I fixed each defect, centralised the cohort definitions, and made CI run every stage end to end on a fixture. Sources: commits `004364f`, `37546ed`, `2d1bc96`.
2. **Compute cost and reproducibility on a 600k-row cohort.**
   - The bootstrap was projected at ~5 hours per comparison and was rewritten to run in about 11 seconds, exact to ~1e-15.
   - Multi-hour runs risked being lost, so I added checkpoint/resume keyed to configuration and code version.
   - OpenPowerlifting changes over time, so I fingerprinted the dataset with SHA-256 and a committed manifest.
   - Sources: commits `20b4cbb`, `2e19d70`, `cf0d810`, `2d1bc96`.
3. **Fairness and validity of comparisons.**
   - The traditional equations only score a non-random subset of rows, so I added matched-subset comparisons.
   - Fourteen significance tests were reported unadjusted, so I added Holm-Bonferroni.
   - Hyperparameter search had used K-fold on time-ordered data, so I replaced it with forward-chaining splits.
   - Earlier README claims were corrected to match regenerated results.
   - Sources: commits `01d47a8`, `26d4e69`, `4bde618`.

## Best material for the website
- **Files that could be shown publicly (subject to the sensitive list below):**
  - `README.md` (public-facing, already on GitHub).
  - `docs/publication_positioning.md` and `docs/results_summary.md`.
  - `Pascal_Skannavis_230280128_Poster.pdf`: it contains the student number, so redact or confirm first. The poster text says "Preliminary Results" and uses k-fold CV, so it predates the walk-forward work.
  - Generated figures in `results/forensic_audit/`: `cleaning_attrition_by_rule.png`, `competition_counts_histogram.png`, `cumulative_eligibility_curve.png`, `raw_missingness_by_column.png`.
  - Plots and tables under `results/openpowerlifting/` (not individually reviewed).
  - A diagram of the architecture could be derived from `README.md` (the repository layout) or `src/`. The dissertation's Figure 4.1 (system architecture) is also an option.
- **Quotes, testimonials, feedback:** NOT FOUND. The dissertation acknowledgements are generic and unattributed.
- **Links:**
  - GitHub repository: https://github.com/pascalskan/temporal-strength-performance-prediction
  - CI badge: in `README.md`.
  - Live site or published paper: NOT FOUND.
  - Whether the repository is public: appears intended to be (the README is written for public readers and the `.gitignore` was repaired because it is "a public repository"). Not independently verified.

## Sensitive or confidential items
- **Personal identifiers:**
  - Student number on the poster, dissertation cover and proposal (`docs/source_material/*.pdf`). Do not publish the number.
  - Supervisor's name (Alma Cantu) appears in the poster and dissertation. Get their permission before naming them publicly.
- **Private individuals / personal data:**
  - `data/openpowerlifting.csv` (~307 MB) is public competition data but contains named athletes (including birth-year classes). Do not publish raw rows. It is git-ignored.
  - `results/` and `data/test_fixture.csv` may contain athlete names or per-athlete predictions. The fixture and results were not checked in detail. Confirm before showing any per-athlete output.
- **Unpublished or academic-integrity material:**
  - The full dissertation and proposal PDFs are in `docs/source_material/`. They are deliberately git-ignored (`.gitignore` rules were fixed to keep them off the public repo). Publishing them is the author's decision (university rules, similarity-checking, assessment status).
- **Internal-only:**
  - `tracked_files.txt` (local scratch) and `.idea/` (IDE settings).
  - Local working copies `venv/` and `.venv/`.
- **Credentials/keys:** none found.
- **Financial or employer/client confidential information:** none found.
- **Contact details:** the git configuration holds an email address. It is not included in this summary.

## Gaps and questions for me
- University name, degree classification or grade, and submission or marking outcome are not in any file. Do you want a result stated?
- Exact project start date, and which commits or work were done before 2026-05-18, since the initial commit is a single bulk import.
- How much of the September engineering work (the 31 branch commits) was done by you versus AI-assisted. Many commits carry `Co-Authored-By: Claude Opus 5`. Decide how to describe this honestly on the site.
- Is the dissertation's "ML generalises better" conclusion still your position, given the later finding that baselines match learned models under walk-forward? How should the site frame the evolution?
- Has the feature branch been merged to `main`? Is the repository public? Has a paper been drafted or submitted? (`publication_positioning.md` describes a planned paper only.)
- The test suite was not run during this review. The 217 count is of test functions, and a current pass or fail result is not confirmed. The README states "25 automated tests passing", which is out of date.
- No production result was found for the tuned walk-forward stage. Only the fixture result (in a commit message) exists. Was it run on the full dataset?
- The grade, supervisor feedback and any presentation or poster-session outcome are not recorded.
- The dissertation's poster says supervisor "Alma Cantu"; confirm the preferred display name and permission.
- `docs/results_summary.md` reports Brzycki coverage as 59.1% of Raw forward predictions, while `docs/publication_positioning.md` and a commit message cite 62.7% of raw records. These are different denominators (predictions versus records). Check which to use.
