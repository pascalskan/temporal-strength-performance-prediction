"""
Single definition of the experimental configuration.

Model specifications previously existed in two places: the walk-forward
pipeline and the ablation pipeline each constructed their own. That is a
correctness problem rather than an untidiness one -- if the two drift, the
ablation is measuring different models than the headline evaluation, and the
two tables can no longer be read together. The same was true of the walk-forward
settings and the bootstrap sample count.

Everything an experiment is configured by now lives here, so a reader can see
the full specification in one file rather than reconstructing it from literals
scattered across pipelines.
"""
from typing import Any, Dict, List, Tuple

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.constants import RANDOM_STATE
from src.models.baselines import (
    DriftBaseline,
    PersistenceBaseline,
    RollingMeanBaseline,
)
from src.models.traditional_baselines import BrzyckiBaseline, EpleyBaseline

# ---------------------------------------------------------------------------
# Walk-forward protocol
# ---------------------------------------------------------------------------

# Yearly folds. Finer granularity would give more folds but many would contain
# too few competitions to train on; coarser would lose the temporal resolution
# the drift analysis depends on.
WALK_FORWARD_GRANULARITY = "year"

# Periods required before the first forecast is made. Early folds train on very
# little data and produce unstable estimates; five years is the point at which
# the training window is substantial enough to be worth reporting.
MIN_TRAIN_PERIODS = 5

# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------

# Fixed hyperparameters, applied identically in every protocol. These are
# deliberately untuned: see TUNING_ASYMMETRY_NOTE below.
RANDOM_FOREST_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

GRADIENT_BOOSTING_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "random_state": RANDOM_STATE,
}

RIDGE_PARAMS: Dict[str, Any] = {
    "random_state": RANDOM_STATE,
}


def build_models() -> Dict[str, Any]:
    """
    Learned models, freshly constructed.

    Returns new instances on every call: an evaluator fits these in place, so
    handing out shared instances would carry one fold's fit into the next.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(**RIDGE_PARAMS)),
        ]),
        "Random Forest": RandomForestRegressor(**RANDOM_FOREST_PARAMS),
        "Gradient Boosting": GradientBoostingRegressor(**GRADIENT_BOOSTING_PARAMS),
    }


def build_temporal_baselines() -> Dict[str, Any]:
    """Forecasting baselines that extrapolate recent performance."""
    return {
        "Persistence": PersistenceBaseline(),
        "Rolling Mean": RollingMeanBaseline(),
        "Drift": DriftBaseline(),
    }


def build_traditional_baselines() -> Dict[str, Any]:
    """
    Traditional 1RM equations.

    Scored only on competitions with recorded attempts -- 62.7% of raw records
    and 30.4% of equipped -- so their pooled metrics describe a different
    subpopulation than the other models. The matched-subset output exists to
    make that comparison like-for-like.
    """
    return {
        "Epley": EpleyBaseline(),
        "Brzycki": BrzyckiBaseline(),
    }


def build_all_baselines() -> Dict[str, Any]:
    return {**build_temporal_baselines(), **build_traditional_baselines()}


# ---------------------------------------------------------------------------
# Hyperparameter search (walk-forward tuned variant)
# ---------------------------------------------------------------------------

# Grids are deliberately small. Cost multiplies by grid size times inner folds
# at every selection point across an expanding window, and the purpose here is
# to establish whether tuning changes the conclusion -- not to find a global
# optimum. A grid that cannot finish is worth less than a modest one that does.
#
# Each spans the axis that most affects capacity for its model family, bracketing
# the fixed default so the search can move in either direction.
RANDOM_FOREST_GRID = {
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 5],
}

GRADIENT_BOOSTING_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 4],
}

RIDGE_GRID = {
    "ridge__alpha": [0.1, 1.0, 10.0, 100.0],
}

# Folds between full searches; in between, the last selection is carried
# forward. See src/models/tuned.py for why this is periodic rather than
# per-fold.
RETUNE_EVERY_N_FOLDS = 5


def build_tuned_models() -> Dict[str, Any]:
    """
    The learned models, each selecting its own hyperparameters within the
    training window of the fold it is fitted on.

    Linear Regression is excluded: it has nothing to tune, so wrapping it would
    add cost and an empty selection history for no gain. It still appears in
    the tuned run via the untuned model set, unchanged.
    """
    from src.models.tuned import PeriodicallyTunedRegressor

    return {
        "Linear Regression": LinearRegression(),
        "Ridge": PeriodicallyTunedRegressor(
            estimator=Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(**RIDGE_PARAMS)),
            ]),
            param_grid=RIDGE_GRID,
            retune_every=RETUNE_EVERY_N_FOLDS,
            name="Ridge",
        ),
        "Random Forest": PeriodicallyTunedRegressor(
            estimator=RandomForestRegressor(**RANDOM_FOREST_PARAMS),
            param_grid=RANDOM_FOREST_GRID,
            retune_every=RETUNE_EVERY_N_FOLDS,
            name="Random Forest",
        ),
        "Gradient Boosting": PeriodicallyTunedRegressor(
            estimator=GradientBoostingRegressor(**GRADIENT_BOOSTING_PARAMS),
            param_grid=GRADIENT_BOOSTING_GRID,
            retune_every=RETUNE_EVERY_N_FOLDS,
            name="Gradient Boosting",
        ),
    }


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

# Resamples per bootstrap. 5,000 is enough for stable 95% interval endpoints;
# since the cluster bootstrap was reformulated to work from per-athlete
# aggregates, this is no longer a meaningful share of runtime.
BOOTSTRAP_RESAMPLES = 5000

# Pairs submitted to paired comparison. Chosen to answer specific questions
# rather than to enumerate every pair, which would invite multiplicity problems
# without adding insight:
#   - the strongest temporal baseline against each learned model
#   - the strongest learned model against the strongest traditional equation
MODEL_COMPARISONS: List[Tuple[str, str]] = [
    ("Rolling Mean", "Gradient Boosting"),
    ("Rolling Mean", "Ridge"),
    ("Rolling Mean", "Random Forest"),
    ("Rolling Mean", "Linear Regression"),
    ("Persistence", "Gradient Boosting"),
    ("Rolling Mean", "Brzycki"),
    ("Gradient Boosting", "Brzycki"),
]

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

TUNING_ASYMMETRY_NOTE = """
The walk_forward stage uses fixed hyperparameters; walk_forward_tuned selects
them within each fold's training window by forward-chaining search. Running
both is the point: the difference between them measures what tuning is worth
under this protocol, rather than leaving it to be asserted either way.

The standalone retrospective pipeline also tunes. That asymmetry is deliberate
but bounded, and matters when reading the tables:

  - The headline protocol contrast (forward/metrics_retro.csv against
    forward/metrics.csv) is unaffected. Both arms are built from the same
    factories with no tuning on either side, so the difference between them is
    attributable to the protocol alone.

  - The standalone retrospective table (evaluation/model_comparison_full.csv)
    IS tuned, and so should not be placed beside untuned walk-forward numbers
    as though the only difference were the protocol.

Grid search there uses forward-chaining splits rather than K-fold: on
temporally ordered data, K-fold trains on later records to predict earlier
ones, which is the leakage this project exists to avoid, and tuning is not
exempt from it.
"""
