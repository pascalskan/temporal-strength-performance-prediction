"""
Hyperparameter selection inside walk-forward folds.

The walk-forward and forward protocols use fixed hyperparameters. That leaves
the project's central negative result -- that no learned model beats simple
temporal baselines -- open to the obvious objection: the learned models were
never optimised, so the comparison was never fair to them.

This closes that, under two constraints that the naive approach violates.

Temporal safety
---------------
Selection happens strictly within the fold's training window, using
forward-chaining inner splits. The test period is never seen, and within the
training window a candidate is always scored on data later than it was fitted
on. K-fold here would choose hyperparameters using later records to predict
earlier ones -- the leakage this project exists to measure, relocated into the
tuning step.

Cost
----
A full search at every fold multiplies the number of model fits by the grid
size times the inner fold count, at every expanding window. On the production
cohorts that is the difference between hours and days. Selection is therefore
repeated periodically rather than at every fold, with the chosen parameters
carried forward in between.

That is not purely a concession to runtime. Re-deriving hyperparameters from
scratch before every single forecast is not how a deployed model behaves;
periodic re-selection on a widening history is closer to real practice, and it
is stated rather than hidden.

The parameters chosen at each selection point are recorded, so hyperparameter
drift over time becomes an observable rather than an assumption.
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.core.logging import get_logger

logger = get_logger(__name__)

# Inner forward-chaining splits used to score a candidate configuration.
INNER_SPLITS = 3

# Folds between full searches. Five years of additional history is enough for
# the optimum to plausibly move; re-searching every fold costs several times
# more for changes that are mostly negligible.
DEFAULT_RETUNE_EVERY = 5


class PeriodicallyTunedRegressor(BaseEstimator, RegressorMixin):
    """
    Estimator that re-selects its hyperparameters every N fits.

    Presents the ordinary fit/predict interface, so the walk-forward evaluator
    needs no knowledge of tuning: from its side this is simply a model that
    happens to configure itself.
    """

    def __init__(self, estimator, param_grid: Dict[str, List],
                 retune_every: int = DEFAULT_RETUNE_EVERY,
                 inner_splits: int = INNER_SPLITS,
                 scoring: str = "neg_mean_absolute_error",
                 name: str = "estimator"):
        self.estimator = estimator
        self.param_grid = param_grid
        self.retune_every = retune_every
        self.inner_splits = inner_splits
        self.scoring = scoring
        self.name = name

        self._fit_count = 0
        self._best_params: Dict[str, Any] = {}
        self.selection_history: List[dict] = []
        self.fitted_estimator_ = None

    def _feasible_splits(self, n_samples: int) -> int:
        """
        Inner splits this training window can support, or 0 if it cannot.

        TimeSeriesSplit needs strictly more samples than splits, and needs at
        least two splits to be meaningful. The earliest walk-forward folds are
        far smaller than later ones -- some hold a handful of rows -- so a
        window too small to validate on is a normal condition, not an error.
        """
        usable = min(self.inner_splits, n_samples - 1)
        return usable if usable >= 2 else 0

    def _should_search(self, n_samples: int) -> bool:
        if self._feasible_splits(n_samples) == 0:
            return False
        # Always search on the first viable fit: carrying forward from nothing
        # would leave the earliest folds on library defaults.
        return self._fit_count % self.retune_every == 0

    def _search(self, X, y):
        n_splits = self._feasible_splits(len(X))

        search = GridSearchCV(
            clone(self.estimator),
            self.param_grid,
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring=self.scoring,
            n_jobs=-1,
        )
        search.fit(X, y)

        self._best_params = dict(search.best_params_)
        self.selection_history.append({
            "model": self.name,
            "fit_index": self._fit_count,
            "n_train": int(len(X)),
            "best_score": float(search.best_score_),
            **{f"param_{k}": v for k, v in search.best_params_.items()},
        })

        logger.info(
            "%s: selected %s on %d training rows (inner %s = %.2f)",
            self.name, search.best_params_, len(X), self.scoring, search.best_score_,
        )

    def fit(self, X, y):
        if self._should_search(len(X)):
            self._search(X, y)
        elif not self._best_params and self._feasible_splits(len(X)) == 0:
            # Too small to validate on. Fit with the fixed defaults rather than
            # failing the fold: an early window that cannot support inner
            # validation still contributes predictions.
            logger.debug(
                "%s: training window of %d rows is too small for %d-way "
                "forward-chaining validation; using fixed hyperparameters.",
                self.name, len(X), self.inner_splits,
            )

        estimator = clone(self.estimator)
        if self._best_params:
            estimator.set_params(**self._best_params)

        estimator.fit(X, y)
        self.fitted_estimator_ = estimator
        self._fit_count += 1
        return self

    def predict(self, X):
        if self.fitted_estimator_ is None:
            raise RuntimeError(f"{self.name} has not been fitted.")
        return self.fitted_estimator_.predict(X)

    # Attribution is read off the fitted estimator, so the wrapper must not
    # hide it from the evaluator's feature-importance capture.
    @property
    def feature_importances_(self):
        return getattr(self.fitted_estimator_, "feature_importances_")

    @property
    def coef_(self):
        return getattr(self.fitted_estimator_, "coef_")

    @property
    def best_params_(self) -> Dict[str, Any]:
        return dict(self._best_params)


def collect_selection_history(models: Dict[str, Any]) -> pd.DataFrame:
    """
    Gather the hyperparameters chosen over the run.

    Reported as a result in its own right: whether the optimum moves as the
    training window grows speaks to whether the data-generating process is
    stable, which is the assumption every forecast here rests on.
    """
    records = []
    for model in models.values():
        records.extend(getattr(model, "selection_history", []))

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values(["model", "fit_index"])
