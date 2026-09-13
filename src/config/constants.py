RANDOM_STATE = 42
TIME_SPLIT_RATIO = 0.8

# ---------------------------------------------------------------------------
# Equipment cohorts
#
# Single source of truth. These were previously defined in three places --
# the cleaning filter and one hardcoded list in each entrypoint -- which
# allowed them to disagree: cleaning admitted only {Raw, Wraps} while the
# entrypoints selected {Wraps, Single-ply} for the equipped cohort. Single-ply
# was therefore removed before the entrypoints ever looked for it, and the
# equipped cohort silently collapsed to Wraps alone (496 records after
# cleaning, ~11 forward predictions).
#
# Composition of the equipped cohort on the current dataset:
#   Single-ply  412,438 records (99.88%)
#   Wraps           496 records ( 0.12%)
#
# "Equipped" is the standard term for competition with supportive equipment,
# and both categories qualify. But the cohort is Single-ply in all but name:
# Wraps cannot materially affect any aggregate estimate, so this is not a
# merge that gains statistical power. Report the composition rather than
# describing it as combining two comparable subgroups.
#
# The surviving Wraps records are also a heavily selected subsample: 81.7% of
# source Wraps rows lack Age and are dropped during cleaning (2,911 -> 496),
# which is why their mean TotalKg (533kg) sits far above Single-ply (417kg).
# Do not read that gap as an equipment effect.
# ---------------------------------------------------------------------------
RAW_EQUIPMENT = frozenset({"Raw"})
EQUIPPED_EQUIPMENT = frozenset({"Single-ply", "Wraps"})

# Everything the pipeline is willing to model. Deriving this from the cohort
# definitions is what prevents the filter and the cohorts drifting apart.
VALID_EQUIPMENT = RAW_EQUIPMENT | EQUIPPED_EQUIPMENT

SMALL_DATASET_THRESHOLD = 100
MIN_ANALYSIS_ROWS = 30
MIN_SUBGROUP_ROWS = 100

RF_DEFAULT_PARAMS = {
    "n_estimators": 50,
    "max_depth": 10,
    "random_state": RANDOM_STATE
}

GB_DEFAULT_PARAMS = {
    "n_estimators": 50,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": RANDOM_STATE
}

RF_SMALL_GRID = {
    "n_estimators": [50],
    "max_depth": [5, None]
}

RF_FULL_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

GB_SMALL_GRID = {
    "n_estimators": [50],
    "learning_rate": [0.1],
    "max_depth": [2]
}

GB_FULL_GRID = {
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3]
}