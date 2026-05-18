RANDOM_STATE = 42
TIME_SPLIT_RATIO = 0.8

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