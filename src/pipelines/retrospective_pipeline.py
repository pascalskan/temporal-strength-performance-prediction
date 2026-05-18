from src.evaluation.evaluation import feature_set_comparison, run_statistical_evaluation, run_subgroup_analysis
from src.pipelines.data_preparation import prepare_modelling_data
from src.analysis.feature_importance import run_feature_importance_analysis
from src.models.training import train_models
from src.models.baselines import run_naive_baseline
from src.models.traditional_runner import run_traditional_models
from src.evaluation.reporting import generate_model_report
from src.analysis.lift_analysis import run_lift_specific_models, plot_lift_comparison
from src.config.constants import MIN_SUBGROUP_ROWS
from src.config.features import LEAKAGE_COLUMNS
from src.core.logging import get_logger
from src.io.paths import ProjectPaths
from src.reproducibility.environment import capture_experiment_metadata
from src.reproducibility.metadata import save_metadata

logger = get_logger(__name__)

def run_pipeline(df, group_name):
    logger.info("Processing retrospective pipeline for group: %s", group_name)

    if df.empty:
        logger.warning("Dataset for group '%s' is empty. Skipping pipeline execution.", group_name)
        return None

    paths = {
        "group": ProjectPaths.retrospective_group_dir(group_name),
        "traditional": ProjectPaths.retrospective_traditional_dir(group_name),
        "feature_importance": ProjectPaths.retrospective_importance_dir(group_name),
        "evaluation": ProjectPaths.retrospective_evaluation_dir(group_name)
    }

    metadata = capture_experiment_metadata(
        df=df,
        dataset_name=group_name,
        execution_mode="retrospective",
        random_seed=42  # Project standard baseline seed
    )
    save_metadata(metadata, paths["group"] / "run_metadata.json")

    prep = prepare_modelling_data(df)
    df_model_no_attempts = prep.df_model_no_attempts

    # LIFT-SPECIFIC ANALYSIS
    df_lift_full = prep.df.drop(columns=LEAKAGE_COLUMNS, errors="ignore").copy()
    df_lift_full["Best3SquatKg"] = df_lift_full[["Squat1Kg", "Squat2Kg", "Squat3Kg"]].max(axis=1)
    df_lift_full["Best3BenchKg"] = df_lift_full[["Bench1Kg", "Bench2Kg", "Bench3Kg"]].max(axis=1)
    df_lift_full["Best3DeadliftKg"] = df_lift_full[["Deadlift1Kg", "Deadlift2Kg", "Deadlift3Kg"]].max(axis=1)
    lift_results = run_lift_specific_models(df_lift_full, prep.feature_cols, paths["group"])
    if not lift_results.empty:
        plot_lift_comparison(lift_results, paths["group"])

    # NAIVE BASELINE
    naive_results = run_naive_baseline(df_model_no_attempts=df_model_no_attempts, yb_test=prep.yb_test, split_idx=prep.split_idx)

    # TRAIN ML MODELS
    training_results = train_models(
        X_train=prep.X_train, X_test=prep.X_test, y_train=prep.y_train, y_test=prep.y_test,
        X_train_scaled=prep.X_train_scaled, X_test_scaled=prep.X_test_scaled,
        Xb_train_scaled=prep.Xb_train_scaled, Xb_test_scaled=prep.Xb_test_scaled,
        yb_train=prep.yb_train, yb_test=prep.yb_test
    )

    # REPORTING
    generate_model_report(
        naive_results=naive_results,
        training_results=training_results,
        evaluation_dir=paths["evaluation"]
    )

    # FEATURE IMPORTANCE
    run_feature_importance_analysis(
        df_model_no_attempts=df_model_no_attempts,
        rf_model=training_results.rf_model,
        gb_model=training_results.gb_model,
        X=prep.X, y=prep.y,
        feature_cols=prep.feature_cols,
        evaluation_dir=paths["evaluation"],
        importance_dir=paths["feature_importance"]
    )

    # TRADITIONAL MODELS
    traditional_results = run_traditional_models(df_model=prep.df_model, trad_dir=paths["traditional"])

    # EVALUATION
    results_table = run_statistical_evaluation(
        prep.df_model,
        training_results=training_results,
        save_dir=paths["evaluation"],
        traditional_results=traditional_results
    )

    # FEATURE SET COMPARISON
    feature_set_comparison(df_model_no_attempts, prep.feature_cols, paths["evaluation"])

    # SUBGROUP ANALYSIS
    if len(prep.df_model) > MIN_SUBGROUP_ROWS:
        run_subgroup_analysis(prep.df_model, save_dir=paths["evaluation"])
    else:
        logger.warning("Skipping subgroup analysis (dataset too small for group: %s)", group_name)

    return results_table
